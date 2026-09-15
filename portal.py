# -*- coding: utf-8 -*-
r"""
统一启动器（门户）
- 左侧菜单自动发现 tools/*/manifest.json，加新工具只需建文件夹 + manifest.json
- 点击菜单：工具未运行则自动用当前 python 环境拉起，右侧 iframe 加载工具页面
- 每个工具也可用各自文件夹里的 run.bat 独立启动
端口：7860
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

from flask import Flask, jsonify, render_template_string, request

PORT = 7860
START_TIMEOUT = 120  # 秒（img2prompt 首次 import torch 较慢）

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(ROOT, "tools")

# --------------------------------------------------------------------------
# Windows Job Object：把门户拉起的子进程全部挂到一个 Job 上，并设置
# JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE —— 门户进程退出（点 X、Ctrl+C、崩溃）
# 时系统会自动杀掉 Job 里的所有子进程，避免孤儿工具进程残留。
# --------------------------------------------------------------------------
_JOB = None


def _setup_job_object():
    if os.name != "nt":
        return None

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT_INFO(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class EXTENDED_LIMIT_INFO(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT_INFO),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    h = k32.CreateJobObjectW(None, None)
    if not h:
        return None
    info = EXTENDED_LIMIT_INFO()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)):
        k32.CloseHandle(h)
        return None
    return h


def load_tools():
    """扫描 tools/*/manifest.json，返回工具列表（保持文件夹名字母序）"""
    tools = []
    if not os.path.isdir(TOOLS_DIR):
        return tools
    for name in sorted(os.listdir(TOOLS_DIR)):
        manifest = os.path.join(TOOLS_DIR, name, "manifest.json")
        if not os.path.isfile(manifest):
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tools.append({
                "id": cfg.get("id") or name,
                "group": cfg.get("group") or "",
                "name": cfg.get("name") or name,
                "icon": cfg.get("icon") or "🧩",
                "desc": cfg.get("desc", ""),
                "port": int(cfg["port"]),
                "script": cfg.get("script", "app.py"),
                "dir": os.path.join(TOOLS_DIR, name),
            })
        except Exception as e:
            print(f"[portal] 读取 {manifest} 失败：{e}")
    return tools


TOOLS = load_tools()
TOOL_BY_ID = {t["id"]: t for t in TOOLS}

# 外部工具：不是本地服务，直接内嵌第三方网页（这些网页没有本地 ping/启停概念）
EXTERNAL_TOOLS = [
    {
        "id": "danbooru_tags",
        "group": "外部工具",
        "name": "Danbooru 标签超市",
        "icon": "🛒",
        "desc": "按 Danbooru 标签组合查询并导出提示词",
        "url": "https://tags.novelai.dev/",
    },
]


def ping_tool(port, timeout=0.6):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def spawn_tool(tool):
    """以隐藏窗口方式拉起工具进程，日志写入 logs/<id>.log"""
    log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, f"{tool['id']}.log"), "w", encoding="utf-8")
    env = {**os.environ, "TOOL_NO_BROWSER": "1"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, os.path.join(tool["dir"], tool["script"])],
        cwd=tool["dir"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    log.close()
    # 挂到 Job：门户进程退出时系统自动连带杀掉所有子进程
    if _JOB:
        try:
            ctypes.windll.kernel32.AssignProcessToJobObject(_JOB, int(proc._handle))
        except Exception:
            pass
    return proc


def start_tool(tool):
    """确保工具在运行：已运行直接返回，否则拉起并等待就绪"""
    if ping_tool(tool["port"], timeout=1.0):
        return True, "already-running"

    script = os.path.join(tool["dir"], tool["script"])
    if not os.path.isfile(script):
        return False, f"找不到脚本：{script}"

    # 统一环境：所有工具都用门户自己的 python 解释器
    spawn_tool(tool)
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        time.sleep(1.0)
        if ping_tool(tool["port"], timeout=1.0):
            return True, "started"
    return False, f"启动超时（{START_TIMEOUT}s），请查看 logs/{tool['id']}.log"


def kill_stale_listeners():
    """门户启动前，清掉占用本套工具端口的历史残留进程（孤儿工具、旧门户），
    保证本次拉起的所有进程都由当前门户创建并可被 Job Object 连带回收。"""
    ports = {PORT} | {t["port"] for t in TOOLS}
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
            local = parts[1]
            if ":" in local and local.rsplit(":", 1)[-1].isdigit():
                if int(local.rsplit(":", 1)[-1]) in ports:
                    pids.add(int(parts[4]))
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=10)
    if pids:
        print(f"[portal] 已清理历史残留进程：{len(pids)} 个（旧门户/孤儿工具）")
        time.sleep(0.5)


app = Flask(__name__)


@app.after_request
def no_store(resp):
    # 禁止浏览器缓存门户页面，保证更新代码后立即生效
    resp.headers["Cache-Control"] = "no-store"
    return resp


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comfyui工具箱</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body.light {
    --bg:#f0f2f5; --surface:#ffffff; --border:#e4e6eb; --text:#1c1e21;
    --hover:#f0f2f5; --title:#0b5bb5; --btn:#e4e6eb; --btn-hover:#d8dbe0;
    --dot-off:#c8ccd0; --code-bg:#f0f2f5;
    color-scheme: light;
  }
  body.dark {
    --bg:#1e1f22; --surface:#29292b; --border:#3a3b40; --text:#e6e8ee;
    --hover:#3a3b40; --title:#74b9ff; --btn:#3f4147; --btn-hover:#4d5057;
    --dot-off:#555555; --code-bg:#1e1f22;
    color-scheme: dark;
  }
  body { margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); color:var(--text);
         height:100vh; display:flex; flex-direction:column; overflow:hidden; }

  /* 顶栏（固定不动） */
  .topbar { height:52px; flex-shrink:0; display:flex; align-items:center; gap:14px;
            padding:0 18px; background:var(--surface); border-bottom:1px solid var(--border); }
  .topbar .title { font-size:15px; font-weight:600; color:var(--title); }
  .topbar .dot { width:9px; height:9px; border-radius:50%; background:var(--dot-off); }
  .topbar .dot.on { background:#2ecc71; box-shadow:0 0 6px rgba(46,204,113,.6); }
  .topbar .spacer { flex:1; }
  .topbar button { padding:7px 14px; border:none; border-radius:8px; background:var(--btn);
                   color:var(--text); font-size:13px; cursor:pointer; }
  .topbar button:hover { background:var(--btn-hover); }
  .topbar button.primary { background:#6c5ce7; color:#fff; }
  .topbar button.primary:hover { background:#7d6ef0; }
  .topbar button.icon-btn { font-size:16px; padding:6px 12px; }

  .main { flex:1; display:flex; min-height:0; }

  /* 左侧菜单（宽度可通过右缘拖拽条调整，范围 150-400px） */
  .sidebar { width:var(--sbw, 210px); flex-shrink:0; background:var(--surface); border-right:1px solid var(--border);
             display:flex; flex-direction:column; padding:12px 8px; overflow:hidden;
             transition:width .25s ease, padding .25s ease; }
  .resizer { width:5px; flex-shrink:0; cursor:col-resize; background:transparent; transition:background .15s; }
  .resizer:hover, .resizer.dragging { background:#6c5ce7; }

  /* 折叠态：不完全隐藏，保留 56px 图标条 */
  body.collapsed .sidebar { width:56px; padding-left:6px; padding-right:6px; }
  body.collapsed .resizer { display:none; }
  body.collapsed .item { justify-content:center; padding:10px 0; }
  body.collapsed .item .txt, body.collapsed .item .st { display:none; }
  body.collapsed .sidebar .foot { display:none; }
  .menu { flex:1; display:flex; flex-direction:column; gap:6px; }
  /* 分组（可展开/收起） */
  .group-head { display:flex; align-items:center; gap:8px; padding:9px 10px; cursor:pointer;
                font-size:13px; font-weight:600; border-radius:8px; user-select:none; }
  .group-head:hover { background:var(--hover); }
  .group-head .arrow { font-size:10px; opacity:.6; transition:transform .2s; flex-shrink:0;
                       transform:rotate(90deg); }   /* 展开：尖朝正下方 */
  .group.closed .arrow { transform:rotate(0deg); }  /* 收起：▶ 朝右 */
  .group-head .gname { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .group-tools { display:grid; grid-template-rows:1fr; transition:grid-template-rows .25s ease; }
  .group.closed .group-tools { grid-template-rows:0fr; }
  .group-tools-inner { overflow:hidden; min-height:0; display:flex; flex-direction:column; gap:4px; }
  /* 侧栏折叠成图标条时：分组只显示图标且强制展开 */
  body.collapsed .group-head { justify-content:center; padding:8px 0; pointer-events:none; }
  body.collapsed .group-head .arrow, body.collapsed .group-head .gname { display:none; }
  body.collapsed .group-tools { grid-template-rows:1fr !important; }
  .item { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:8px;
          cursor:pointer; font-size:13.5px; border:1px solid transparent; }
  .item:hover { background:var(--hover); }
  .item.active { background:#6c5ce7; color:#fff; }
  .item.active .txt .d { opacity:.8; }
  .item .icon { font-size:17px; }
  .item .txt { flex:1; min-width:0; }
  .item .txt .n { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .item .txt .d { font-size:11px; opacity:.55; white-space:nowrap; overflow:hidden;
                  text-overflow:ellipsis; margin-top:1px; }
  .item .st { width:7px; height:7px; border-radius:50%; background:var(--dot-off); flex-shrink:0; }
  .item .st.on { background:#2ecc71; }
  .sidebar .foot { padding:10px 12px; font-size:11px; opacity:.45; line-height:1.6;
                   border-top:1px solid var(--border); }

  /* 右侧内容 */
  .content { flex:1; min-width:0; display:flex; flex-direction:column; position:relative; }
  .content iframe { flex:1; border:none; width:100%; background:var(--bg);
                    opacity:0; transition:opacity .25s ease; }
  .content iframe.ready { opacity:1; }
  .welcome { flex:1; display:flex; align-items:center; justify-content:center; }
  .welcome .box { max-width:520px; background:var(--surface); border:1px solid var(--border); border-radius:14px;
                  padding:32px 36px; line-height:1.9; font-size:14px; box-shadow:0 1px 4px rgba(0,0,0,.05); }
  .welcome h2 { margin:0 0 12px; color:var(--title); font-size:19px; }
  .welcome code { background:var(--code-bg); padding:2px 7px; border-radius:5px; font-size:12.5px; }
  .loading { position:absolute; top:70px; left:50%; transform:translateX(-50%);
             background:#6c5ce7; color:#fff; font-size:13px; padding:8px 18px; border-radius:20px;
             display:none; box-shadow:0 4px 16px rgba(0,0,0,.15); }
</style>
</head>
<body>
  <script>document.body.className = localStorage.getItem("portalTheme") || "light";</script>
  <div class="topbar">
    <button class="icon-btn" id="collapseBtn" onclick="toggleSidebar()" title="折叠/展开侧边栏">☰</button>
    <span class="title" id="toolName">Comfyui工具箱</span>
    <span class="dot" id="toolDot"></span>
    <span class="spacer"></span>
    <button class="icon-btn" id="themeBtn" onclick="toggleTheme()" title="切换深色/浅色主题">💡</button>
    <button onclick="refreshStatus()">刷新状态</button>
    <button class="primary" id="newWinBtn" onclick="openNewWindow()" style="display:none">新窗口打开</button>
  </div>
  <div class="main">
    <div class="sidebar">
      <div class="menu" id="menu"></div>
      <div class="foot">加新工具：<br>tools\ 下建文件夹<br>放入 app.py + manifest.json<br>重启门户自动出现</div>
    </div>
    <div class="resizer" id="resizer" title="拖拽调整侧边栏宽度"></div>
    <div class="content" id="content">
      <div class="welcome" id="welcome">
        <div class="box">
          <h2>欢迎使用工具箱</h2>
          启动器启动时已自动逐个检查并拉起所有工具。<br>
          点击左侧菜单即可使用（各自独立端口，互不影响）。<br>
          某个工具中途被关闭也没关系，点菜单会自动重新拉起。<br>
          也可以不通过门户，直接运行工具文件夹里的 <code>run.bat</code> 单独使用。
        </div>
      </div>
      <div class="loading" id="loading">⏳ 正在启动工具，首次加载较慢…</div>
      <iframe id="frame" style="display:none"></iframe>
    </div>
  </div>
<script>
const MENU = {{ tools_json }};
const TOOLS = MENU.tools;                                  // 本地工具（有 port，可启停）
const EX = MENU.external.map(t => Object.assign({ external: true }, t));
const ALL_TOOLS = TOOLS.concat(EX);
let currentId = null;

/* 按分组聚合菜单（本地工具的 manifest.group + 外部工具），保持出现顺序 */
const GROUP_ICONS = { "本地工具箱": "🧰", "外部工具": "🌐" };
const GROUPS = [];
const groupMap = {};
function addToGroup(gname, tool) {
    if (!(gname in groupMap)) {
        const g = { name: gname, tools: [] };
        groupMap[gname] = g;
        GROUPS.push(g);
    }
    groupMap[gname].tools.push(tool);
}
TOOLS.forEach(t => addToGroup(t.group || "其他", t));
EX.forEach(t => addToGroup(t.group || "其他", t));

/* 全局主题：门户为总开关，切换时向 iframe 广播；工具页内的灯泡也能反向同步 */
let globalTheme = localStorage.getItem("portalTheme") || "light";
function applyPortalTheme(t) {
    document.body.className = t;
    const btn = document.getElementById("themeBtn");
    if (btn) btn.textContent = t === "dark" ? "💡" : "🌙";
}
function setTheme(t, notifyFrame = true) {
    globalTheme = t;
    localStorage.setItem("portalTheme", t);
    applyPortalTheme(t);
    const f = document.getElementById("frame");
    if (notifyFrame && f && f.style.display !== "none") {
        try { f.contentWindow.postMessage({ type: "theme", theme: t }, "*"); } catch (e) {}
    }
}
function toggleTheme() {
    setTheme(globalTheme === "dark" ? "light" : "dark");
}
window.addEventListener("message", e => {
    if (e.data && e.data.type === "theme") setTheme(e.data.theme);
});
applyPortalTheme(globalTheme);

/* 侧边栏折叠 */
function toggleSidebar() {
    const c = document.body.classList.toggle("collapsed");
    localStorage.setItem("portalCollapsed", c ? "1" : "0");
}
if (localStorage.getItem("portalCollapsed") === "1") document.body.classList.add("collapsed");

/* 侧边栏宽度拖拽（150-400px，记忆到 localStorage） */
(function () {
    const rz = document.getElementById('resizer');
    const frame = document.getElementById('frame');
    let dragging = false;
    rz.addEventListener('mousedown', e => {
        dragging = true;
        rz.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        if (frame) frame.style.pointerEvents = 'none';  // 防止 iframe 吞掉 mousemove
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        const w = Math.min(400, Math.max(150, e.clientX));
        document.documentElement.style.setProperty('--sbw', w + 'px');
    });
    window.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        rz.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        if (frame) frame.style.pointerEvents = '';
        const w = document.documentElement.style.getPropertyValue('--sbw');
        if (w) localStorage.setItem('portalSidebarW', parseInt(w));
    });
    const saved = parseInt(localStorage.getItem('portalSidebarW'));
    if (saved >= 150 && saved <= 400) document.documentElement.style.setProperty('--sbw', saved + 'px');
})();

const menu = document.getElementById('menu');
const savedOpen = JSON.parse(localStorage.getItem('portalGroups') || '{}');
GROUPS.forEach(g => {
    const wrap = document.createElement('div');
    wrap.className = 'group' + (savedOpen[g.name] === false ? ' closed' : '');
    wrap.dataset.gname = g.name;

    const head = document.createElement('div');
    head.className = 'group-head';
    head.innerHTML = '<span class="arrow">▶</span><span>' + (GROUP_ICONS[g.name] || "📁") + '</span>'
        + '<span class="gname">' + g.name + '</span>';
    head.title = '展开/收起 ' + g.name;
    head.onclick = () => {
        wrap.classList.toggle('closed');
        persistGroups();
    };

    const inner = document.createElement('div');
    inner.className = 'group-tools-inner';
    g.tools.forEach(t => {
        const div = document.createElement('div');
        div.className = 'item';
        div.id = 'item-' + t.id;
        div.title = t.name + '：' + t.desc;   // 折叠成图标条时悬浮可见完整说明
        div.innerHTML = '<span class="icon">' + t.icon + '</span>'
            + '<span class="txt"><div class="n">' + t.name + '</div>'
            + '<div class="d">' + t.desc + '</div></span>'
            + (t.external ? '' : '<span class="st" id="st-' + t.id + '"></span>');
        div.onclick = () => openTool(t.id);
        inner.appendChild(div);
    });
    const body = document.createElement('div');
    body.className = 'group-tools';
    body.appendChild(inner);

    wrap.appendChild(head);
    wrap.appendChild(body);
    menu.appendChild(wrap);
});
function persistGroups() {
    const st = {};
    document.querySelectorAll('.group').forEach(el =>
        st[el.dataset.gname] = !el.classList.contains('closed'));
    localStorage.setItem('portalGroups', JSON.stringify(st));
}

async function openTool(id) {
    currentId = id;
    document.querySelectorAll('.item').forEach(el =>
        el.classList.toggle('active', el.id === 'item-' + id));
    const t = ALL_TOOLS.find(x => x.id === id);
    document.getElementById('toolName').innerText = t.icon + ' ' + t.name;
    document.getElementById('newWinBtn').style.display = '';

    const loading = document.getElementById('loading');
    const frame = document.getElementById('frame');
    document.getElementById('welcome').style.display = 'none';

    if (t.external) {
        // 外部工具：直接内嵌第三方网页，无本地启停概念
        loading.style.display = 'none';
        frame.classList.remove('ready');
        frame.style.display = '';
        frame.src = t.url;
        frame.onload = () => requestAnimationFrame(() => frame.classList.add('ready'));
        refreshStatus();
        return;
    }

    loading.style.display = 'block';
    frame.style.display = 'none';

    try {
        const r = await fetch('/api/start', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ id })
        });
        const d = await r.json();
        if (!d.ok) {
            loading.style.display = 'none';
            alert('启动失败：' + (d.msg || '未知错误'));
            return;
        }
        frame.src = 'http://127.0.0.1:' + t.port + '/?theme=' + globalTheme;
        frame.classList.remove('ready');           // 先淡出旧页面
        frame.onload = () => {
            loading.style.display = 'none';
            frame.style.display = '';
            requestAnimationFrame(() => frame.classList.add('ready'));  // 新页面淡入
        };
    } catch (e) {
        loading.style.display = 'none';
        alert('请求失败：' + e.message);
    }
    refreshStatus();
}

function openNewWindow() {
    const t = ALL_TOOLS.find(x => x.id === currentId);
    if (!t) return;
    window.open(t.external ? t.url : ('http://127.0.0.1:' + t.port + '/?theme=' + globalTheme), '_blank');
}

async function refreshStatus() {
    try {
        const r = await fetch('/api/status');
        const st = await r.json();
        TOOLS.forEach(t => {
            const on = !!st[t.id];
            document.getElementById('st-' + t.id).className = 'st' + (on ? ' on' : '');
        });
        if (currentId) {
            const t = TOOLS.find(x => x.id === currentId);
            document.getElementById('toolDot').className = 'dot' + (st[currentId] ? ' on' : '');
        }
    } catch (e) { /* 门户自身出错时忽略 */ }
}
setInterval(refreshStatus, 15000);
refreshStatus();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    tools_slim = [{k: t[k] for k in ("id", "group", "name", "icon", "desc", "port")} for t in TOOLS]
    # 外部工具（纯网页内嵌，不是本地 Flask 服务）与本地工具一起按分组渲染菜单
    menu_json = {"tools": tools_slim, "external": EXTERNAL_TOOLS}
    # |safe 避免 Jinja 把 JSON 的引号转义成 HTML 实体导致前端菜单为空
    page = PAGE.replace("{{ tools_json }}", json.dumps(menu_json, ensure_ascii=False))
    return page


@app.route("/ping")
def ping():
    return {"ok": True, "app": "portal", "tools": len(TOOLS)}


@app.route("/api/status")
def api_status():
    result = {}

    def check(t):
        result[t["id"]] = ping_tool(t["port"])

    threads = [threading.Thread(target=check, args=(t,), daemon=True) for t in TOOLS]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=2.0)
    return jsonify(result)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    tool = TOOL_BY_ID.get(data.get("id"))
    if not tool:
        return {"ok": False, "msg": f"未知工具 id：{data.get('id')}"}
    ok, msg = start_tool(tool)
    return {"ok": ok, "msg": msg, "url": f"http://127.0.0.1:{tool['port']}/"}


def ensure_all_started():
    """启动器启动时：逐个检查所有工具，未运行的并行拉起，等待全部就绪"""
    launched = []
    for t in TOOLS:
        if ping_tool(t["port"], timeout=1.0):
            print(f"  [已运行] {t['name']} (:{t['port']})")
            continue
        script = os.path.join(t["dir"], t["script"])
        if not os.path.isfile(script):
            print(f"  [跳过] {t['name']}：找不到 {t['script']}")
            continue
        spawn_tool(t)
        launched.append(t)
        print(f"  [启动中] {t['name']} (:{t['port']}) ...")

    pending = {t["port"]: t["name"] for t in launched}
    deadline = time.time() + START_TIMEOUT
    while pending and time.time() < deadline:
        time.sleep(1.0)
        for port in list(pending):
            if ping_tool(port, timeout=1.0):
                print(f"  [就绪] {pending.pop(port)}")
    for port, name in pending.items():
        print(f"  [超时] {name} (:{port}) 仍不可达，请查看 logs/ 下的对应日志")


if __name__ == "__main__":
    import webbrowser
    _JOB = _setup_job_object()          # 门户退出时自动回收所有由它拉起的子进程
    kill_stale_listeners()              # 清理旧门户/孤儿工具，避免端口冲突和进程残留
    print(f"[portal] 发现 {len(TOOLS)} 个工具：")
    for t in TOOLS:
        print(f"  - {t['icon']} {t['name']}  port={t['port']}  ({t['dir']})")
    print("[portal] 逐个检查并启动工具 ...")
    ensure_all_started()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
