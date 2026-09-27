# -*- coding: utf-8 -*-
"""
LoRA 识别（网页版）
扫描文件夹里的 .safetensors，网页上查看每个 LoRA / LyCORIS 的
算法类型、rank/alpha、底模、训练参数与作用层级统计。

识别核心来自 inspect_core.py（与命令行版 inspect_lora.py 同源）。
端口：7864
"""

import gc
import json
import os
import sys
import tkinter
import threading
from tkinter import filedialog

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_core  # noqa: E402

# ===== 离线翻译（腾讯 Hy-MT2-1.8B，llama.cpp 推理）=====
# 与 editor 工具不同：这里不常驻模型，点击"翻译"时才加载，翻译完立刻卸载释放显存
try:
    from llama_cpp import Llama
    _MT_OK = True
except Exception:
    _MT_OK = False

_mt_device_info = "未加载"
# llama.cpp 同一模型上下文不支持多线程并发推理（并发会触发 CUDA error 崩掉进程），必须串行化
_mt_lock = threading.Lock()
_MT_MODEL_NAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
_MT_MODEL_REPO = "tencent/Hy-MT2-1.8B-GGUF"
# 目标语言：label 为界面显示名，value 用于翻译指令（英文语言名）
_MT_LANGS = [
    ("中文（简体）", "Simplified Chinese"),
    ("中文（繁体）", "Traditional Chinese"),
    ("English", "English"),
    ("日本語", "Japanese"),
    ("한국어", "Korean"),
    ("粵語", "Cantonese"),
    ("Français", "French"),
    ("Deutsch", "German"),
    ("Español", "Spanish"),
    ("Русский", "Russian"),
    ("Português", "Portuguese"),
    ("Italiano", "Italian"),
]

# ---- 翻译配置（保存在脚本同目录 translate_config.json）----
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate_config.json")
_translate_cfg = {"target_lang": "Simplified Chinese", "model_src": "auto"}


def _load_translate_cfg():
    global _translate_cfg
    try:
        if os.path.exists(_CFG_FILE):
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _translate_cfg.update(data)
    except Exception as e:
        print(f"[translate] 读取配置失败: {e}")
    if _translate_cfg.get("model_src") not in ("auto", "hf", "mirror"):
        _translate_cfg["model_src"] = "auto"
    if _translate_cfg.get("target_lang") not in [v for _, v in _MT_LANGS]:
        _translate_cfg["target_lang"] = "Simplified Chinese"


def _save_translate_cfg():
    try:
        with open(_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(_translate_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[translate] 保存配置失败: {e}")


_load_translate_cfg()


def _pick_model_base():
    """决定模型下载域名：配置显式指定 > HF_ENDPOINT 环境变量 > 连通性探测。"""
    src = _translate_cfg.get("model_src", "auto")
    if src == "hf":
        return "https://huggingface.co"
    if src == "mirror":
        return "https://hf-mirror.com"
    import urllib.request
    env = os.environ.get("HF_ENDPOINT")
    if env:
        return env.rstrip("/")
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        urllib.request.urlopen(req, timeout=2)
        return "https://huggingface.co"
    except Exception:
        return "https://hf-mirror.com"


def _download_mt_model(dst):
    """本地没有模型时，流式下载到统一启动器/model/ 文件夹。"""
    import urllib.request
    base = _pick_model_base()
    url = f"{base}/{_MT_MODEL_REPO}/resolve/main/{_MT_MODEL_NAME}"
    print(f"[translate] 下载源: {base}")
    tmp = dst + ".part"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[translate] 下载中 {done / 1e6:.0f}/{total / 1e6:.0f} MB ({done * 100 // total}%)", end="", flush=True)
    print()
    os.replace(tmp, dst)


def _mt_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))  # 统一启动器/ 根目录
    return os.path.join(root_dir, "model", _MT_MODEL_NAME)


def _detect_device(model_path):
    """通过 llama.cpp 日志判断实际运行设备（GPU offload 层数）。"""
    import ctypes
    from llama_cpp import llama_cpp as _lc
    lines = []

    @_lc.ggml_log_callback
    def _cb(level, msg, user):
        try:
            lines.append(msg.decode("utf-8", "ignore"))
        except Exception:
            pass

    old_cb = _lc.ggml_log_callback()
    old_ud = ctypes.c_void_p()
    try:
        _lc.llama_log_get(ctypes.byref(old_cb), ctypes.byref(old_ud))
    except Exception:
        pass
    _lc.llama_log_set(_cb, None)
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    finally:
        _lc.llama_log_set(old_cb, old_ud)
    gpu_line = next((l for l in lines if "offloaded" in l and "GPU" in l), "")
    if gpu_line:
        try:
            n = gpu_line.split("offloaded")[1].split("/")[0].strip()
            total = gpu_line.split("/")[1].split("layers")[0].strip()
            device = f"GPU（已卸载 {n}/{total} 层）"
        except Exception:
            device = "GPU"
    else:
        device = "CPU（无可用 GPU，llama.cpp 自动回退）"
    return llm, device


PORT = 7864

app = Flask(__name__)


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lora参数分析</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body.light {
    --bg:#f0f2f5; --card:#ffffff; --border:#e4e6eb; --text:#1c1e21;
    --hover:#f0f2f5; --head:#0b5bb5; --muted:rgba(28,30,33,.55);
    --input-bg:#ffffff; --input-border:#d0d3d8; --btn:#e4e6eb; --btn-text:#1c1e21;
    color-scheme: light;
  }
  body.dark {
    --bg:#1e1f22; --card:#29292b; --border:#3a3b40; --text:#e6e8ee;
    --hover:#3a3b40; --head:#74b9ff; --muted:rgba(230,232,238,.55);
    --input-bg:#29292b; --input-border:#3f4147; --btn:#3f4147; --btn-text:#e6e8ee;
    color-scheme: dark;
  }
  body { margin:0; padding:20px; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--text); }
  .head { display:flex; gap:8px; align-items:center; margin-bottom:16px; }
  .head input { flex:1; padding:10px 12px; border-radius:8px; border:1px solid var(--input-border);
                background:var(--input-bg); color:var(--text); font-size:14px; }
  .head button { padding:10px 16px; border:none; border-radius:8px; background:#0984e3;
                 color:#fff; font-size:14px; cursor:pointer; white-space:nowrap; }
  .head button:hover { background:#0b6fc0; }
  .head button.ghost { background:var(--btn); color:var(--btn-text); }
  .head button.ghost:hover { background:var(--hover); }
  .layout { display:flex; gap:16px; align-items:flex-start; }
  .filelist { width:320px; flex-shrink:0; background:var(--card); border:1px solid var(--border);
              border-radius:10px; padding:8px; max-height:calc(100vh - 120px); overflow-y:auto; }
  .filelist .f { padding:10px 12px; border-radius:8px; cursor:pointer; font-size:13px; }
  .filelist .f:hover { background:var(--hover); }
  .filelist .f.active { background:#6c5ce7; color:#fff; }
  .filelist .f .sz { opacity:.6; margin-left:6px; font-size:12px; }
  .filelist .empty { padding:20px; text-align:center; opacity:.5; font-size:13px; }
  .detail { flex:1; background:var(--card); border:1px solid var(--border); border-radius:10px;
            padding:20px 24px; max-height:calc(100vh - 120px); overflow-y:auto; }
  .detail .empty { text-align:center; opacity:.5; padding:60px 0; }
  .detail h2 { margin:0 0 4px; font-size:20px; word-break:break-all; }
  .detail .path { opacity:.55; font-size:12px; margin-bottom:16px; word-break:break-all; }
  .badges { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
  .badge { padding:5px 12px; border-radius:20px; font-size:13px; background:#6c5ce7; color:#fff; }
  .badge.gray { background:var(--btn); color:var(--btn-text); }
  .badge.blue { background:#0984e3; color:#fff; }
  .badge.unet { background:#ffd9b0; color:#8a4b08; }
  .badge.te { background:#fff1e2; color:#a0651c; }
  .badge.size { background:#d9ecff; color:#0b5bb5; }
  .sec { margin-bottom:18px; }
  .sec h3 { margin:0 0 8px; font-size:14px; color:var(--head); }
  .kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:6px 20px; }
  .kv .row { display:flex; justify-content:space-between; gap:10px; font-size:13px;
             padding:5px 0; border-bottom:1px dashed var(--border); }
  .kv .row b { font-weight:600; opacity:.65; white-space:nowrap; }
  .kv .row span { text-align:right; word-break:break-all; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td { padding:6px 10px; text-align:left; border-bottom:1px solid var(--border); }
  th { color:var(--head); font-weight:600; }
  .tagbar { display:flex; flex-wrap:wrap; gap:6px; max-height:340px; overflow-y:auto; padding:2px; }
  .tagbar .tg { padding:3px 10px; border-radius:12px; background:var(--btn); color:var(--btn-text); font-size:12.5px; }
  .tagbar .tg b { color:#0984e3; margin-left:4px; font-weight:600; }
  .tagbar .tg .zh { color:#00b894; margin-left:5px; font-style:normal; }
  .mini-btn { padding:4px 12px; border:none; border-radius:14px; background:var(--btn);
              color:var(--btn-text); font-size:12px; cursor:pointer; white-space:nowrap; }
  .mini-btn:hover { background:var(--hover); }
  .mini-btn.primary { background:#6c5ce7; color:#fff; }
  .mini-btn:disabled { opacity:.6; cursor:default; }
  .modal-mask { display:none; position:fixed; inset:0; background:rgba(0,0,0,.45);
                z-index:1000; align-items:center; justify-content:center; }
  .modal-box { background:var(--card); border:1px solid var(--border); border-radius:12px;
               padding:22px 24px; width:400px; max-width:92vw; position:relative; }
  .modal-box h3 { margin:0 0 10px; font-size:16px; color:var(--head); }
  .modal-box label { display:block; font-size:12.5px; opacity:.65; margin:12px 0 4px; }
  .modal-box select { width:100%; padding:8px 10px; border-radius:8px; border:1px solid var(--input-border);
                      background:var(--input-bg); color:var(--text); font-size:13px; }
  .modal-btn { padding:8px 20px; border:none; border-radius:8px; background:#0984e3;
               color:#fff; cursor:pointer; font-size:13px; }
  .modal-btn:hover { background:#0b6fc0; }
  .modal-close { position:absolute; top:10px; right:14px; border:none; background:transparent;
                 color:var(--text); font-size:16px; cursor:pointer; opacity:.6; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #0984e3;
          border-top-color:transparent; border-radius:50%; animation:r .8s linear infinite;
          vertical-align:-2px; margin-right:6px; }
  @keyframes r { to { transform:rotate(360deg); } }
  .theme-toggle { position:fixed; bottom:20px; right:20px; font-size:24px; cursor:pointer;
                  background:transparent; border:none; padding:6px 10px; border-radius:6px;
                  transition:background .2s; z-index:999; }
  body.dark .theme-toggle:hover { background:#38383a; }
  body.light .theme-toggle:hover { background:#e9e9eb; }
</style>
</head>
<body>
  <div class="head">
    <input id="folder" placeholder="输入或选择包含 .safetensors 的文件夹路径">
    <button onclick="pickFolder()">浏览…</button>
    <button onclick="scan()">扫描</button>
    <button class="ghost" onclick="scan(true)">扫描当前目录</button>
  </div>
  <div class="layout">
    <div class="filelist" id="filelist"><div class="empty">输入路径后点“扫描”</div></div>
    <div class="detail" id="detail"><div class="empty">左侧选择一个文件查看详情</div></div>
  </div>
<!-- 翻译设置弹窗 -->
<div class="modal-mask" id="trCfgModal">
  <div class="modal-box">
    <h3>翻译设置</h3>
    <p id="tcfgDevice" style="margin:0 0 6px;font-size:12.5px;opacity:.6;">翻译模型：未加载（点击“翻译”时按需加载，翻完立刻卸载）</p>
    <label>目标语言</label>
    <select id="tcfgLang">
      {% for label, value in mt_langs %}
      <option value="{{ value }}">{{ label }}</option>
      {% endfor %}
    </select>
    <label>模型下载源（本地已有模型时无效）</label>
    <select id="tcfgSrc">
      <option value="auto">自动探测（推荐）</option>
      <option value="hf">HuggingFace 官网</option>
      <option value="mirror">hf-mirror 镜像</option>
    </select>
    <div style="margin-top:20px;text-align:right;">
      <button class="modal-btn" type="button" onclick="saveTranslateCfg()">保存</button>
    </div>
    <button class="modal-close" type="button" onclick="closeTranslateCfg()">✕</button>
  </div>
</div>
<button class="theme-toggle" id="themeBtn">💡</button>
<script>
// 全局主题协议：URL ?theme= 优先，其次本地记忆；接收门户 postMessage；灯泡点击时反向通知
(function () {
    function applyTheme(t) {
        document.body.className = t;
        try { localStorage.setItem("toolTheme", t); } catch (e) {}
        var btn = document.getElementById("themeBtn");
        if (btn) btn.textContent = t === "dark" ? "💡" : "🌙";
    }
    window.__applyTheme = applyTheme;
    var urlTheme = new URLSearchParams(location.search).get("theme");
    applyTheme(urlTheme || localStorage.getItem("toolTheme") || "light");
    window.addEventListener("message", function (e) {
        if (e.data && e.data.type === "theme") applyTheme(e.data.theme);
    });
    var themeBtn = document.getElementById("themeBtn");
    if (themeBtn) themeBtn.onclick = function () {
        var t = document.body.classList.contains("dark") ? "light" : "dark";
        applyTheme(t);
        try { if (window.parent !== window) window.parent.postMessage({ type: "theme", theme: t }, "*"); } catch (e) {}
    };
})();

let currentPath = null;
// 标签翻译状态（切换文件时重置）
let _trOn = false;      // 当前详情是否已显示译文
let _trBusy = false;    // 翻译请求进行中
let _trLangLabel = '';  // 目标语言显示名（按钮文案用）

async function pickFolder() {
    const r = await fetch('/api/pick_folder');
    const d = await r.json();
    if (d.path) { document.getElementById('folder').value = d.path; scan(); }
}

async function scan(useCwd) {
    const folder = useCwd ? '' : document.getElementById('folder').value.trim();
    const list = document.getElementById('filelist');
    list.innerHTML = '<div class="empty"><span class="spin"></span>扫描中…</div>';
    try {
        const r = await fetch('/api/scan', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder })
        });
        const d = await r.json();
        if (!d.ok) { list.innerHTML = '<div class="empty">' + (d.error || '失败') + '</div>'; return; }
        if (!d.files.length) { list.innerHTML = '<div class="empty">没有找到 .safetensors 文件</div>'; return; }
        list.innerHTML = '';
        d.files.forEach(f => {
            const div = document.createElement('div');
            div.className = 'f';
            div.innerHTML = f.name + '<span class="sz">' + f.size + '</span>';
            div.onclick = () => { currentPath = f.path; inspect(); };
            list.appendChild(div);
        });
    } catch (e) { list.innerHTML = '<div class="empty">请求失败：' + e.message + '</div>'; }
}

async function inspect() {
    document.querySelectorAll('.filelist .f').forEach(el => {
        el.classList.toggle('active', el.textContent.startsWith(currentPath.split('\\').pop()));
    });
    _trOn = false; _trBusy = false;  // 切换文件重置翻译状态
    const box = document.getElementById('detail');
    box.innerHTML = '<div class="empty"><span class="spin"></span>读取中…</div>';
    try {
        const r = await fetch('/api/inspect', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: currentPath })
        });
        const d = await r.json();
        box.innerHTML = d.ok ? render(d.info) : ('<div class="empty">' + (d.error || '读取失败') + '</div>');
        if (d.ok) refreshTrBtn();
    } catch (e) { box.innerHTML = '<div class="empty">请求失败：' + e.message + '</div>'; }
}

function fmt(v) { return (v === null || v === undefined || v === '') ? '-' : v; }
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function filterTags(q) {
    q = q.toLowerCase();
    document.querySelectorAll('#tagCloud .tg').forEach(el => {
        el.style.display = el.dataset.t.toLowerCase().includes(q) ? '' : 'none';
    });
}

// ===== 标签翻译（后端按需加载 Hy-MT2 模型，翻完立刻卸载）=====
function trBtnText() {
    if (_trBusy) return '⏳ 翻译中…';
    if (_trOn) return '🌐 关闭翻译';
    return _trLangLabel ? '🌐 翻译(' + _trLangLabel + ')' : '🌐 翻译';
}
function refreshTrBtn() {
    const btn = document.getElementById('trBtn');
    if (!btn) return;
    btn.innerText = trBtnText();
    btn.disabled = _trBusy;
    btn.classList.toggle('primary', _trOn || _trBusy);
}
async function doTranslate() {
    if (_trBusy) return;
    const cloud = document.getElementById('tagCloud');
    if (!cloud) return;
    if (_trOn) {  // 关闭翻译：只移除译文
        _trOn = false;
        cloud.querySelectorAll('.zh').forEach(el => el.remove());
        refreshTrBtn();
        return;
    }
    const tags = Array.from(cloud.querySelectorAll('.tg'))
        .filter(el => !el.querySelector('.zh')).map(el => el.dataset.t);
    if (!tags.length) return;
    _trBusy = true; refreshTrBtn();
    try {
        const r = await fetch('/api/translate', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ tags: tags })
        });
        const d = await r.json();
        if (d && d.ok && d.translations) {
            cloud.querySelectorAll('.tg').forEach(el => {
                const zh = d.translations[el.dataset.t];
                if (zh && !el.querySelector('.zh')) {
                    const i = document.createElement('i');
                    i.className = 'zh';
                    i.innerText = zh;
                    el.appendChild(i);
                }
            });
            _trOn = true;
        } else {
            alert('翻译失败：' + ((d && d.error) || '服务异常'));
        }
    } catch (e) {
        alert('请求失败：' + e.message);
    } finally {
        _trBusy = false; refreshTrBtn();
    }
}

// ===== 翻译设置 =====
async function openTranslateCfg() {
    try {
        const r = await fetch('/api/translate_config');
        const d = await r.json();
        if (d && d.config) {
            document.getElementById('tcfgLang').value = d.config.target_lang;
            document.getElementById('tcfgSrc').value = d.config.model_src;
        }
        const dev = document.getElementById('tcfgDevice');
        if (dev && d) dev.innerText = '翻译模型：' + (d.device || '未加载')
            + '（点击“翻译”时按需加载，翻完立刻卸载）';
    } catch (e) { /* 读取失败也允许打开面板修改 */ }
    document.getElementById('trCfgModal').style.display = 'flex';
}
function closeTranslateCfg() {
    document.getElementById('trCfgModal').style.display = 'none';
}
async function saveTranslateCfg() {
    const langSel = document.getElementById('tcfgLang');
    try {
        const r = await fetch('/api/translate_config', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ target_lang: langSel.value, model_src: document.getElementById('tcfgSrc').value })
        });
        const d = await r.json();
        if (d && d.ok) {
            _trLangLabel = langSel.selectedOptions[0] ? langSel.selectedOptions[0].text : '';
            refreshTrBtn();
            closeTranslateCfg();
            // 已显示旧语言译文：清掉，用户可再点翻译按新语言重翻
            if (_trOn) {
                _trOn = false;
                document.querySelectorAll('#tagCloud .zh').forEach(el => el.remove());
                refreshTrBtn();
            }
        } else {
            alert('保存失败');
        }
    } catch (e) {
        alert('保存失败：' + e.message);
    }
}
// 页面加载时取目标语言名，更新按钮文案
(async function () {
    try {
        const r = await fetch('/api/translate_config');
        const d = await r.json();
        if (d && d.config) {
            const sel = document.getElementById('tcfgLang');
            const opt = Array.from(sel.options).find(o => o.value === d.config.target_lang);
            _trLangLabel = opt ? opt.text : '';
            refreshTrBtn();
        }
    } catch (e) { /* 用默认文案 */ }
})();

function render(i) {
    const rank = i.rank_meta || i.rank_inferred;
    const alpha = i.alpha_effective;
    // 作用目标：按 tensor 前缀统计（unet / te1 / te2 / te），只显示实际包含的
    const sub = i.submodel_summary || {};
    const targets = [];
    if (sub.unet) targets.push('Unet');
    if (sub.te || sub.te1 || sub.te2) targets.push('TextEncoder');
    let html = '<h2>' + i.file + '</h2><div class="path">' + i.path + '</div>';
    html += '<div class="badges">'
        + '<span class="badge">' + i.algorithm + '</span>'
        + (i.network_module_pretty && i.network_module_pretty !== i.algorithm
            ? '<span class="badge gray">' + i.network_module_pretty + '</span>' : '')
        + targets.map(t => '<span class="badge ' + (t === 'Unet' ? 'unet' : 'te') + '">' + t + '</span>').join('')
        + (rank !== null && rank !== undefined ? '<span class="badge blue">dim ' + rank + ' / alpha ' + fmt(alpha) + '</span>' : '')
        + '<span class="badge size">' + i.file_size + '</span>'
        + '<span class="badge gray">' + i.num_tensors + ' tensors</span>'
        + '</div>';

    const kv = (pairs) => '<div class="kv">' + pairs
        .filter(p => p[1] !== null && p[1] !== undefined && p[1] !== '')
        .map(p => '<div class="row"><b>' + p[0] + '</b><span>' + fmt(p[1]) + '</span></div>').join('') + '</div>';

    html += '<div class="sec"><h3>模型</h3>' + kv([
        ['基础架构', i.architecture || (i.is_sdxl ? 'SDXL' : (String(i.is_v2).toLowerCase() === 'true' ? 'SD v2' : 'SD 1.5 (?)'))],
        ['底模', i.base_model_name], ['底模版本', i.base_model_version], ['底模 hash', i.sd_model_hash]
    ]) + '</div>';

    const ddirs = (i.dataset_dirs && Object.keys(i.dataset_dirs).length)
        ? Object.entries(i.dataset_dirs)
            .map(([k, v]) => k + '（' + v.img_count + ' 张 × ' + v.n_repeats + ' 次）').join('；')
        : null;
    const ds0 = (i.datasets && i.datasets[0]) || null;
    const bucket = ds0 ? (ds0.enable_bucket
        ? '开启 ' + ds0.min_bucket_reso + '–' + ds0.max_bucket_reso : '关闭') : null;

    html += '<div class="sec"><h3>训练参数</h3>' + kv([
        ['步数', i.final_step !== null ? i.final_step + ' / max ' + fmt(i.max_train_steps) : null],
        ['epoch', i.final_epoch !== null ? i.final_epoch + ' / ' + fmt(i.num_epochs) : null],
        ['训练图片', i.num_train_images], ['正则图片', i.num_reg_images],
        ['batch', (i.batch_size_per_device ?? '') !== '' ? '每卡 ' + i.batch_size_per_device + ' / 总 ' + fmt(i.total_batch_size) : null],
        ['梯度累加', i.grad_accum],
        ['分辨率', i.resolution], ['clip skip', i.clip_skip], ['max token length', i.max_token_length],
        ['数据集', ddirs], ['分桶', bucket],
        ['学习率', i.learning_rate ?? i.unet_lr], ['unet lr', i.unet_lr], ['TE lr', i.text_encoder_lr],
        ['调度器', i.lr_scheduler], ['warmup', i.lr_warmup_steps], ['优化器', i.optimizer],
        ['种子', i.seed], ['精度', i.mixed_precision],
        ['min_snr_gamma', i.min_snr_gamma], ['noise_offset', i.noise_offset],
        ['开始时间', i.started_at], ['耗时', i.training_duration]
    ]) + '</div>';

    if (i.tag_frequency && i.tag_frequency.length) {
        html += '<div class="sec"><h3 style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">训练标签'
            + ' <span style="opacity:.6;font-weight:normal">' + i.tag_frequency.length + ' 个 / 出现 ' + i.tag_total + ' 次</span>'
            + '<span style="flex:1"></span>'
            + '<button class="mini-btn" id="trBtn" title="加载翻译模型，把全部标签翻成目标语言，翻完立刻卸载模型" onclick="doTranslate()">🌐 翻译</button>'
            + '<button class="mini-btn" title="翻译设置" onclick="openTranslateCfg()">⚙️ 翻译设置</button>'
            + '</h3>'
            + '<input id="tagFilter" placeholder="筛选标签…" oninput="filterTags(this.value)" '
            + 'style="width:100%;max-width:360px;padding:8px 10px;border-radius:8px;'
            + 'border:1px solid var(--input-border);background:var(--input-bg);'
            + 'color:var(--text);font-size:13px;margin-bottom:10px">'
            + '<div class="tagbar" id="tagCloud">'
            + i.tag_frequency.map(t =>
                '<span class="tg" data-t="' + esc(t[0]) + '">' + esc(t[0]) + '<b>×' + t[1] + '</b></span>').join('')
            + '</div></div>';
    }

    if (i.network_args && typeof i.network_args === 'object' && Object.keys(i.network_args).length) {
        html += '<div class="sec"><h3>network_args</h3>' + kv(
            Object.entries(i.network_args)) + '</div>';
    }

    const regions = i.region_summary || {};
    const keys = Object.keys(regions);
    if (keys.length) {
        html += '<div class="sec"><h3>作用层级</h3><table><tr><th>区域</th><th>tensors</th><th>不同层</th><th>示例</th></tr>';
        keys.forEach(k => {
            html += '<tr><td>' + k + '</td><td>' + regions[k] + '</td><td>' + fmt((i.region_unique_layers || {})[k]) + '</td><td>' +
                ((i.region_samples || {})[k] || []).join(', ').slice(0, 120) + '</td></tr>';
        });
        html += '</table></div>';
    }
    return html;
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, mt_langs=_MT_LANGS)


@app.route("/ping")
def ping():
    return {"ok": True, "app": "lora-inspect"}


@app.route("/api/pick_folder")
def api_pick_folder():
    win = tkinter.Tk()
    win.withdraw()
    win.attributes("-topmost", True)
    path = filedialog.askdirectory(title="选择包含 .safetensors 的文件夹")
    win.destroy()
    return {"ok": True, "path": path}


@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json(silent=True) or {}
    folder = (data.get("folder") or "").strip() or os.getcwd()
    if not os.path.isdir(folder):
        return {"ok": False, "error": f"文件夹不存在：{folder}"}
    files = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(".safetensors"):
            files.append({"name": name, "path": p,
                          "size": inspect_core.human_size(os.path.getsize(p))})
    return {"ok": True, "folder": folder, "files": files}


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    data = request.get_json(silent=True) or {}
    path = (data.get("path") or "").strip()
    if not os.path.isfile(path):
        return {"ok": False, "error": f"文件不存在：{path}"}
    try:
        info = inspect_core.inspect_one(path)
        return {"ok": True, "info": info}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """按需加载翻译模型 → 逐条翻译 → 立刻卸载释放显存/内存。"""
    global _mt_device_info
    data = request.get_json(silent=True) or {}
    tags = []
    for t in (data.get("tags") or []):
        if isinstance(t, str) and t.strip() and t not in tags:
            tags.append(t.strip())
    if not tags:
        return {"ok": False, "error": "没有要翻译的标签"}
    if not _MT_OK:
        return {"ok": False, "error": "未安装 llama-cpp-python，请先安装"}
    model_path = _mt_model_path()
    if not os.path.isfile(model_path):
        return {"ok": False, "error": f"缺少模型文件：{model_path}"}
    lang = _translate_cfg.get("target_lang", "Simplified Chinese")
    out = {}
    with _mt_lock:  # 串行化：加载+推理全程持锁，防止并发崩掉 llama.cpp
        llm = None
        try:
            llm, _mt_device_info = _detect_device(model_path)
            for t in tags:
                prompt = f"Translate the following segment into {lang}, without additional explanation.\n\n{t}"
                resp = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=128, temperature=0.0)
                zh = resp["choices"][0]["message"]["content"].strip()
                if zh:
                    out[t] = zh
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "translations": out}
        finally:
            if llm is not None:
                del llm  # 翻译完（或出错）立刻卸载模型
                gc.collect()
    return {"ok": True, "translations": out}


@app.route("/api/translate_config", methods=["GET"])
def api_get_tcfg():
    return {"ok": True, "config": _translate_cfg,
            "device": _mt_device_info, "loaded": False}  # 模型按需加载，平时不驻留


@app.route("/api/translate_config", methods=["POST"])
def api_set_tcfg():
    payload = request.get_json(silent=True) or {}
    lang = payload.get("target_lang")
    src = payload.get("model_src")
    if isinstance(lang, str) and lang in [v for _, v in _MT_LANGS]:
        _translate_cfg["target_lang"] = lang
    if src in ("auto", "hf", "mirror"):
        _translate_cfg["model_src"] = src
    _save_translate_cfg()
    return {"ok": True, "config": _translate_cfg}


if __name__ == "__main__":
    import webbrowser
    import threading
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"[lora-inspect] serving at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
