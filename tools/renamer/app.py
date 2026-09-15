# -*- coding: utf-8 -*-
"""
批量文件改名（网页版）
- 追加模式：img1.txt -> img1 (realesrgan-x4plus-anime s1024).txt
- 删除模式：反向恢复原名
支持：指定文件夹 / 文件后缀过滤 / 预览 / 幂等跳过 / 防覆盖。
端口：7865
"""

import os
import tkinter
from tkinter import filedialog

from flask import Flask, jsonify, render_template_string, request

PORT = 7865

app = Flask(__name__)


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp


def plan_renames(folder, extension, suffix, mode):
    """枚举目标文件，返回 (plans, skipped)。
    plans: [(旧名, 新名)]；skipped: [(旧名, 原因)]"""
    plans, skipped = [], []
    extension = extension.lower()

    for name in sorted(os.listdir(folder)):
        old_path = os.path.join(folder, name)
        if not os.path.isfile(old_path):
            continue

        stem, ext = os.path.splitext(name)
        if ext.lower() != extension:
            continue

        if mode == "add":
            if stem.endswith(suffix):
                skipped.append((name, "已包含该字符串（幂等跳过）"))
                continue
            new_name = stem + suffix + ext
        elif mode == "remove":
            if not stem.endswith(suffix):
                skipped.append((name, "文件名不含该字符串"))
                continue
            new_stem = stem[: -len(suffix)]
            if not new_stem:
                skipped.append((name, "删除后文件名为空"))
                continue
            new_name = new_stem + ext
        else:
            raise ValueError(f"未知模式：{mode!r}")

        if os.path.exists(os.path.join(folder, new_name)):
            skipped.append((name, f"目标文件已存在：{new_name}"))
            continue

        plans.append((name, new_name))
    return plans, skipped


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>批量文件后缀更改</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body.light {
    --bg:#f0f2f5; --card:#ffffff; --border:#e4e6eb; --text:#1c1e21;
    --head:#0b5bb5; --muted:rgba(28,30,33,.6); --inner:#f7f8fa;
    --input-bg:#ffffff; --input-border:#d0d3d8;
    color-scheme: light;
  }
  body.dark {
    --bg:#1e1f22; --card:#29292b; --border:#3a3b40; --text:#e6e8ee;
    --head:#74b9ff; --muted:rgba(230,232,238,.55); --inner:#1e1f22;
    --input-bg:#29292b; --input-border:#3f4147;
    color-scheme: dark;
  }
  body { margin:0; padding:24px; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--text); }
  .wrap { max-width: 900px; margin: 0 auto; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:20px 24px; margin-bottom:16px; box-shadow:0 1px 4px rgba(0,0,0,.05); }
  .row { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
  .row label { width:110px; flex-shrink:0; opacity:.75; font-size:14px; }
  .row input { flex:1; padding:10px 12px; border-radius:8px; border:1px solid var(--input-border);
               background:var(--input-bg); color:var(--text); font-size:14px; }
  .row select { padding:10px 12px; border-radius:8px; border:1px solid var(--input-border);
                background:var(--input-bg); color:var(--text); font-size:14px; }
  button { padding:10px 18px; border:none; border-radius:8px; background:#0984e3;
           color:#fff; font-size:14px; cursor:pointer; }
  button:hover { background:#0b6fc0; }
  button.warn { background:#e17055; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  h1 { font-size:20px; margin:0 0 16px; color:var(--head); }
  .tip { opacity:.6; font-size:12.5px; line-height:1.6; margin-top:8px; }
  .summary { margin:12px 0; font-size:14px; }
  .summary b { color:var(--head); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { padding:7px 10px; text-align:left; border-bottom:1px solid var(--border); word-break:break-all; }
  th { color:var(--head); }
  .skipped td { opacity:.55; }
  .banner { background:#e6f7ec; border:1px solid #b7e4c7; color:#1b7a3d; border-radius:8px;
            padding:10px 14px; margin-bottom:10px; font-size:14px; }
  body.dark .banner { background:#1f3a2a; border-color:#2e5c3f; color:#7ee2a0; }
  .example { background:var(--inner); border:1px solid var(--border); border-radius:8px;
             padding:10px 14px; font-size:12.5px; opacity:.9; margin-top:10px; line-height:1.7; }
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
<div class="wrap">
  <h1>📂 批量文件后缀更改</h1>
  <div class="card">
    <div class="row">
      <label>文件夹</label>
      <input id="folder" placeholder="输入或选择目标文件夹路径">
      <button onclick="pickFolder()">浏览…</button>
    </div>
    <div class="row">
      <label>文件后缀</label>
      <input id="ext" value=".txt" style="max-width:120px">
      <label style="width:80px">模式</label>
      <select id="mode">
        <option value="add">追加字符串</option>
        <option value="remove">删除字符串（还原）</option>
      </select>
    </div>
    <div class="row">
      <label>字符串</label>
      <input id="suffix" value=" (realesrgan-x4plus-anime s1024)">
      <button onclick="preview()">预览</button>
      <button class="warn" id="execBtn" onclick="exec()" disabled>执行改名</button>
    </div>
    <div class="example" id="example"></div>
    <div class="tip">先"预览"确认改名清单，再"执行改名"。已处理过的文件自动跳过（可重复运行）；
    目标文件名已存在时跳过保护，不会覆盖。</div>
  </div>
  <div class="card" id="result" style="display:none"></div>
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

function updateExample() {
    const ext = document.getElementById('ext').value || '.txt';
    const s = document.getElementById('suffix').value;
    const mode = document.getElementById('mode').value;
    const name = 'img1';
    const ex = mode === 'add' ? (name + ext + ' → ' + name + s + ext)
                              : (name + s + ext + ' → ' + name + ext);
    document.getElementById('example').innerText = '示例：' + ex;
}
document.getElementById('mode').onchange = updateExample;
document.getElementById('suffix').oninput = updateExample;
updateExample();

async function pickFolder() {
    const r = await fetch('/api/pick_folder');
    const d = await r.json();
    if (d.path) { document.getElementById('folder').value = d.path; }
}

function args() {
    return {
        folder: document.getElementById('folder').value.trim(),
        ext: document.getElementById('ext').value.trim(),
        suffix: document.getElementById('suffix').value,
        mode: document.getElementById('mode').value
    };
}

async function preview() {
    const box = document.getElementById('result');
    box.style.display = 'block';
    box.innerHTML = '<div class="summary"><span class="spin"></span>生成预览中…</div>';
    try {
        const r = await fetch('/api/preview', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(args())
        });
        const d = await r.json();
        if (!d.ok) { box.innerHTML = '<div class="summary">❌ ' + d.error + '</div>'; return; }
        // 有实际待改名文件时才允许执行
        document.getElementById('execBtn').disabled = !(d.plans && d.plans.length);
        render(d.plans, d.skipped, false);
    } catch (e) { box.innerHTML = '<div class="summary">请求失败：' + e.message + '</div>'; }
}

async function exec() {
    if (!confirm('确认执行改名？此操作会真实重命名文件。')) return;
    const box = document.getElementById('result');
    const btn = document.getElementById('execBtn');
    btn.disabled = true;
    try {
        const r = await fetch('/api/execute', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(args())
        });
        const d = await r.json();
        if (!d.ok) { box.innerHTML = '<div class="summary">❌ ' + d.error + '</div>'; btn.disabled = false; return; }
        render(d.plans, d.skipped, true);
        // 执行成功后按钮保持灰色；修改参数重新"预览"后才会再次启用
    } catch (e) { box.innerHTML = '<div class="summary">请求失败：' + e.message + '</div>'; btn.disabled = false; }
}

function render(plans, skipped, done) {
    const box = document.getElementById('result');
    let html = '';
    if (done) {
        html += '<div class="banner">✅ 改名完成！成功重命名 <b>' + plans.length + '</b> 个文件'
            + (skipped.length ? '，跳过 <b>' + skipped.length + '</b> 个' : '') + '</div>';
    }
    html += '<div class="summary">共 <b>' + plans.length + '</b> 个文件' + (done ? '已' : '将被')
        + '重命名' + (skipped.length ? '，<b>' + skipped.length + '</b> 个跳过' : '')
        + (plans.length && !done ? '。确认无误后点"执行改名"' : '') + '</div>';
    html += '<table><tr><th>原文件名</th><th>新文件名</th></tr>';
    plans.forEach(p => { html += '<tr><td>' + p[0] + '</td><td>' + p[1] + '</td></tr>'; });
    skipped.forEach(p => { html += '<tr class="skipped"><td>' + p[0] + '</td><td>跳过：' + p[1] + '</td></tr>'; });
    html += '</table>';
    box.innerHTML = html;
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/ping")
def ping():
    return {"ok": True, "app": "renamer"}


@app.route("/api/pick_folder")
def api_pick_folder():
    win = tkinter.Tk()
    win.withdraw()
    win.attributes("-topmost", True)
    path = filedialog.askdirectory(title="选择目标文件夹")
    win.destroy()
    return {"ok": True, "path": path}


def _check(data):
    folder = (data.get("folder") or "").strip()
    ext = (data.get("ext") or "").strip()
    suffix = data.get("suffix") or ""
    mode = data.get("mode") or ""
    if not os.path.isdir(folder):
        return None, None, None, None, f"文件夹不存在：{folder}"
    if not ext.startswith("."):
        return None, None, None, None, "文件后缀需要以点开头，例如 .txt"
    if mode not in ("add", "remove"):
        return None, None, None, None, "模式只支持 add / remove"
    if mode == "add" and not suffix:
        return None, None, None, None, "追加模式需要填写要追加的字符串"
    return folder, ext, suffix, mode, None


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data = request.get_json(silent=True) or {}
    folder, ext, suffix, mode, err = _check(data)
    if err:
        return {"ok": False, "error": err}
    try:
        plans, skipped = plan_renames(folder, ext, suffix, mode)
        return {"ok": True, "plans": plans, "skipped": skipped}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.route("/api/execute", methods=["POST"])
def api_execute():
    data = request.get_json(silent=True) or {}
    folder, ext, suffix, mode, err = _check(data)
    if err:
        return {"ok": False, "error": err}
    try:
        plans, skipped = plan_renames(folder, ext, suffix, mode)
        done = []
        for old_name, new_name in plans:
            os.rename(os.path.join(folder, old_name), os.path.join(folder, new_name))
            done.append((old_name, new_name))
        return {"ok": True, "plans": done, "skipped": skipped}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import webbrowser
    import threading
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"[renamer] serving at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
