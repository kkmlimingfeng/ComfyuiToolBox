# -*- coding: utf-8 -*-
"""
LoRA 识别（网页版）
扫描文件夹里的 .safetensors，网页上查看每个 LoRA / LyCORIS 的
算法类型、rank/alpha、底模、训练参数与作用层级统计。

识别核心来自 inspect_core.py（与命令行版 inspect_lora.py 同源）。
端口：7864
"""

import os
import sys
import tkinter
from tkinter import filedialog

from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_core  # noqa: E402

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
    const box = document.getElementById('detail');
    box.innerHTML = '<div class="empty"><span class="spin"></span>读取中…</div>';
    try {
        const r = await fetch('/api/inspect', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: currentPath })
        });
        const d = await r.json();
        box.innerHTML = d.ok ? render(d.info) : ('<div class="empty">' + (d.error || '读取失败') + '</div>');
    } catch (e) { box.innerHTML = '<div class="empty">请求失败：' + e.message + '</div>'; }
}

function fmt(v) { return (v === null || v === undefined || v === '') ? '-' : v; }

function render(i) {
    const rank = i.rank_meta || i.rank_inferred;
    const alpha = i.alpha_effective;
    let html = '<h2>' + i.file + '</h2><div class="path">' + i.path + '</div>';
    html += '<div class="badges">'
        + '<span class="badge">' + i.algorithm + '</span>'
        + (i.network_module_pretty ? '<span class="badge gray">' + i.network_module_pretty + '</span>' : '')
        + (rank !== null && rank !== undefined ? '<span class="badge blue">dim ' + rank + ' / alpha ' + fmt(alpha) + '</span>' : '')
        + '<span class="badge gray">' + i.file_size + '</span>'
        + '<span class="badge gray">' + i.num_tensors + ' tensors</span>'
        + '</div>';

    const kv = (pairs) => '<div class="kv">' + pairs
        .filter(p => p[1] !== null && p[1] !== undefined && p[1] !== '')
        .map(p => '<div class="row"><b>' + p[0] + '</b><span>' + fmt(p[1]) + '</span></div>').join('') + '</div>';

    html += '<div class="sec"><h3>模型</h3>' + kv([
        ['基础架构', i.architecture || (i.is_sdxl ? 'SDXL' : (String(i.is_v2).toLowerCase() === 'true' ? 'SD v2' : 'SD 1.5 (?)'))],
        ['底模', i.base_model_name], ['底模版本', i.base_model_version], ['底模 hash', i.sd_model_hash]
    ]) + '</div>';

    html += '<div class="sec"><h3>训练参数</h3>' + kv([
        ['步数', i.final_step !== null ? i.final_step + ' / max ' + fmt(i.max_train_steps) : null],
        ['epoch', i.final_epoch !== null ? i.final_epoch + ' / ' + fmt(i.num_epochs) : null],
        ['训练图片', i.num_train_images], ['正则图片', i.num_reg_images],
        ['batch', (i.batch_size_per_device ?? '') !== '' ? '每卡 ' + i.batch_size_per_device + ' / 总 ' + fmt(i.total_batch_size) : null],
        ['分辨率', i.resolution], ['clip skip', i.clip_skip],
        ['学习率', i.learning_rate ?? i.unet_lr], ['unet lr', i.unet_lr], ['TE lr', i.text_encoder_lr],
        ['调度器', i.lr_scheduler], ['warmup', i.lr_warmup_steps], ['优化器', i.optimizer],
        ['种子', i.seed], ['精度', i.mixed_precision],
        ['min_snr_gamma', i.min_snr_gamma], ['noise_offset', i.noise_offset],
        ['开始时间', i.started_at], ['耗时', i.training_duration]
    ]) + '</div>';

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
    return render_template_string(PAGE)


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


if __name__ == "__main__":
    import webbrowser
    import threading
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    print(f"[lora-inspect] serving at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
