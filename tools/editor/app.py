from flask import Flask, render_template_string, request, redirect, url_for
import os
from collections import Counter
import webbrowser
import time
import threading
import tkinter
from tkinter import filedialog

# ===== 离线翻译（腾讯 Hy-MT2-1.8B，llama.cpp 推理，GPU 自动加速 / 无 GPU 自动用 CPU）=====
try:
    from llama_cpp import Llama
    _MT_OK = True
except Exception:
    _MT_OK = False

_mt_model = None
_mt_device_info = "未加载"
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
            import json
            with open(_CFG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _translate_cfg.update(data)
    except Exception as e:
        print(f"[translate] 读取配置失败: {e}")
    # 兼容非法值
    if _translate_cfg.get("model_src") not in ("auto", "hf", "mirror"):
        _translate_cfg["model_src"] = "auto"
    if _translate_cfg.get("target_lang") not in [v for _, v in _MT_LANGS]:
        _translate_cfg["target_lang"] = "Simplified Chinese"

def _save_translate_cfg():
    try:
        import json
        with open(_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(_translate_cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[translate] 保存配置失败: {e}")

_load_translate_cfg()

def _pick_model_base():
    """决定模型下载域名：配置显式指定 > HF_ENDPOINT 环境变量 > 连通性探测（官网 2 秒内可达用官网，否则用国内镜像）。"""
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
    """本地没有模型时，流式下载到根目录 model/ 文件夹。"""
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

def _detect_device(model):
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

    # 暂存并替换默认日志回调，加载后恢复
    old_cb = _lc.ggml_log_callback()
    old_ud = ctypes.c_void_p()
    try:
        _lc.llama_log_get(ctypes.byref(old_cb), ctypes.byref(old_ud))
    except Exception:
        pass
    _lc.llama_log_set(_cb, None)
    try:
        llm = Llama(model_path=model, n_gpu_layers=-1, n_ctx=2048, verbose=False)
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

def _load_translator():
    """启动时加载 Hy-MT2 翻译模型：有 GPU 自动 GPU 加速，无 GPU 自动 CPU 运行。"""
    global _mt_model, _mt_device_info
    if not _MT_OK:
        _mt_device_info = "未安装 llama-cpp-python"
        print("[translate] 未安装 llama-cpp-python，请先安装")
        return False
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(script_dir))  # 工作区根目录（脚本在 标签编辑器/v2/ 下，往上两级）
        model_path = os.path.join(root_dir, "model", _MT_MODEL_NAME)
        print(f"[translate] 查找: {model_path}")
        if not os.path.isfile(model_path):
            print("[translate] 本地没有模型，开始下载（约 1 GB）...")
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                _download_mt_model(model_path)
            except Exception as e:
                _mt_device_info = "模型缺失"
                print(f"[translate] 下载失败: {e}")
                print(f"[translate] 可手动从 {_pick_model_base()}/{_MT_MODEL_REPO}/resolve/main/{_MT_MODEL_NAME} 下载后放到 model/ 文件夹")
                return False
        _mt_model, _mt_device_info = _detect_device(model_path)
        print(f"[translate] Hy-MT2 翻译模型已就绪 · 设备: {_mt_device_info}")
        return True
    except Exception as e:
        _mt_device_info = f"加载失败: {e}"
        print(f"[translate] 加载失败: {e}")
        return False

def translate_text(text):
    """按当前配置的目标语言翻译一段文本。"""
    if _mt_model is None or not text:
        return ""
    try:
        lang = _translate_cfg.get("target_lang", "Simplified Chinese")
        prompt = f"Translate the following segment into {lang}, without additional explanation.\n\n{text}"
        out = _mt_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128, temperature=0.0)
        return out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[translate error] {e}")
        return ""

app = Flask(__name__)


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp

PATH_RECORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_path.txt")

def load_last_folder():
    if os.path.exists(PATH_RECORD_FILE):
        with open(PATH_RECORD_FILE, "r", encoding="utf-8") as f:
            path = f.read().strip()
            if os.path.isdir(path):
                return path
    return ""

def save_last_folder(path):
    with open(PATH_RECORD_FILE, "w", encoding="utf-8") as f:
        f.write(path)

def open_system_folder_chooser():
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    last_dir = load_last_folder()
    selected_dir = filedialog.askdirectory(initialdir=last_dir, title="选择图片数据集文件夹")
    root.destroy()
    return selected_dir

CURRENT_IMG_DIR = load_last_folder()
IMG_SUFFIX = (".png", ".jpg", ".jpeg", ".webp")
img_list = []
current_idx = 0

def refresh_file_list():
    global img_list
    if not os.path.isdir(CURRENT_IMG_DIR):
        img_list = []
        return
    all_files = sorted(os.listdir(CURRENT_IMG_DIR))
    img_list = [f for f in all_files if f.lower().endswith(IMG_SUFFIX)]

def get_img_path():
    if not img_list:
        return None
    return os.path.join(CURRENT_IMG_DIR, img_list[current_idx])

def get_txt_path():
    if not img_list:
        return ""
    name, ext = os.path.splitext(img_list[current_idx])
    return os.path.join(CURRENT_IMG_DIR, f"{name}.txt")

def count_all_tags_frequency():
    all_tags = []
    for img_name in img_list:
        stem, _ = os.path.splitext(img_name)
        txt_file = os.path.join(CURRENT_IMG_DIR, f"{stem}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                tag_text = f.read().strip()
                tags = [t.strip() for t in tag_text.split(",") if t.strip()]
                all_tags.extend(tags)
    return Counter(all_tags)

def get_current_tag_list():
    txt_path = get_txt_path()
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return [t.strip() for t in content.split(",") if t.strip()]

# 初始空白选择页
SELECT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>图片标签编辑器 · 选择数据集目录</title>
    <style>
        html { background: #1c1c1e; }
        * {margin: 0; padding: 0; box-sizing: border-box;}
        body {
            font-family: system-ui;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.3s, color 0.3s;
        }
        body.dark {
            background: #1c1c1e;
            color: #e6e6e6;
        }
        body.light {
            background: #f5f5f7;
            color: #2c2c2e;
        }
        .wrap {
            padding: 50px 40px;
            border-radius: 14px;
            text-align: center;
            min-width: 650px;
            transition: background 0.3s;
        }
        body.dark .wrap { background: #29292b; }
        body.light .wrap { background: #ffffff; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }

        h2 {
            font-size: 32px;
            margin-bottom: 24px;
            color: #74b9ff;
        }
        .tip {
            font-size: 15px;
            margin-bottom: 32px;
            line-height: 1.7;
            opacity: 0.8;
        }
        .choose-btn {
            padding: 18px 42px;
            background: #0984e3;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .choose-btn:hover {
            background: #1994f0;
        }
        .theme-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 24px;
            cursor: pointer;
            background: transparent;
            border: none;
            padding: 6px 10px;
            border-radius: 6px;
            transition: background 0.2s;
            z-index: 999;
        }
        body.dark .theme-toggle:hover { background: #38383a; }
        body.light .theme-toggle:hover { background: #e9e9eb; }
    </style>
</head>
<body class="dark">
    <script>
        // 首帧前修正主题：body 硬编码 dark 仅作无 JS 兜底；切图整页导航时
        // 若等底部脚本才改 class，浅色模式下会先画一帧深色再渐变过来（黑闪）
        (function () {
            var t = new URLSearchParams(location.search).get("theme")
                 || localStorage.getItem("toolTheme") || "dark";
            document.body.className = t;
            document.documentElement.style.background = t === "dark" ? "#1c1c1e" : "#f5f5f7";
        })();
    </script>
    <button class="theme-toggle" id="themeBtn">💡</button>
    <div class="wrap">
        <h2>选择图片数据集文件夹</h2>
        <p class="tip">点击按钮唤起系统文件资源管理器，挑选存放图片标签的文件夹<br>程序会自动记忆上一次打开的目录</p>
        <form method="post" action="/pick_folder">
            <button class="choose-btn" type="submit">📂 浏览文件夹</button>
        </form>
    </div>

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
            applyTheme(urlTheme || localStorage.getItem("toolTheme") || "dark");
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
    </script>
</body>
</html>
"""

# 主编辑页面（内含弹窗选择文件夹）
EDIT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>图片标签编辑器</title>
    <style>
        html { background: #1c1c1e; }
        *{box-sizing:border-box;margin:0;padding:0;}
        body { 
            display: flex; gap: 24px; padding: 20px; 
            font-family: system-ui;
            transition: background 0.3s, color 0.3s;
        }
        body.dark {
            background: #1c1c1e;
            color: #e6e6e6;
        }
        body.light {
            background: #f5f5f7;
            color: #2c2c2e;
        }

        .preview-area { 
            flex: 1.8;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .preview-area img { 
            max-width: 100%;
            max-height: 92vh;
            object-fit: contain;
            border-radius: 4px;
            transition: border-color 0.3s;
        }
        body.dark .preview-area img { border:1px solid #3a3a3c; }
        body.light .preview-area img { border:1px solid #d2d2d7; }

        .control-area { flex: 1.2; display: flex; flex-direction: column; gap: 16px; }

        .top-bar {
            display:flex;justify-content:space-between;align-items:center;
            padding-bottom:8px;
            transition: border-color 0.3s;
        }
        body.dark .top-bar { border-bottom:1px solid #3a3a3c; }
        body.light .top-bar { border-bottom:1px solid #d2d2d7; }

        .progress { font-size:16px; font-weight:bold; color:#74b9ff; }
        .change-dir-btn {
            padding:6px 10px;border:none;border-radius:4px;cursor:pointer;
            transition: all 0.2s;
        }
        body.dark .change-dir-btn { background:#38383a;color:#e6e6e6; }
        body.light .change-dir-btn { background:#e9e9eb;color:#2c2c2e; }

        textarea { 
            width:100%; height: 160px; padding:10px; border-radius:6px; font-size:14px;
            border-width:1px; border-style:solid;
            transition: background 0.3s, color 0.3s, border-color 0.3s;
        }
        body.dark textarea {
            background:#29292b; color:#e6e6e6; border-color:#3a3a3c;
        }
        body.light textarea {
            background:#ffffff; color:#2c2c2e; border-color:#d2d2d7;
        }
        
        .tag-group {
            overflow-y: visible;
            max-height: unset;
            height: auto;
            padding:14px;
            border-radius:6px;
            border-width:1px; border-style:solid;
            transition: background 0.3s, border-color 0.3s;
        }
        body.dark .tag-group { background:#29292b; border-color:#3a3a3c; }
        body.light .tag-group { background:#ffffff; border-color:#d2d2d7; }

        .tag-item {
            display: inline-flex;
            align-items: center;
            margin: 6px 8px;
            padding: 6px 10px;
            border-radius:4px;
            cursor:default;
            user-select:none;
            position: relative;
            gap: 6px;
            transition: background 0.2s;
        }
        body.dark .tag-item { background:#374151; }
        body.light .tag-item { background:#e4e7ec; color:#1f2937; }

        .tag-item input[type="checkbox"] { cursor: pointer; }
        .freq { font-size:12px; opacity:0.75; }

        .edit-icon {
            display: none;
            position: absolute;
            right: 4px;
            top: 50%;
            transform: translateY(-50%);
            width: 18px;
            height: 18px;
            border-radius: 3px;
            color: #fff;
            text-align: center;
            line-height: 18px;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.2s;
        }
        body.dark .edit-icon { background: #4a5568; }
        body.light .edit-icon { background: #9ca3af; }
        .tag-item:hover .edit-icon {
            display: block;
        }

        .inline-input {
            border: 1px solid #74b9ff;
            padding: 2px 4px;
            outline: none;
            font-size: inherit;
            min-width: 80px;
            transition: background 0.3s, color 0.3s;
        }
        body.dark .inline-input { background: #1c1c1e; color:#e6e6e6; }
        body.light .inline-input { background: #ffffff; color:#2c2c2e; }

        .btn-wrap { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
        button {
            padding:9px 14px;
            border:none;
            border-radius:5px;
            cursor:pointer;
            font-size:14px;
        }
        .btn-save { background:#0984e3; color:white; }
        .btn-del-check { background:#d63031; color:white; }
        .btn-del-img { background:#636e72; color:white; }
        .btn-add-tag { background:#00897b; color:white; }
        .btn-del-all { background:#e17055; color:white; }
        .btn-add-all { background:#00b894; color:white; }
        .btn-translate { background:#6c5ce7; color:white; cursor:pointer; }
        .btn-translate.active { background:#a29bfe; }
        .tag-zh { color:#6c5ce7; font-size:0.85em; margin-left:6px; }
        
        .add-tag-line { display:flex; gap:8px; align-items:center; }
        .add-tag-line input {
            flex:1;
            padding:8px;
            border-radius:4px;
            border-width:1px; border-style:solid;
            transition: background 0.3s, color 0.3s, border-color 0.3s;
        }
        body.dark .add-tag-line input {
            background:#29292b; color:#e6e6e6; border-color:#3a3a3c;
        }
        body.light .add-tag-line input {
            background:#ffffff; color:#2c2c2e; border-color:#d2d2d7;
        }

        .theme-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 24px;
            cursor: pointer;
            background: transparent;
            border: none;
            padding: 6px 10px;
            border-radius: 6px;
            z-index: 999;
            transition: background 0.2s;
        }
        body.dark .theme-toggle:hover { background: #38383a; }
        body.light .theme-toggle:hover { background: #e9e9eb; }

        .btn-replace { background: #fdcb6e; color: #2c2c2e; cursor: pointer; }
        .btn-replace:hover { background: #ffeaa7; }

        /* 文件夹选择弹窗遮罩层 */
        .modal-mask {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.6);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal-box {
            padding: 40px;
            border-radius: 14px;
            text-align: center;
            min-width: 520px;
        }
        body.dark .modal-box { background: #29292b; }
        body.light .modal-box { background: #fff; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
        .modal-title {
            font-size: 26px;
            color: #74b9ff;
            margin-bottom: 20px;
        }
        .modal-tip {
            opacity: 0.8;
            margin-bottom: 28px;
            line-height: 1.6;
        }
        .modal-btn {
            padding: 16px 36px;
            background: #0984e3;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 17px;
            cursor: pointer;
        }
        .modal-close {
            margin-top: 16px;
            background: transparent;
            color: #aaa;
            border: none;
            cursor: pointer;
            font-size: 15px;
        }

        /* 翻译设置弹窗 */
        .tcfg-row { margin: 14px 0; text-align: left; }
        .tcfg-row label { display: block; margin-bottom: 6px; opacity: 0.75; font-size: 14px; }
        .tcfg-row select {
            width: 100%;
            padding: 10px 12px;
            border-radius: 8px;
            border: 1px solid rgba(128,128,128,0.4);
            background: transparent;
            color: inherit;
            font-size: 15px;
            box-sizing: border-box;
        }
        .tcfg-row select option { color: #222; }

        /* 批量替换弹窗 */
        .replace-modal-box {
            min-width: 560px;
            max-width: 80vw;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            text-align: left;
        }
        .replace-table-wrap {
            max-height: 50vh;
            overflow-y: auto;
            margin: 6px 0 20px 0;
            border-radius: 6px;
            border-width: 1px; border-style: solid;
        }
        body.dark .replace-table-wrap { border-color: #3a3a3c; }
        body.light .replace-table-wrap { border-color: #d2d2d7; }
        .replace-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .replace-table th {
            text-align: left;
            padding: 10px 14px;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        body.dark .replace-table th { background: #1c1c1e; color: #74b9ff; }
        body.light .replace-table th { background: #f5f5f7; color: #0984e3; border-bottom: 1px solid #d2d2d7; }
        .replace-table td {
            padding: 8px 14px;
            border-top-width: 1px; border-top-style: solid;
            word-break: break-all;
        }
        body.dark .replace-table td { border-top-color: #3a3a3c; }
        body.light .replace-table td { border-top-color: #e9e9eb; }
        .replace-table input {
            width: 100%;
            padding: 6px 8px;
            border-radius: 4px;
            border-width: 1px; border-style: solid;
            font-size: 14px;
            transition: background 0.3s, color 0.3s, border-color 0.3s;
        }
        body.dark .replace-table input { background: #1c1c1e; color: #e6e6e6; border-color: #3a3a3c; }
        body.light .replace-table input { background: #ffffff; color: #2c2c2e; border-color: #d2d2d7; }
        .replace-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            flex-wrap: wrap;
        }
        .replace-actions button {
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
        }
        .replace-btn-cancel { background: #636e72; }
        .replace-btn-cancel:hover { background: #7f8c8d; }
        .replace-btn-current { background: #0984e3; }
        .replace-btn-current:hover { background: #1994f0; }
        .replace-btn-all { background: #e17055; }
        .replace-btn-all:hover { background: #ff7f50; }

        /* 标签拖拽排序 */
        .tag-item { cursor: grab; }
        .tag-item:active { cursor: grabbing; }
        .tag-item.dragging {
            opacity: 0.85;
            cursor: grabbing;
            box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        }
        body.dark .tag-item.dragging { background: #4a5568; }
        .tag-item-placeholder {
            display: inline-block;
            vertical-align: middle;
            border-width: 2px; border-style: dashed; border-color: #74b9ff;
            border-radius: 4px;
            background: transparent !important;
            margin: 6px 8px;
        }
        body.dark .tag-item.drag-over-left { box-shadow: -2px 0 0 0 #74b9ff; }
        body.light .tag-item.drag-over-left { box-shadow: -2px 0 0 0 #0984e3; }
        body.dark .tag-item.drag-over-right { box-shadow: 2px 0 0 0 #74b9ff; }
        body.light .tag-item.drag-over-right { box-shadow: 2px 0 0 0 #0984e3; }
    </style>
</head>
<body class="dark">
    <script>
        // 首帧前修正主题（同 SELECT 页，防切图黑闪）
        (function () {
            var t = new URLSearchParams(location.search).get("theme")
                 || localStorage.getItem("toolTheme") || "dark";
            document.body.className = t;
            document.documentElement.style.background = t === "dark" ? "#1c1c1e" : "#f5f5f7";
        })();
    </script>
    <button class="theme-toggle" id="themeBtn">💡</button>

    <!-- 文件夹选择弹窗 -->
    <div class="modal-mask" id="folderModal">
        <div class="modal-box">
            <h3 class="modal-title">切换数据集文件夹</h3>
            <p class="modal-tip">点击下方按钮，打开系统文件资源管理器选择新目录<br>选择完成后页面将自动刷新加载图片</p>
            <form method="post" action="/pick_folder">
                <button class="modal-btn" type="submit">📂 浏览文件夹</button>
            </form>
            <button class="modal-close" onclick="closeModal()">取消</button>
        </div>
    </div>

    <!-- 批量替换弹窗 -->
    <div class="modal-mask" id="replaceModal">
        <div class="modal-box replace-modal-box">
            <h3 class="modal-title">批量替换标签</h3>
            <p class="modal-tip">左侧为原标签，右侧输入新标签（留空则删除该标签）</p>
            <div class="replace-table-wrap">
                <table class="replace-table">
                    <thead>
                        <tr><th>原标签</th><th>替换为</th></tr>
                    </thead>
                    <tbody id="replaceTableBody"></tbody>
                </table>
            </div>
            <div class="replace-actions">
                <button class="replace-btn-cancel" type="button" onclick="closeReplaceModal()">取消</button>
                <button class="replace-btn-current" type="button" onclick="doReplace(false)">仅替换本图</button>
                <button class="replace-btn-all" type="button" onclick="doReplace(true)">替换全部图片</button>
            </div>
        </div>
    </div>

    <!-- 翻译设置弹窗 -->
    <div class="modal-mask" id="translateCfgModal">
        <div class="modal-box" style="min-width: 420px;">
            <h3 class="modal-title">翻译设置</h3>
            <p class="modal-tip" id="tcfgDevice" style="margin-bottom: 12px;">翻译模型：未加载</p>
            <div class="tcfg-row">
                <label>目标语言</label>
                <select id="tcfgLang">
                    {% for label, value in mt_langs %}
                    <option value="{{ value }}">{{ label }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="tcfg-row">
                <label>模型下载源（本地已有模型时无效）</label>
                <select id="tcfgSrc">
                    <option value="auto">自动探测（推荐）</option>
                    <option value="hf">HuggingFace 官网</option>
                    <option value="mirror">hf-mirror 镜像</option>
                </select>
            </div>
            <div style="margin-top: 24px;">
                <button class="modal-btn" type="button" onclick="saveTranslateCfg()">保存</button>
            </div>
            <button class="modal-close" type="button" onclick="closeTranslateCfg()">取消</button>
        </div>
    </div>

    <div class="preview-area">
        {% if img_path %}
            <img src="{{ img_path }}">
        {% else %}
            <h2>当前文件夹无图片文件</h2>
        {% endif %}
    </div>

    <div class="control-area">
        <div class="top-bar">
            <div class="progress">进度：{{now}} / {{total}}</div>
            <div style="display: flex; gap: 10px;">
                <button class="change-dir-btn" onclick="openTranslateCfg()">⚙️ 翻译设置</button>
                <button class="change-dir-btn" onclick="openModal()">切换数据集文件夹</button>
            </div>
        </div>

        <form method="post" class="img-nav-bar" id="navForm">
            <div class="btn-wrap">
                <button type="submit" formaction="/prev_img">上一张</button>
                <button type="submit" formaction="/next_img">下一张</button>
                <button formaction="/del_img_file" class="btn-del-img" type="submit" onclick="return confirm('确认删除当前图片及其标签文件？');">删除图片+标签文件</button>
            </div>
        </form>

        <form method="post" action="/add_new_tag" class="add-tag-line">
            <input type="text" name="new_tag" placeholder="输入标签，添加至当前图片末尾">
            <button class="btn-add-tag" type="submit">添加标签</button>
        </form>

        <form method="post" action="/batch_del_tags">
            <div>标签列表（勾选删除 | 悬浮点击铅笔原位修改）</div>
            <div class="tag-group">
                {% for tag in current_tags %}
                <div class="tag-item" data-orig-tag="{{tag}}">
                    <input type="checkbox" name="del_tag_list" value="{{tag}}">
                    <span class="tag-text">{{tag}}</span>
                    <span class="freq">[{{tag_count[tag]}}次]</span>
                    <span class="edit-icon" onclick="startEdit(this)">✏</span>
                </div>
                {% endfor %}
            </div>
            <div class="btn-wrap">
                <button class="btn-del-check" type="submit">删除勾选标签</button>
                <button class="btn-del-all" type="submit" formaction="/batch_del_all_imgs" onclick="return confirm('确认从所有图片中删除所选标签？');">所有图片删除</button>
                <button class="btn-add-all" type="submit" formaction="/batch_add_all_imgs">所有图片添加</button>
                <button class="btn-translate" id="translateBtn" type="button" onclick="doTranslate()">🌐 翻译</button>
                <button class="btn-replace" type="button" onclick="openReplaceModal()">🔄 批量替换</button>
            </div>
        </form>

        <form method="post" action="/save_all_text">
            <textarea name="full_tag_text">{{ raw_tag_text }}</textarea>
            <div class="btn-wrap">
                <button class="btn-save" type="submit">保存文本标签</button>
            </div>
        </form>
    </div>

    <script>
        // 弹窗控制
        function openModal(){
            document.getElementById('folderModal').style.display = 'flex';
        }
        function closeModal(){
            document.getElementById('folderModal').style.display = 'none';
        }

        // 全局主题协议（同 SELECT 页）
        (function () {
            function applyTheme(t) {
                document.body.className = t;
                try { localStorage.setItem("toolTheme", t); } catch (e) {}
                var btn = document.getElementById("themeBtn");
                if (btn) btn.textContent = t === "dark" ? "💡" : "🌙";
            }
            window.__applyTheme = applyTheme;
            var urlTheme = new URLSearchParams(location.search).get("theme");
            applyTheme(urlTheme || localStorage.getItem("toolTheme") || "dark");
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

        // 原位编辑标签逻辑
        let editingDom = null;
        let originText = "";
        function startEdit(btn) {
            if(editingDom) cancelEdit();
            const itemWrap = btn.closest(".tag-item");
            const textSpan = itemWrap.querySelector(".tag-text");
            originText = textSpan.innerText;
            editingDom = itemWrap;
            const input = document.createElement("input");
            input.className = "inline-input";
            input.value = originText;
            input.style.width = originText.length * 8 + "px";
            textSpan.replaceWith(input);
            input.focus();
            input.select();
            input.onkeydown = (e) => {
                if(e.key === "Enter") submitEdit(input.value.trim());
                if(e.key === "Escape") cancelEdit();
            }
            input.onblur = () => submitEdit(input.value.trim());
        }
        function submitEdit(newVal) {
            if(!editingDom) return;
            const oldVal = editingDom.dataset.origTag;
            if(!newVal || newVal === oldVal) { cancelEdit(); return; }
            location.href = `/edit_single_tag?old=${encodeURIComponent(oldVal)}&new=${encodeURIComponent(newVal)}`;
        }
        function cancelEdit() {
            if(!editingDom) return;
            const input = editingDom.querySelector(".inline-input");
            if(input) {
                const span = document.createElement("span");
                span.className = "tag-text";
                span.innerText = originText;
                input.replaceWith(span);
            }
            editingDom = null;
            originText = "";
        }

        // ===== 翻译（目标语言可在"翻译设置"里切换）=====
        // 开关状态存 sessionStorage：切图（整页刷新）后保持翻译模式，新图自动翻译；关浏览器自动复位
        let _zhOn = sessionStorage.getItem("zhOn") === "1";
        let _targetLangLabel = '';
        function translateBtnText(on) {
            return on ? '🌐 关闭翻译' : (_targetLangLabel ? ('🌐 翻译(' + _targetLangLabel + ')') : '🌐 翻译');
        }
        async function fetchAndFillTranslations() {
            const items = document.querySelectorAll('.tag-item');
            const need = [];
            items.forEach(it => {
                if (!it.querySelector('.tag-zh')) {
                    const t = it.dataset.origTag;
                    if (t) need.push(t);
                }
            });
            if (!need.length) return;
            const resp = await fetch('/api/translate?tags=' + encodeURIComponent(need.join(',')));
            const data = await resp.json();
            if (data && data.translations) {
                items.forEach(it => {
                    const t = it.dataset.origTag;
                    if (t && data.translations[t] && !it.querySelector('.tag-zh')) {
                        const sp = document.createElement('span');
                        sp.className = 'tag-zh';
                        sp.innerText = data.translations[t];
                        it.appendChild(sp);
                    }
                });
            } else {
                alert('翻译失败：' + (data && data.error || '服务异常'));
            }
        }
        async function doTranslate() {
            const btn = document.getElementById('translateBtn');
            if (_zhOn) {
                // 关闭
                _zhOn = false;
                sessionStorage.setItem("zhOn", "0");
                document.querySelectorAll('.tag-zh').forEach(el => el.remove());
                if (btn) { btn.innerText = translateBtnText(false); btn.classList.remove('active'); }
                return;
            }
            _zhOn = true;
            sessionStorage.setItem("zhOn", "1");
            if (btn) { btn.innerText = '⏳ 翻译中...'; btn.disabled = true; }
            try {
                await fetchAndFillTranslations();
            } catch (e) {
                alert('请求失败：' + e.message);
            } finally {
                if (btn) { btn.disabled = false; btn.innerText = translateBtnText(true); btn.classList.add('active'); }
            }
        }

        // 页面加载时恢复翻译模式：切图后自动翻译当前图片
        (async function restoreTranslateState() {
            const btn = document.getElementById('translateBtn');
            try {
                const resp = await fetch('/api/translate_config');
                const d = await resp.json();
                if (d && d.config) {
                    const sel = document.getElementById('tcfgLang');
                    sel.value = d.config.target_lang;
                    _targetLangLabel = sel.selectedOptions[0] ? sel.selectedOptions[0].text : '';
                }
            } catch (e) { /* 拿不到语言标签就用默认文案 */ }
            if (btn) btn.innerText = translateBtnText(_zhOn);
            if (_zhOn && btn) {
                btn.classList.add('active');
                btn.disabled = true;
                try { await fetchAndFillTranslations(); } catch (e) { /* 静默重试 */ }
                btn.disabled = false;
                btn.innerText = translateBtnText(true);
            }
        })();

        // 切图使用原生 POST → redirect → 服务端重新渲染。
        // 这是最稳定的路径，避免内嵌 WebView / iframe 对 AJAX DOM 更新产生兼容问题。

        // ===== 翻译设置 =====
        async function openTranslateCfg() {
            try {
                const resp = await fetch('/api/translate_config');
                const data = await resp.json();
                if (data && data.config) {
                    document.getElementById('tcfgLang').value = data.config.target_lang;
                    document.getElementById('tcfgSrc').value = data.config.model_src;
                }
                if (data) {
                    const dev = document.getElementById('tcfgDevice');
                    dev.innerText = '翻译模型：' + (data.loaded ? ('已加载 · ' + data.device) : data.device);
                }
            } catch (e) { /* 读取失败也允许打开面板修改 */ }
            document.getElementById('translateCfgModal').style.display = 'flex';
        }
        function closeTranslateCfg() {
            document.getElementById('translateCfgModal').style.display = 'none';
        }
        async function saveTranslateCfg() {
            const langSel = document.getElementById('tcfgLang');
            const src = document.getElementById('tcfgSrc').value;
            try {
                const resp = await fetch('/api/translate_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_lang: langSel.value, model_src: src })
                });
                const data = await resp.json();
                if (data && data.ok && data.config) {
                    _targetLangLabel = langSel.selectedOptions[0].text;
                    const btn = document.getElementById('translateBtn');
                    if (btn && !_zhOn) btn.innerText = translateBtnText(false);
                    closeTranslateCfg();
                    // 翻译开启中：清空旧语言译文，按新语言重新拉取
                    if (_zhOn) {
                        document.querySelectorAll('.tag-zh').forEach(el => el.remove());
                        fetchAndFillTranslations().catch(() => {});
                    }
                } else {
                    alert('保存失败');
                }
            } catch (e) {
                alert('保存失败：' + e.message);
            }
        }
        // 初始化按钮文案（显示当前目标语言）
        (function initTranslateBtn() {
            fetch('/api/translate_config').then(r => r.json()).then(data => {
                if (data && data.config) {
                    const sel = document.getElementById('tcfgLang');
                    const opt = Array.from(sel.options).find(o => o.value === data.config.target_lang);
                    _targetLangLabel = opt ? opt.text : '';
                    const btn = document.getElementById('translateBtn');
                    if (btn) btn.innerText = translateBtnText(false);
                }
            }).catch(() => {});
        })();

        // ===== 批量替换 =====
        function openReplaceModal() {
            const checked = document.querySelectorAll('.tag-item input[type="checkbox"]:checked');
            if (checked.length === 0) {
                alert('请先勾选要替换的标签');
                return;
            }
            const tbody = document.getElementById('replaceTableBody');
            tbody.innerHTML = '';
            checked.forEach(cb => {
                const item = cb.closest('.tag-item');
                const orig = item.dataset.origTag;
                const tr = document.createElement('tr');
                const td1 = document.createElement('td');
                td1.textContent = orig;
                const td2 = document.createElement('td');
                const inp = document.createElement('input');
                inp.type = 'text';
                inp.dataset.orig = orig;
                inp.placeholder = '新标签（留空则删除）';
                td2.appendChild(inp);
                tr.appendChild(td1);
                tr.appendChild(td2);
                tbody.appendChild(tr);
            });
            document.getElementById('replaceModal').style.display = 'flex';
            const firstInput = tbody.querySelector('input');
            if (firstInput) firstInput.focus();
        }
        function closeReplaceModal() {
            document.getElementById('replaceModal').style.display = 'none';
        }
        async function doReplace(applyAll) {
            const rows = document.querySelectorAll('#replaceTableBody tr');
            const pairs = [];
            rows.forEach(tr => {
                const orig = tr.cells[0].textContent;
                const neu = tr.querySelector('input').value.trim();
                pairs.push({orig: orig, neu: neu});
            });
            if (pairs.length === 0) { closeReplaceModal(); return; }
            if (applyAll && !confirm('确认将选中的标签在所有图片中替换？')) return;
            try {
                const resp = await fetch('/api/replace_tags', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({pairs: pairs, all: applyAll})
                });
                const data = await resp.json();
                if (data && data.ok) {
                    location.reload();
                } else {
                    alert('替换失败：' + (data && data.error || '未知错误'));
                }
            } catch (e) {
                alert('请求失败：' + e.message);
            }
        }

        // ===== 标签拖拽排序（长按触发） =====
        (function() {
            const LONG_PRESS_MS = 350;
            const MOVE_THRESHOLD = 5;
            let lpTimer = null;
            let lpStartX = 0, lpStartY = 0;
            let isDragging = false;
            let draggedItem = null;
            let dragOffsetX = 0, dragOffsetY = 0;
            let placeholder = null;

            function clearDropIndicators() {
                document.querySelectorAll('.tag-item.drag-over-left, .tag-item.drag-over-right')
                    .forEach(el => el.classList.remove('drag-over-left', 'drag-over-right'));
            }

            function getDropTarget(x, y) {
                const elements = document.elementsFromPoint(x, y);
                for (const el of elements) {
                    if (el.classList && el.classList.contains('tag-item')
                        && el !== draggedItem && el !== placeholder) {
                        return el;
                    }
                }
                return null;
            }

            function updateDropIndicator(target, x) {
                clearDropIndicators();
                if (!target) return;
                const rect = target.getBoundingClientRect();
                const mid = rect.left + rect.width / 2;
                if (x < mid) {
                    target.classList.add('drag-over-left');
                } else {
                    target.classList.add('drag-over-right');
                }
            }

            function startDrag(item, clientX, clientY) {
                isDragging = true;
                draggedItem = item;
                const rect = item.getBoundingClientRect();
                dragOffsetX = clientX - rect.left;
                dragOffsetY = clientY - rect.top;
                placeholder = document.createElement('span');
                placeholder.className = 'tag-item-placeholder';
                placeholder.style.width = rect.width + 'px';
                placeholder.style.height = rect.height + 'px';
                item.parentNode.insertBefore(placeholder, item);
                item.style.position = 'fixed';
                item.style.left = rect.left + 'px';
                item.style.top = rect.top + 'px';
                item.style.width = rect.width + 'px';
                item.style.zIndex = '9999';
                item.classList.add('dragging');
                document.body.style.userSelect = 'none';
            }

            function endDrag(clientX, clientY) {
                const target = getDropTarget(clientX, clientY);
                if (target) {
                    const rect = target.getBoundingClientRect();
                    const mid = rect.left + rect.width / 2;
                    if (clientX < mid) {
                        target.parentNode.insertBefore(placeholder, target);
                    } else {
                        target.parentNode.insertBefore(placeholder, target.nextSibling);
                    }
                } else {
                    draggedItem.parentNode.appendChild(placeholder);
                }
                placeholder.parentNode.replaceChild(draggedItem, placeholder);
                placeholder = null;
                draggedItem.style.position = '';
                draggedItem.style.left = '';
                draggedItem.style.top = '';
                draggedItem.style.width = '';
                draggedItem.style.zIndex = '';
                draggedItem.classList.remove('dragging');
                document.body.style.userSelect = '';
                clearDropIndicators();
                draggedItem = null;
                isDragging = false;

                const items = document.querySelectorAll('.tag-item');
                const newOrder = Array.from(items).map(it => it.dataset.origTag);
                const textarea = document.querySelector('textarea[name="full_tag_text"]');
                if (textarea) textarea.value = newOrder.join(', ');
                fetch('/api/reorder_tags', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({order: newOrder})
                }).then(r => r.json()).then(data => {
                    if (!data || !data.ok) {
                        alert('顺序保存失败：' + ((data && data.error) || '未知错误'));
                    }
                }).catch(e => {
                    alert('请求失败：' + e.message);
                });
            }

            function setupDrag(item) {
                item.addEventListener('mousedown', (e) => {
                    if (e.button !== 0) return;
                    if (e.target.tagName === 'INPUT') return;
                    if (e.target.classList && e.target.classList.contains('edit-icon')) return;
                    if (e.target.classList && e.target.classList.contains('inline-input')) return;
                    lpStartX = e.clientX;
                    lpStartY = e.clientY;
                    lpTimer = setTimeout(() => {
                        startDrag(item, e.clientX, e.clientY);
                    }, LONG_PRESS_MS);
                });
            }

            document.addEventListener('mousemove', (e) => {
                if (lpTimer && !isDragging) {
                    const dx = e.clientX - lpStartX;
                    const dy = e.clientY - lpStartY;
                    if (Math.abs(dx) > MOVE_THRESHOLD || Math.abs(dy) > MOVE_THRESHOLD) {
                        clearTimeout(lpTimer);
                        lpTimer = null;
                    }
                }
                if (!isDragging) return;
                e.preventDefault();
                draggedItem.style.left = (e.clientX - dragOffsetX) + 'px';
                draggedItem.style.top = (e.clientY - dragOffsetY) + 'px';
                const target = getDropTarget(e.clientX, e.clientY);
                updateDropIndicator(target, e.clientX);
            });

            document.addEventListener('mouseup', (e) => {
                if (lpTimer) { clearTimeout(lpTimer); lpTimer = null; }
                if (isDragging) endDrag(e.clientX, e.clientY);
            });

            function init() {
                document.querySelectorAll('.tag-item').forEach(setupDrag);
            }
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', init);
            } else {
                init();
            }
        })();
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    global CURRENT_IMG_DIR
    if not CURRENT_IMG_DIR or not os.path.isdir(CURRENT_IMG_DIR):
        return render_template_string(SELECT_HTML)
    refresh_file_list()
    total_num = len(img_list)
    if total_num == 0:
        return render_template_string(EDIT_HTML, img_path="", now=0, total=0, current_tags=[], tag_count={}, raw_tag_text="", mt_langs=_MT_LANGS)
    tag_counter = count_all_tags_frequency()
    tag_list = get_current_tag_list()
    txt_path = get_txt_path()
    raw_text = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    return render_template_string(EDIT_HTML,
        img_path=url_for("send_img", fname=img_list[current_idx]),
        now=current_idx + 1,
        total=total_num,
        current_tags=tag_list,
        tag_count=tag_counter,
        raw_tag_text=raw_text,
        mt_langs=_MT_LANGS
    )

@app.route("/pick_folder", methods=["POST"])
def pick_folder():
    global CURRENT_IMG_DIR, current_idx
    selected_path = open_system_folder_chooser()
    if selected_path and os.path.isdir(selected_path):
        CURRENT_IMG_DIR = selected_path
        save_last_folder(selected_path)
        current_idx = 0
    return redirect(url_for("home"))

@app.route("/edit_single_tag")
def edit_tag():
    old = request.args.get("old", "").strip()
    new = request.args.get("new", "").strip()
    if not old or not new:
        return redirect(url_for("home"))
    tags = get_current_tag_list()
    res = [new if t == old else t for t in tags]
    txt_path = get_txt_path()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(res))
    return redirect(url_for("home"))

@app.route("/img/<fname>")
def send_img(fname):
    from flask import send_from_directory
    # 已看过的图让浏览器缓存 1 小时，往回翻时秒切
    return send_from_directory(CURRENT_IMG_DIR, fname, max_age=3600)

def _current_image_payload():
    """当前图片 + 标签的 JSON 载荷，供 AJAX 切图原地更新页面"""
    if not img_list:
        return {"ok": False}
    idx = max(0, min(current_idx, len(img_list) - 1))
    txt_path = get_txt_path()
    raw_text = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    return {
        "ok": True,
        "idx": idx + 1,
        "total": len(img_list),
        "img_url": url_for("send_img", fname=img_list[idx]),
        "tags": get_current_tag_list(),
        "tag_count": count_all_tags_frequency(),
        "raw": raw_text,
    }

@app.route("/api/image", methods=["GET"])
def api_image():
    if not CURRENT_IMG_DIR or not os.path.isdir(CURRENT_IMG_DIR):
        return jsonify({"ok": False})
    refresh_file_list()
    return jsonify(_current_image_payload())

@app.route("/prev_img", methods=["POST"])
def prev():
    global current_idx
    if current_idx > 0:
        current_idx -= 1
    return redirect(url_for("home"))

@app.route("/next_img", methods=["POST"])
def next_img():
    global current_idx
    if current_idx < len(img_list) - 1:
        current_idx += 1
    return redirect(url_for("home"))

@app.route("/save_all_text", methods=["POST"])
def save_full_text():
    text = request.form.get("full_tag_text", "").strip()
    txt_path = get_txt_path()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return redirect(url_for("home"))

@app.route("/batch_del_tags", methods=["POST"])
def batch_del():
    del_list = request.form.getlist("del_tag_list")
    if not del_list: return redirect(url_for("home"))
    tags = get_current_tag_list()
    new_tags = [t for t in tags if t not in del_list]
    txt_path = get_txt_path()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(new_tags))
    return redirect(url_for("home"))

@app.route("/batch_del_all_imgs", methods=["POST"])
def batch_del_all_imgs():
    del_list = request.form.getlist("del_tag_list")
    if not del_list: return redirect(url_for("home"))
    for img_name in img_list:
        stem, _ = os.path.splitext(img_name)
        txt_file = os.path.join(CURRENT_IMG_DIR, f"{stem}.txt")
        if not os.path.exists(txt_file):
            continue
        with open(txt_file, "r", encoding="utf-8") as f:
            tag_text = f.read().strip()
        tags = [t.strip() for t in tag_text.split(",") if t.strip()]
        new_tags = [t for t in tags if t not in del_list]
        if new_tags != tags:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(", ".join(new_tags))
    return redirect(url_for("home"))

@app.route("/batch_add_all_imgs", methods=["POST"])
def batch_add_all_imgs():
    add_list = request.form.getlist("del_tag_list")
    if not add_list: return redirect(url_for("home"))
    for img_name in img_list:
        stem, _ = os.path.splitext(img_name)
        txt_file = os.path.join(CURRENT_IMG_DIR, f"{stem}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                tag_text = f.read().strip()
            tags = [t.strip() for t in tag_text.split(",") if t.strip()]
        else:
            tags = []
        changed = False
        for t in add_list:
            if t not in tags:
                tags.append(t)
                changed = True
        if changed:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(", ".join(tags))
    return redirect(url_for("home"))

@app.route("/add_new_tag", methods=["POST"])
def add_tag():
    new_t = request.form.get("new_tag", "").strip()
    if not new_t: return redirect(url_for("home"))
    tags = get_current_tag_list()
    if new_t not in tags:
        tags.append(new_t)
        txt_path = get_txt_path()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))
    return redirect(url_for("home"))

@app.route("/del_img_file", methods=["POST"])
def delete_img():
    global current_idx
    img_p = get_img_path()
    txt_p = get_txt_path()
    if os.path.exists(img_p): os.remove(img_p)
    if os.path.exists(txt_p): os.remove(txt_p)
    refresh_file_list()
    if current_idx >= len(img_list):
        current_idx = max(0, len(img_list)-1)
    return redirect(url_for("home"))

@app.route("/ping")
def ping():
    return {"ok": True, "app": "editor-v2"}

@app.route("/api/translate")
def api_translate():
    tags = [t for t in request.args.get("tags", "").split(",") if t.strip()]
    out = {}
    for t in tags:
        zh = translate_text(t)
        if zh:
            out[t] = zh
    return {"ok": _mt_model is not None, "translations": out}

@app.route("/api/translate_config", methods=["GET"])
def api_get_tcfg():
    return {"ok": True, "config": _translate_cfg,
            "device": _mt_device_info, "loaded": _mt_model is not None}

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

@app.route("/api/replace_tags", methods=["POST"])
def api_replace_tags():
    payload = request.get_json(silent=True) or {}
    pairs = payload.get("pairs", [])
    apply_all = bool(payload.get("all"))
    if not isinstance(pairs, list) or not pairs:
        return {"ok": False, "error": "无替换项"}
    mapping = {}
    for p in pairs:
        if not isinstance(p, dict):
            continue
        orig = (p.get("orig") or "").strip()
        if not orig:
            continue
        mapping[orig] = (p.get("neu") or "").strip()
    if not mapping:
        return {"ok": False, "error": "无有效替换项"}
    if not img_list:
        return {"ok": False, "error": "没有可操作的目标图片"}
    targets = img_list if apply_all else [img_list[current_idx]]
    changed_files = 0
    for img_name in targets:
        stem, _ = os.path.splitext(img_name)
        txt_file = os.path.join(CURRENT_IMG_DIR, f"{stem}.txt")
        if not os.path.exists(txt_file):
            continue
        with open(txt_file, "r", encoding="utf-8") as f:
            tag_text = f.read().strip()
        tags = [t.strip() for t in tag_text.split(",") if t.strip()]
        new_tags = []
        for t in tags:
            if t in mapping:
                neu = mapping[t]
                if neu and neu not in new_tags:
                    new_tags.append(neu)
            else:
                new_tags.append(t)
        if new_tags != tags:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(", ".join(new_tags))
            changed_files += 1
    return {"ok": True, "changed_files": changed_files}

@app.route("/api/reorder_tags", methods=["POST"])
def api_reorder_tags():
    payload = request.get_json(silent=True) or {}
    new_order = payload.get("order", [])
    if not isinstance(new_order, list) or not new_order:
        return {"ok": False, "error": "无顺序数据"}
    current = get_current_tag_list()
    current_set = set(current)
    new_set = set(str(x) for x in new_order)
    if new_set != current_set:
        return {"ok": False, "error": "标签集合不匹配（缺少或多余）"}
    txt_path = get_txt_path()
    if not txt_path:
        return {"ok": False, "error": "无当前图片"}
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(str(x) for x in new_order))
    return {"ok": True}

if __name__ == "__main__":
    _load_translator()
    time.sleep(0.5)
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        webbrowser.open("http://127.0.0.1:7866")
    app.run(host="0.0.0.0", port=7866, debug=False)
