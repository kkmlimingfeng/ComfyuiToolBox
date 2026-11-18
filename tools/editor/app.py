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
# llama.cpp 同一模型上下文不支持多线程并发推理（并发会触发 CUDA error 直接崩掉整个进程），
# 快速切图时页面会并发发起 /api/translate，必须用锁串行化
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

# ---- 翻译/界面配置（保存在脚本同目录 translate_config.json）----
_CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate_config.json")
_translate_cfg = {"target_lang": "Simplified Chinese", "model_src": "auto",
                  "translate_src": "model", "vocab_file": "all",
                  "show_edit_btn": True, "show_del_btn": True, "bb_on": False,
                  "bb_click_add": False,
                  "zh_color": "#6c5ce7", "zh_size": 13}

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
    if _translate_cfg.get("translate_src") not in ("model", "vocab"):
        _translate_cfg["translate_src"] = "model"
    for key in ("show_edit_btn", "show_del_btn", "bb_on", "bb_click_add"):
        _translate_cfg[key] = bool(_translate_cfg.get(key))
    # 词表翻译仅支持简体中文
    if (_translate_cfg.get("translate_src") == "vocab"
            and _translate_cfg.get("target_lang") != "Simplified Chinese"):
        _translate_cfg["target_lang"] = "Simplified Chinese"
    if _translate_cfg.get("target_lang") not in [v for _, v in _MT_LANGS]:
        _translate_cfg["target_lang"] = "Simplified Chinese"
    c = _translate_cfg.get("zh_color")
    if not (isinstance(c, str) and len(c) == 7 and c.startswith("#")):
        _translate_cfg["zh_color"] = "#6c5ce7"
    try:
        _translate_cfg["zh_size"] = min(40, max(9, int(_translate_cfg.get("zh_size"))))
    except (TypeError, ValueError):
        _translate_cfg["zh_size"] = 13

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
    """按需加载 Hy-MT2 翻译模型（第一次点翻译时触发）：有 GPU 自动 GPU 加速，无 GPU 自动 CPU 运行。"""
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

def _unload_translator():
    """关闭翻译时立刻卸载翻译模型，释放显存/内存。"""
    global _mt_model, _mt_device_info
    with _mt_lock:
        if _mt_model is None:
            return
        _mt_model = None
        import gc
        gc.collect()
        _mt_device_info = "未加载"
        print("[translate] 翻译模型已卸载，显存已释放")

# ---- 词表翻译（读取标签超市 tags_merged*.csv 的 zh 列，仅简体中文）----
# 词表里 zh 是人工/词表翻译（zh_mt 是机翻，不用）；查不到的标签由前端显示红色"无法翻译"
_VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "label_shop")
_vocab_lock = threading.Lock()
_vocab = None        # {归一化标签名或别名: 中文翻译}
_vocab_sig = None    # 各 CSV 的 mtime + 词表选择，变了自动重载（跟随标签超市的编辑更新）
_vocab_info = "未加载"

def _vocab_files():
    """label_shop 下可用的词表 CSV（tags_merged*.csv，排除 classified 覆盖表）。"""
    files = []
    try:
        for f in sorted(os.listdir(_VOCAB_DIR)):
            if (f.startswith("tags_merged") and f.endswith(".csv")
                    and f != "tags_merged_classified.csv"):
                files.append(f)
    except OSError:
        pass
    return files

def _vocab_norm(s):
    """标签归一化：去转义反斜杠、下划线视为空格、折叠空白、转小写。
    使 blue_eyes / blue eyes / \\(blue eyes\\) 都能命中词表。"""
    s = s.replace("\\", "").strip().lower().replace("_", " ")
    return " ".join(s.split())

def _load_vocab():
    """懒加载词表（name + aliases → zh）；按配置选择词表文件，CSV 更新后自动重载。
    vocab_file="all" 时合并全部词表（同名保留先出现的），否则只读指定文件。"""
    global _vocab, _vocab_sig, _vocab_info
    with _vocab_lock:
        files = _vocab_files()
        vf = _translate_cfg.get("vocab_file", "all")
        use = files if (vf == "all" or vf not in files) else [vf]
        sig = (vf, tuple(os.path.getmtime(os.path.join(_VOCAB_DIR, f))
                         if os.path.isfile(os.path.join(_VOCAB_DIR, f)) else 0
                         for f in use))
        if _vocab is not None and sig == _vocab_sig:
            return _vocab
        import csv as _csv
        t0 = time.time()
        table = {}
        for f in use:
            p = os.path.join(_VOCAB_DIR, f)
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
                for r in _csv.DictReader(fh):
                    zh = (r.get("zh") or "").strip()
                    if not zh:
                        continue
                    name = (r.get("name") or "").strip()
                    if name:
                        table.setdefault(_vocab_norm(name), zh)
                    for a in (r.get("aliases") or "").split("|"):
                        a = a.strip()
                        if a:
                            table.setdefault(_vocab_norm(a), zh)
        _vocab = table
        _vocab_sig = sig
        _vocab_info = (f"词表：{'全部合并' if vf == 'all' else vf} · {len(table)} 条（标签超市）"
                       if table else "未找到标签超市词表")
        print(f"[translate] {_vocab_info} · 加载 {time.time() - t0:.1f}s")
        return _vocab

def _vocab_lookup(table, text):
    """两级查询：先按归一化原文查；未命中且带括号时去掉括号再查一次
    （词表标签多为无括号形态，如 \\(censored\\) → censored）。"""
    n = _vocab_norm(text)
    zh = table.get(n)
    if zh is None:
        s = n.replace("(", "").replace(")", "")
        if s != n:
            zh = table.get(s)
    return zh

def translate_text(text):
    """按当前配置翻译一段文本。
    词表模式：查标签超市词表（仅简体中文），命中返回译文，未命中返回 None。
    模型模式：走本地 Hy-MT2，失败返回 ""（与旧行为一致）。"""
    if not text:
        return ""
    if _translate_cfg.get("translate_src") == "vocab":
        table = _load_vocab()
        if not table:
            return ""
        return _vocab_lookup(table, text)   # 未命中返回 None → 前端标"无法翻译"
    if _mt_model is None:
        with _mt_lock:  # 懒加载：第一次点翻译时才加载模型
            if _mt_model is None:
                _load_translator()
    if _mt_model is None:
        return ""
    try:
        lang = _translate_cfg.get("target_lang", "Simplified Chinese")
        prompt = f"Translate the following segment into {lang}, without additional explanation.\n\n{text}"
        with _mt_lock:  # 串行化推理，防止并发导致 llama.cpp / CUDA 崩溃
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
    # 三重声明，兼容对 no-store 处理不彻底的浏览器（如联想 SLBrowser 的加速缓存层）
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

PATH_RECORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_path.txt")

def load_last_folder():
    if os.path.exists(PATH_RECORD_FILE):
        # utf-8-sig 兼容外部编辑器写入 BOM 的情况
        with open(PATH_RECORD_FILE, "r", encoding="utf-8-sig") as f:
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
    global img_list, current_idx
    if not os.path.isdir(CURRENT_IMG_DIR):
        img_list = []
        current_idx = 0
        return
    all_files = sorted(os.listdir(CURRENT_IMG_DIR))
    img_list = [f for f in all_files if f.lower().endswith(IMG_SUFFIX)]
    # 文件列表变短（图片被删/目录变化）时钳制索引，避免越界 500
    if current_idx >= len(img_list):
        current_idx = max(0, len(img_list) - 1)

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
        .img-name {
            margin-left:10px;
            font-weight:normal;
            font-size:13px;
            opacity:0.75;
            max-width:320px;
            display:inline-block;
            overflow:hidden;
            text-overflow:ellipsis;
            white-space:nowrap;
            vertical-align:bottom;
        }
        .nav-row { display:flex; gap:8px; align-items:flex-start; flex-wrap:wrap; }
        .jump-line { display:flex; gap:8px; align-items:center; margin-top:8px; }
        .jump-line input {
            width:80px;
            padding:8px;
            border-radius:4px;
            border-width:1px; border-style:solid;
            transition: background 0.3s, color 0.3s, border-color 0.3s;
        }
        body.dark .jump-line input { background:#29292b; color:#e6e6e6; border-color:#3a3a3c; }
        body.light .jump-line input { background:#ffffff; color:#2c2c2e; border-color:#d2d2d7; }
        .btn-jump { background:#0984e3; color:white; }
        .btn-jump:hover { background:#1994f0; }
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
        /* 按钮容器：悬浮显示，右对齐排列；某个按钮隐藏时其余自动右移补位 */
        .tag-acts {
            display: none;
            position: absolute;
            right: 4px;
            top: 50%;
            transform: translateY(-50%);
            gap: 4px;
            flex-direction: row-reverse;   /* 垃圾桶在最右，往左依次是铅笔、加号 */
        }
        .tag-item:hover .tag-acts { display: flex; }
        /* 设置控制显隐 */
        body.hide-edit .tag-item .edit-icon:not(.add-icon):not(.del-icon) { display: none !important; }
        body.hide-del .tag-item .del-icon { display: none !important; }
        body.bb-off .tag-item .add-icon { display: none !important; }

        /* ===== 悬浮小黑板 ===== */
        #blackboard {
            position: fixed; right: 24px; top: 90px; width: 280px; height: 200px;
            min-width: 200px; min-height: 120px;
            z-index: 1600; border-radius: 8px;
            border-width: 1px; border-style: solid;
            box-shadow: 0 6px 24px rgba(0,0,0,0.35);
            display: none; flex-direction: column;
            resize: both; overflow: hidden;   /* 右下角可拖拽调整大小 */
        }
        body.dark #blackboard { background: #232326; border-color: #3a3a3c; }
        body.light #blackboard { background: #ffffff; border-color: #d2d2d7; }
        #bb-head {
            padding: 8px 12px; font-weight: 600; font-size: 14px;
            cursor: move; user-select: none;
        }
        #bb-tags {
            flex: 1; min-height: 60px; overflow-y: auto; padding: 10px;
            display: flex; flex-wrap: wrap; gap: 6px; align-content: flex-start;
        }
        .bb-tag {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 8px; border-radius: 4px; font-size: 13px;
            cursor: grab; user-select: none; max-width: 100%;
            transition: background 0.2s;
        }
        .bb-tag:active { cursor: grabbing; }
        body.dark .bb-tag { background: #374151; }
        body.light .bb-tag { background: #e4e7ec; color: #1f2937; }
        .bb-tag .bb-del { color: #e74c3c; font-weight: 700; cursor: pointer; visibility: hidden; }
        .bb-tag:hover .bb-del { visibility: visible; }
        .bb-tag.dragging { position: fixed; z-index: 9999; opacity: .85; box-shadow: 0 4px 12px rgba(0,0,0,.4); }
        .bb-drag-left { box-shadow: -2px 0 0 0 #74b9ff; }
        .bb-drag-right { box-shadow: 2px 0 0 0 #74b9ff; }
        .bb-placeholder { border: 2px dashed #74b9ff; border-radius: 4px; }
        #bb-foot {
            padding: 6px 12px; font-size: 12px;
            border-top-width: 1px; border-top-style: solid;
        }
        body.dark #bb-foot { border-top-color: #3a3a3c; }
        body.light #bb-foot { border-top-color: #d2d2d7; }
        #bb-empty { opacity: .5; font-size: 12px; }

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
        .tag-zh { color:var(--tag-zh-color, #6c5ce7); font-size:var(--tag-zh-size, 13px); margin-left:6px; }
        .tag-zh.tag-zh-miss { color:#e74c3c; }
        
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
            z-index: 2000;
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
        /* 勾选框行：勾选框在描述前面，同一行对齐不换行 */
        .tcfg-row label.tcfg-check {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0;
            opacity: 1;
            font-size: 14px;
            cursor: pointer;
            white-space: nowrap;
        }
        .tcfg-row label.tcfg-check input[type="checkbox"] { width: 16px; height: 16px; margin: 0; cursor: pointer; flex: none; }
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
        .tcfg-row input[type="color"], .tcfg-row input[type="number"] {
            width: 100%;
            padding: 6px 8px;
            border-radius: 8px;
            border: 1px solid rgba(128,128,128,0.4);
            background: transparent;
            color: inherit;
            box-sizing: border-box;
        }
        .tcfg-inline { display: flex; gap: 14px; }
        .tcfg-inline .tcfg-row { flex: 1; }

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

    <!-- 设置弹窗 -->
    <div class="modal-mask" id="translateCfgModal">
        <div class="modal-box" style="min-width: 420px;">
            <h3 class="modal-title">设置</h3>
            <p class="modal-tip" id="tcfgDevice" style="margin-bottom: 12px;">翻译模型：未加载</p>
            <div class="tcfg-row">
                <label class="tcfg-check"><input type="checkbox" id="tcfgEditBtn"> 显示编辑按钮（悬浮标签时）</label>
            </div>
            <div class="tcfg-row">
                <label class="tcfg-check"><input type="checkbox" id="tcfgDelBtn"> 显示删除按钮（悬浮标签时）</label>
            </div>
            <div class="tcfg-row">
                <label class="tcfg-check"><input type="checkbox" id="tcfgBB"> 显示小黑板</label>
            </div>
            <div class="tcfg-row">
                <label class="tcfg-check"><input type="checkbox" id="tcfgBbClick"> 点击小黑板标签添加到当前图片末尾</label>
            </div>
            <div class="tcfg-row">
                <label>翻译方式</label>
                <select id="tcfgMode" onchange="tcfgModeChange()">
                    <option value="model">模型翻译（AI，支持多语言）</option>
                    <option value="vocab">词表翻译（标签超市词表，仅简体中文）</option>
                </select>
            </div>
            <div class="tcfg-row" id="tcfgVocabRow" style="display:none">
                <label>词表</label>
                <select id="tcfgVocab"></select>
            </div>
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
            <div class="tcfg-inline">
                <div class="tcfg-row">
                    <label>译文颜色</label>
                    <input type="color" id="tcfgColor">
                </div>
                <div class="tcfg-row">
                    <label>译文字号（px，9-40）</label>
                    <input type="number" id="tcfgSize" min="9" max="40" step="1">
                </div>
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
            <div class="progress">进度：{{now}} / {{total}}{% if img_name %}<span class="img-name" title="{{ img_name }}">{{ img_name }}</span>{% endif %}</div>
            <div style="display: flex; gap: 10px;">
                <button class="change-dir-btn" onclick="openTranslateCfg()">⚙️ 设置</button>
                <button class="change-dir-btn" onclick="openModal()">切换数据集文件夹</button>
            </div>
        </div>

        <div class="nav-row">
            <form method="post" class="img-nav-bar" id="navForm">
                <div class="btn-wrap">
                    <button type="submit" formaction="/prev_img">上一张</button>
                    <button type="submit" formaction="/next_img">下一张</button>
                    <button formaction="/del_img_file" class="btn-del-img" type="submit" onclick="return confirm('确认删除当前图片及其标签文件？');">删除图片+标签文件</button>
                </div>
            </form>
            <form method="post" action="/jump_img" class="jump-line">
                <input type="number" name="jump_idx" min="1" max="{{total}}" placeholder="序号">
                <button class="btn-jump" type="submit">跳转</button>
            </form>
        </div>

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
                    <span class="tag-acts"><span class="edit-icon add-icon" onclick="bbAdd(this)" title="加入小黑板">＋</span><span class="edit-icon" onclick="startEdit(this)">✏</span><span class="edit-icon del-icon" onclick="delTag(this)" title="删除该标签">🗑</span></span>
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

    <!-- 悬浮小黑板（常用标签收藏板） -->
    <div id="blackboard">
        <div id="bb-head">📌 小黑板</div>
        <div id="bb-tags"></div>
        <div id="bb-foot">
            <label><input type="checkbox" id="bb-click-add"> 点击标签添加到当前图片末尾</label>
        </div>
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
                document.body.classList.remove('dark', 'light');
                document.body.classList.add(t);
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
                const miss = new Set(data.missing || []);
                items.forEach(it => {
                    const t = it.dataset.origTag;
                    if (t && !it.querySelector('.tag-zh')) {
                        const zh = data.translations[t];
                        // 词表/模型都查不到的标签：显示红色"无法翻译"
                        if (zh || miss.has(t)) {
                            const sp = document.createElement('span');
                            sp.className = 'tag-zh' + (zh ? '' : ' tag-zh-miss');
                            sp.innerText = zh || '无法翻译';
                            it.appendChild(sp);
                        }
                    }
                });
            } else {
                alert('翻译失败：' + (data && data.error || '服务异常'));
            }
        }
        async function doTranslate() {
            const btn = document.getElementById('translateBtn');
            if (_zhOn) {
                // 关闭：清空译文并通知服务端立刻卸载翻译模型
                _zhOn = false;
                sessionStorage.setItem("zhOn", "0");
                document.querySelectorAll('.tag-zh').forEach(el => el.remove());
                if (btn) { btn.innerText = translateBtnText(false); btn.classList.remove('active'); }
                fetch('/api/translate_unload', { method: 'POST' }).catch(() => {});
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
                    applyUIPrefs(d.config);   // 按钮显隐 + 小黑板开关
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

        // ===== 设置 =====
        // 译文颜色/字号：写到 CSS 变量，立即生效
        function applyZhStyle(cfg) {
            if (cfg.zh_color) {
                document.documentElement.style.setProperty('--tag-zh-color', cfg.zh_color);
            }
            const size = parseInt(cfg.zh_size);
            if (!isNaN(size)) {
                document.documentElement.style.setProperty('--tag-zh-size', size + 'px');
            }
        }
        // 界面偏好：编辑/删除按钮显隐、小黑板显隐、小黑板点击添加开关
        function applyUIPrefs(cfg) {
            document.body.classList.toggle('hide-edit', !cfg.show_edit_btn);
            document.body.classList.toggle('hide-del', !cfg.show_del_btn);
            document.body.classList.toggle('bb-off', !cfg.bb_on);
            const bbClick = document.getElementById('bb-click-add');
            if (bbClick) bbClick.checked = cfg.bb_click_add === true;
            if (window.__bbApply) window.__bbApply(!!cfg.bb_on);
        }
        // 翻译方式切换：词表模式锁定简体中文、禁用模型下载源、显示词表选择
        function tcfgModeChange() {
            const vocab = document.getElementById('tcfgMode').value === 'vocab';
            const lang = document.getElementById('tcfgLang');
            if (vocab) lang.value = 'Simplified Chinese';
            lang.disabled = vocab;
            document.getElementById('tcfgSrc').disabled = vocab;
            document.getElementById('tcfgVocabRow').style.display = vocab ? '' : 'none';
            const dev = document.getElementById('tcfgDevice');
            if (vocab && tcfgVocabInfo) dev.innerText = tcfgVocabInfo;
            else dev.innerText = '翻译模型：' + (tcfgModelLoaded ? '已加载 · ' + tcfgModelDevice : tcfgModelDevice);
        }
        let tcfgVocabInfo = '', tcfgModelDevice = '未加载', tcfgModelLoaded = false;
        async function openTranslateCfg() {
            try {
                const resp = await fetch('/api/translate_config');
                const data = await resp.json();
                if (data && data.config) {
                    document.getElementById('tcfgMode').value = data.config.translate_src || 'model';
                    document.getElementById('tcfgLang').value = data.config.target_lang;
                    document.getElementById('tcfgSrc').value = data.config.model_src;
                    document.getElementById('tcfgColor').value = data.config.zh_color;
                    document.getElementById('tcfgSize').value = data.config.zh_size;
                    document.getElementById('tcfgEditBtn').checked = data.config.show_edit_btn !== false;
                    document.getElementById('tcfgDelBtn').checked = data.config.show_del_btn !== false;
                    document.getElementById('tcfgBB').checked = data.config.bb_on === true;
                    document.getElementById('tcfgBbClick').checked = data.config.bb_click_add === true;
                }
                // 词表下拉：全部合并 + 各词表文件
                const vocabSel = document.getElementById('tcfgVocab');
                vocabSel.innerHTML = '';
                const files = (data && data.vocab_files) || [];
                const optAll = document.createElement('option');
                optAll.value = 'all'; optAll.textContent = '全部合并';
                vocabSel.appendChild(optAll);
                files.forEach(f => {
                    const o = document.createElement('option');
                    o.value = f; o.textContent = f;
                    vocabSel.appendChild(o);
                });
                if (data && data.config) {
                    vocabSel.value = (data.config.vocab_file === 'all' || files.includes(data.config.vocab_file))
                        ? data.config.vocab_file : 'all';
                }
                if (data) {
                    tcfgVocabInfo = data.vocab_info || '';
                    tcfgModelDevice = data.device || '未加载';
                    tcfgModelLoaded = !!data.loaded;
                }
            } catch (e) { /* 读取失败也允许打开面板修改 */ }
            tcfgModeChange();
            document.getElementById('translateCfgModal').style.display = 'flex';
        }
        function closeTranslateCfg() {
            document.getElementById('translateCfgModal').style.display = 'none';
        }
        async function saveTranslateCfg() {
            const langSel = document.getElementById('tcfgLang');
            const src = document.getElementById('tcfgSrc').value;
            const mode = document.getElementById('tcfgMode').value;
            const zhColor = document.getElementById('tcfgColor').value;
            const zhSize = parseInt(document.getElementById('tcfgSize').value);
            try {
                const resp = await fetch('/api/translate_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_lang: langSel.value, model_src: src,
                                           translate_src: mode,
                                           vocab_file: document.getElementById('tcfgVocab').value,
                                           show_edit_btn: document.getElementById('tcfgEditBtn').checked,
                                           show_del_btn: document.getElementById('tcfgDelBtn').checked,
                                           bb_on: document.getElementById('tcfgBB').checked,
                                           bb_click_add: document.getElementById('tcfgBbClick').checked,
                                           zh_color: zhColor,
                                           zh_size: isNaN(zhSize) ? 13 : zhSize })
                });
                const data = await resp.json();
                if (data && data.ok && data.config) {
                    _targetLangLabel = langSel.selectedOptions[0].text;
                    const btn = document.getElementById('translateBtn');
                    if (btn && !_zhOn) btn.innerText = translateBtnText(false);
                    applyZhStyle(data.config);
                    applyUIPrefs(data.config);
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
                    applyZhStyle(data.config);
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
            // 供小黑板插入新标签后绑定同样的拖拽排序
            window.__setupDrag = setupDrag;
        })();

        // ===== 悬浮小黑板（常用标签收藏板） =====
        (function() {
            const KEY_TAGS = 'bbTags', KEY_POS = 'bbPos', KEY_SIZE = 'bbSize';
            const LONG_PRESS_MS = 350, MOVE_THRESHOLD = 5;
            const panel = document.getElementById('blackboard');
            const tagBox = document.getElementById('bb-tags');
            const tagGroup = document.querySelector('.tag-group');
            let dragSuppress = false;   // 拖拽结束后的 click 不当作点击添加

            function load() {
                try { return JSON.parse(localStorage.getItem(KEY_TAGS)) || []; }
                catch (e) { return []; }
            }
            function save(list) {
                try { localStorage.setItem(KEY_TAGS, JSON.stringify(list)); } catch (e) {}
            }
            function esc(s) {
                const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
            }

            function saveGeom() {
                // 面板隐藏时 rect 全 0（ResizeObserver 初始回调也会触发），不能覆盖已保存的位置
                if (!panel.offsetParent) return;
                try {
                    const r = panel.getBoundingClientRect();
                    localStorage.setItem(KEY_POS, JSON.stringify({x: r.left, y: r.top}));
                    // 尺寸异常（如被旧 bug 存成 1x1）时不写入，避免坏数据循环
                    if (r.width >= 160 && r.height >= 80) {
                        localStorage.setItem(KEY_SIZE, JSON.stringify({w: r.width, h: r.height}));
                    }
                } catch (e) {}
            }
            function applyGeom() {
                // 开启时恢复上次关闭时的位置和大小
                try {
                    const pos = JSON.parse(localStorage.getItem(KEY_POS) || 'null');
                    if (pos) {
                        panel.style.left = Math.max(0, pos.x) + 'px';
                        panel.style.top = Math.max(0, pos.y) + 'px';
                        panel.style.right = 'auto';
                    }
                    const size = JSON.parse(localStorage.getItem(KEY_SIZE) || 'null');
                    if (size && size.w >= 160 && size.h >= 80) {
                        panel.style.width = size.w + 'px';
                        panel.style.height = size.h + 'px';
                    }
                } catch (e) {}
            }

            function openBB() {
                applyGeom();
                panel.style.display = 'flex';
                // 强制最小尺寸：localStorage 残留的 1x1 等异常尺寸在此自愈
                if (panel.offsetWidth < 160) panel.style.width = '280px';
                if (panel.offsetHeight < 80) panel.style.height = '200px';
                // 显示后钳制到视口内，防止 localStorage 里的异常坐标让黑板跑到屏幕外
                const r = panel.getBoundingClientRect();
                const maxX = Math.max(0, window.innerWidth - 60);
                const maxY = Math.max(0, window.innerHeight - 50);
                if (r.left > maxX || r.top > maxY) {
                    panel.style.left = Math.min(r.left, maxX) + 'px';
                    panel.style.top = Math.min(r.top, maxY) + 'px';
                }
            }
            function closeBB() {
                saveGeom();
                panel.style.display = 'none';
            }
            // 拖右下角调整大小/移动后立即持久化，刷新页面也保持
            if (window.ResizeObserver) new ResizeObserver(saveGeom).observe(panel);

            // 设置里的"显示小黑板"开关（applyUIPrefs 调用）
            window.__bbApply = function(on) {
                if (on) openBB(); else closeBB();
            };
            // 右上角 ✕ 已移除：小黑板显隐统一走设置里的"显示小黑板"开关

            // 主编辑区标签上的 ＋：收藏到小黑板
            window.bbAdd = function(btn) {
                const tag = btn.closest('.tag-item').dataset.origTag;
                const list = load();
                if (list.some(t => tagKey(t) === tagKey(tag))) { openBB(); return; }
                list.push(tag);
                save(list);
                render();
                openBB();
            };

            window.syncTextarea = function() {
                const items = document.querySelectorAll('.tag-group .tag-item');
                const ta = document.querySelector('textarea[name="full_tag_text"]');
                if (ta) ta.value = Array.from(items).map(it => it.dataset.origTag).join(', ');
            };
            const syncTextarea = window.syncTextarea;

            function makeTagItem(tag, freq) {
                const div = document.createElement('div');
                div.className = 'tag-item';
                div.dataset.origTag = tag;
                const cb = document.createElement('input');
                cb.type = 'checkbox'; cb.name = 'del_tag_list'; cb.value = tag;
                const txt = document.createElement('span');
                txt.className = 'tag-text'; txt.textContent = tag;
                const fq = document.createElement('span');
                fq.className = 'freq'; fq.textContent = '[' + freq + '次]';
                const add = document.createElement('span');
                add.className = 'edit-icon add-icon'; add.textContent = '＋';
                add.title = '加入小黑板'; add.onclick = () => bbAdd(add);
                const pen = document.createElement('span');
                pen.className = 'edit-icon'; pen.textContent = '✏';
                pen.onclick = () => startEdit(pen);
                const del = document.createElement('span');
                del.className = 'edit-icon del-icon'; del.textContent = '🗑';
                del.title = '删除该标签'; del.onclick = () => delTag(del);
                const acts = document.createElement('span');
                acts.className = 'tag-acts';
                acts.append(add, pen, del);
                div.append(cb, txt, fq, acts);
                return div;
            }

            function render() {
                const list = load();
                tagBox.innerHTML = '';
                if (!list.length) {
                    const tip = document.createElement('span');
                    tip.id = 'bb-empty';
                    tip.textContent = '点击主编辑区标签上的 ＋ 收藏到这里';
                    tagBox.appendChild(tip);
                    return;
                }
                list.forEach(tag => {
                    const el = document.createElement('div');
                    el.className = 'bb-tag';
                    el.dataset.tag = tag;
                    const del = document.createElement('span');
                    del.className = 'bb-del'; del.title = '从小黑板移除'; del.textContent = '−';
                    const txt = document.createElement('span');
                    txt.className = 'bb-text'; txt.textContent = tag;
                    el.append(del, txt);
                    tagBox.appendChild(el);
                });
                setupBB();
            }

            function setupBB() {
                tagBox.querySelectorAll('.bb-tag').forEach(item => {
                    item.querySelector('.bb-del').onclick = () => {
                        save(load().filter(t => t !== item.dataset.tag));
                        render();
                    };
                    bindDrag(item);
                });
            }

            // 点击标签（开关开启且非拖拽）：追加到当前图片标签列表末尾
            tagBox.addEventListener('click', e => {
                if (dragSuppress) { dragSuppress = false; return; }
                if (e.target.classList.contains('bb-del')) return;
                const item = e.target.closest('.bb-tag');
                if (!item) return;
                if (!document.getElementById('bb-click-add').checked) return;
                // 查重：当前标签列表里已有相同标签则不添加（红色闪烁提示）
                const addTag = item.dataset.tag;
                const exists = Array.from(document.querySelectorAll('.tag-group .tag-item'))
                    .some(el => tagKey(el.dataset.origTag || '') === tagKey(addTag));
                if (exists) {
                    item.style.background = '#e17055';
                    setTimeout(() => { item.style.background = ''; }, 600);
                    return;
                }
                fetch('/api/append_tag', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tag: addTag})
                }).then(r => r.json()).then(d => {
                    if (d && d.ok) {
                        const ni = makeTagItem(item.dataset.tag, d.freq || 0);
                        tagGroup.appendChild(ni);
                        // 补上拖拽绑定，否则新加的标签要刷新后才能拖动排序
                        if (window.__setupDrag) window.__setupDrag(ni);
                        syncTextarea();
                        item.style.background = '#00b894';
                        setTimeout(() => { item.style.background = ''; }, 400);
                        // 翻译开启中：给新标签补译文
                        if (typeof fetchAndFillTranslations === 'function') {
                            fetchAndFillTranslations().catch(() => {});
                        }
                    } else {
                        alert('添加失败：' + ((d && d.error) || '未知错误'));
                    }
                }).catch(err => alert('请求失败：' + err.message));
            });

            // 点击添加开关：切换后即时持久化到配置（切图刷新后不丢）
            const bbClickChk = document.getElementById('bb-click-add');
            if (bbClickChk) {
                bbClickChk.addEventListener('change', () => {
                    fetch('/api/translate_config', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({bb_click_add: bbClickChk.checked})
                    }).catch(() => {});
                });
            }

            // ---- 小黑板标签长按拖拽：黑板内排序 / 拖到主编辑区插入鼠标位置 ----
            let lpTimer = null, sx = 0, sy = 0, dragging = false;
            let dragged = null, offX = 0, offY = 0, ph = null, phParent = null, phNext = null;

            function clearIndicators() {
                document.querySelectorAll('.drag-over-left, .drag-over-right, .bb-drag-left, .bb-drag-right')
                    .forEach(el => el.classList.remove('drag-over-left', 'drag-over-right', 'bb-drag-left', 'bb-drag-right'));
            }
            function pickTarget(x, y, cls) {
                for (const el of document.elementsFromPoint(x, y)) {
                    if (el.classList && el.classList.contains(cls) && el !== dragged && el !== ph) return el;
                }
                return null;
            }

            function startDrag(item, x, y) {
                dragging = true;
                dragSuppress = true;
                dragged = item;
                const rect = item.getBoundingClientRect();
                offX = x - rect.left; offY = y - rect.top;
                ph = document.createElement('div');
                ph.className = 'bb-placeholder';
                ph.style.width = rect.width + 'px'; ph.style.height = rect.height + 'px';
                phParent = item.parentNode; phNext = item.nextSibling;
                item.parentNode.insertBefore(ph, item);
                item.style.position = 'fixed';
                item.style.left = rect.left + 'px'; item.style.top = rect.top + 'px';
                item.style.width = rect.width + 'px';
                item.classList.add('dragging');
                document.body.style.userSelect = 'none';
            }

            function endDrag(x, y) {
                const mainTarget = pickTarget(x, y, 'tag-item');
                const bbTarget = pickTarget(x, y, 'bb-tag');
                if (mainTarget && tagGroup.contains(mainTarget)) {
                    // 插入主编辑区：按鼠标在目标标签的左右侧计算 index
                    clearIndicators();
                    restore();
                    const items = Array.from(document.querySelectorAll('.tag-group .tag-item'));
                    const rect = mainTarget.getBoundingClientRect();
                    let index = items.indexOf(mainTarget) + (x < rect.left + rect.width / 2 ? 0 : 1);
                    // 先取出标签文本：endDrag 返回时 dragged 会被置空，异步回调里不能再读它
                    const draggedTag = dragged.dataset.tag;
                    fetch('/api/insert_tag', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({tag: draggedTag, index: index})
                    }).then(r => r.json()).then(d => {
                        if (d && d.ok) {
                            const item = makeTagItem(draggedTag, d.freq || 0);
                            if (index >= items.length) tagGroup.appendChild(item);
                            else tagGroup.insertBefore(item, items[index]);
                            if (window.__setupDrag) window.__setupDrag(item);
                            syncTextarea();
                        } else {
                            alert('插入失败：' + ((d && d.error) || '未知错误'));
                        }
                    }).catch(err => alert('请求失败：' + err.message));
                } else if (bbTarget) {
                    // 黑板内排序
                    const rect = bbTarget.getBoundingClientRect();
                    if (x < rect.left + rect.width / 2) bbTarget.parentNode.insertBefore(ph, bbTarget);
                    else bbTarget.parentNode.insertBefore(ph, bbTarget.nextSibling);
                    ph.parentNode.replaceChild(dragged, ph);
                    save(Array.from(tagBox.querySelectorAll('.bb-tag')).map(el => el.dataset.tag));
                    cleanup();
                    render();
                } else {
                    restore();   // 落点无效，回到原位
                }
                dragged = null; dragging = false;
                document.body.style.userSelect = '';
                clearIndicators();
            }

            function restore() {
                if (ph && ph.parentNode) ph.parentNode.removeChild(ph);
                if (dragged && phParent) {
                    dragged.classList.remove('dragging');
                    dragged.style.position = ''; dragged.style.left = '';
                    dragged.style.top = ''; dragged.style.width = '';
                    phParent.insertBefore(dragged, phNext);
                }
                ph = null;
            }

            function cleanup() {
                if (dragged) {
                    dragged.classList.remove('dragging');
                    dragged.style.position = ''; dragged.style.left = '';
                    dragged.style.top = ''; dragged.style.width = '';
                }
                ph = null; phParent = null; phNext = null;
            }

            function bindDrag(item) {
                item.addEventListener('mousedown', e => {
                    if (e.button !== 0) return;
                    if (e.target.classList.contains('bb-del')) return;
                    sx = e.clientX; sy = e.clientY;
                    lpTimer = setTimeout(() => startDrag(item, e.clientX, e.clientY), LONG_PRESS_MS);
                });
            }

            document.addEventListener('mousemove', e => {
                if (lpTimer && !dragging) {
                    if (Math.abs(e.clientX - sx) > MOVE_THRESHOLD || Math.abs(e.clientY - sy) > MOVE_THRESHOLD) {
                        clearTimeout(lpTimer); lpTimer = null;
                    }
                }
                if (!dragging) return;
                e.preventDefault();
                dragged.style.left = (e.clientX - offX) + 'px';
                dragged.style.top = (e.clientY - offY) + 'px';
                clearIndicators();
                const mainTarget = pickTarget(e.clientX, e.clientY, 'tag-item');
                const bbTarget = pickTarget(e.clientX, e.clientY, 'bb-tag');
                if (mainTarget && tagGroup.contains(mainTarget)) {
                    const rect = mainTarget.getBoundingClientRect();
                    mainTarget.classList.add(e.clientX < rect.left + rect.width / 2 ? 'drag-over-left' : 'drag-over-right');
                } else if (bbTarget) {
                    const rect = bbTarget.getBoundingClientRect();
                    bbTarget.classList.add(e.clientX < rect.left + rect.width / 2 ? 'bb-drag-left' : 'bb-drag-right');
                }
            });

            document.addEventListener('mouseup', e => {
                if (lpTimer) { clearTimeout(lpTimer); lpTimer = null; }
                if (dragging) endDrag(e.clientX, e.clientY);
            });

            // ---- 面板整体移动（按住标题栏） ----
            const head = document.getElementById('bb-head');
            let mvTimer = null, mvOff = null;
            head.addEventListener('mousedown', e => {
                if (e.button !== 0) return;
                const rect = panel.getBoundingClientRect();
                mvOff = {x: e.clientX - rect.left, y: e.clientY - rect.top};
                const move = ev => {
                    panel.style.left = Math.max(0, ev.clientX - mvOff.x) + 'px';
                    panel.style.top = Math.max(0, ev.clientY - mvOff.y) + 'px';
                    panel.style.right = 'auto';
                };
                const up = () => {
                    document.removeEventListener('mousemove', move);
                    document.removeEventListener('mouseup', up);
                    try {
                        const r = panel.getBoundingClientRect();
                        localStorage.setItem(KEY_POS, JSON.stringify({x: r.left, y: r.top}));
                    } catch (err) {}
                };
                document.addEventListener('mousemove', move);
                document.addEventListener('mouseup', up);
                e.preventDefault();
            });

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', render);
            } else {
                render();
            }
        })();

        // 删除按钮：直接从当前图片删除该标签（悬浮垃圾桶）
        function delTag(btn) {
            const item = btn.closest('.tag-item');
            const tag = item.dataset.origTag;
            fetch('/api/delete_tag', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tag: tag})
            }).then(r => r.json()).then(d => {
                if (d && d.ok) {
                    item.remove();
                    window.syncTextarea();
                } else {
                    alert('删除失败：' + ((d && d.error) || '未知错误'));
                }
            }).catch(e => alert('请求失败：' + e.message));
        }

        // ---- 标签去重（忽略大小写、下划线/空格与转义括号差异）----
        function tagKey(s) {
            s = s.replace(/\\\\([()])/g, '$1').trim().toLowerCase();
            return s.replace(/_/g, ' ').split(/\\s+/).join(' ');
        }
        // 每张图片加载后对 textarea 自动去重：重复标签只保留首次出现的
        function dedupTextarea() {
            const ta = document.querySelector('textarea[name="full_tag_text"]');
            if (!ta || !ta.value.trim()) return;
            const seen = new Set(), out = [];
            for (const p of ta.value.split(',')) {
                const t = p.trim();
                if (!t) continue;
                const k = tagKey(t);
                if (seen.has(k)) continue;
                seen.add(k);
                out.push(t);
            }
            ta.value = out.join(', ');
        }
        dedupTextarea();
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
        return render_template_string(EDIT_HTML, img_path="", now=0, total=0, img_name="", current_tags=[], tag_count={}, raw_tag_text="", mt_langs=_MT_LANGS)
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
        img_name=img_list[current_idx],
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
    if not txt_path:
        return redirect(url_for("home"))
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
        "img_name": img_list[idx],
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

@app.route("/jump_img", methods=["POST"])
def jump_img():
    global current_idx
    raw = request.form.get("jump_idx", "").strip()
    try:
        idx = int(float(raw)) - 1
    except ValueError:
        return redirect(url_for("home"))
    if img_list:
        current_idx = max(0, min(idx, len(img_list) - 1))
    return redirect(url_for("home"))

@app.route("/save_all_text", methods=["POST"])
def save_full_text():
    text = request.form.get("full_tag_text", "").strip()
    txt_path = get_txt_path()
    if not txt_path:
        return redirect(url_for("home"))
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
    if not txt_path:
        return redirect(url_for("home"))
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
        if not txt_path:
            return redirect(url_for("home"))
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
    out, missing = {}, []
    err = ""
    vocab_mode = _translate_cfg.get("translate_src") == "vocab"
    if vocab_mode and not _load_vocab():
        err = "未找到标签超市词表（tools/label_shop/tags_merged*.csv）"
        missing = list(tags)
    else:
        for t in tags:
            zh = translate_text(t)
            if zh:
                out[t] = zh
            else:
                missing.append(t)
    return {"ok": not err, "translations": out, "missing": missing,
            "error": err,
            "device": (_vocab_info if vocab_mode else _mt_device_info)}

@app.route("/api/translate_unload", methods=["POST"])
def api_translate_unload():
    _unload_translator()
    # 词表模式下一并释放词表内存（下次翻译按 mtime 自动重载）
    global _vocab, _vocab_sig, _vocab_info
    if _translate_cfg.get("translate_src") == "vocab":
        with _vocab_lock:
            _vocab, _vocab_sig, _vocab_info = None, None, "未加载"
    return {"ok": True, "loaded": _mt_model is not None}

@app.route("/api/translate_config", methods=["GET"])
def api_get_tcfg():
    vocab_info = ""
    if _translate_cfg.get("translate_src") == "vocab":
        _load_vocab()   # 词表模式下预加载，便于设置窗口显示词表状态
        vocab_info = _vocab_info
    return {"ok": True, "config": _translate_cfg,
            "device": _mt_device_info, "loaded": _mt_model is not None,
            "vocab_info": vocab_info, "vocab_files": _vocab_files()}

@app.route("/api/translate_config", methods=["POST"])
def api_set_tcfg():
    payload = request.get_json(silent=True) or {}
    lang = payload.get("target_lang")
    src = payload.get("model_src")
    if isinstance(lang, str) and lang in [v for _, v in _MT_LANGS]:
        _translate_cfg["target_lang"] = lang
    if src in ("auto", "hf", "mirror"):
        _translate_cfg["model_src"] = src
    if payload.get("translate_src") in ("model", "vocab"):
        _translate_cfg["translate_src"] = payload["translate_src"]
    vf = payload.get("vocab_file")
    if vf == "all" or (isinstance(vf, str) and vf in _vocab_files()):
        _translate_cfg["vocab_file"] = vf
    for key in ("show_edit_btn", "show_del_btn", "bb_on", "bb_click_add"):
        if key in payload:
            _translate_cfg[key] = bool(payload[key])
    # 词表翻译仅支持简体中文
    if (_translate_cfg.get("translate_src") == "vocab"
            and _translate_cfg.get("target_lang") != "Simplified Chinese"):
        _translate_cfg["target_lang"] = "Simplified Chinese"
    c = payload.get("zh_color")
    if (isinstance(c, str) and len(c) == 7 and c.startswith("#")
            and all(ch in "0123456789abcdefABCDEF" for ch in c[1:])):
        _translate_cfg["zh_color"] = c
    try:
        _translate_cfg["zh_size"] = min(40, max(9, int(payload.get("zh_size"))))
    except (TypeError, ValueError):
        pass
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

def _tag_key(s):
    """标签归一化：去转义括号、忽略大小写与下划线/空白差异，用于重复判断。"""
    s = str(s).replace("\\(", "(").replace("\\)", ")").strip().lower()
    return " ".join(s.replace("_", " ").split())

@app.route("/api/append_tag", methods=["POST"])
def api_append_tag():
    """小黑板：把一个标签追加到当前图片标签列表末尾。"""
    payload = request.get_json(silent=True) or {}
    tag = (payload.get("tag") or "").strip()
    if not tag:
        return {"ok": False, "error": "缺少标签"}
    txt_path = get_txt_path()
    if not txt_path:
        return {"ok": False, "error": "无当前图片"}
    tags = get_current_tag_list()
    key = _tag_key(tag)
    if any(_tag_key(t) == key for t in tags):
        return {"ok": False, "error": "标签已存在"}
    tags.append(tag)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))
    return {"ok": True, "freq": count_all_tags_frequency().get(tag, 0)}

@app.route("/api/insert_tag", methods=["POST"])
def api_insert_tag():
    """小黑板：把一个标签插入当前图片标签列表的指定位置。"""
    payload = request.get_json(silent=True) or {}
    tag = (payload.get("tag") or "").strip()
    if not tag:
        return {"ok": False, "error": "缺少标签"}
    txt_path = get_txt_path()
    if not txt_path:
        return {"ok": False, "error": "无当前图片"}
    tags = get_current_tag_list()
    try:
        index = int(payload.get("index", len(tags)))
    except (TypeError, ValueError):
        index = len(tags)
    index = max(0, min(index, len(tags)))
    key = _tag_key(tag)
    if any(_tag_key(t) == key for t in tags):
        return {"ok": False, "error": "标签已存在"}
    tags.insert(index, tag)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))
    return {"ok": True, "freq": count_all_tags_frequency().get(tag, 0)}

@app.route("/api/delete_tag", methods=["POST"])
def api_delete_tag():
    """从当前图片标签列表中删除指定标签（全部同名项）。"""
    payload = request.get_json(silent=True) or {}
    tag = (payload.get("tag") or "").strip()
    if not tag:
        return {"ok": False, "error": "缺少标签"}
    txt_path = get_txt_path()
    if not txt_path:
        return {"ok": False, "error": "无当前图片"}
    tags = get_current_tag_list()
    new_tags = [t for t in tags if t != tag]
    if new_tags == tags:
        return {"ok": False, "error": "标签不存在"}
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(new_tags))
    return {"ok": True}

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
    time.sleep(0.5)
    if os.environ.get("TOOL_NO_BROWSER") != "1":  # 被门户拉起时不自动开浏览器
        webbrowser.open("http://127.0.0.1:7866")
    app.run(host="0.0.0.0", port=7866, debug=False)
