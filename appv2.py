from flask import Flask, render_template_string, request, redirect, url_for
import os
from collections import Counter
import webbrowser
import time
import threading
import tkinter
from tkinter import filedialog

# ===== 离线翻译（Helsinki-NLP/opus-mt-en-zh，支持 GPU）=====
try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    _MT_OK = True
except Exception:
    _MT_OK = False

_mt_model = None
_mt_tokenizer = None
_mt_device = "cpu"

def _load_translator():
    """启动时加载 opus-mt-en-zh 模型。自动检测 GPU。"""
    global _mt_model, _mt_tokenizer, _mt_device
    if not _MT_OK:
        print("[translate] 未安装 transformers/torch，请 pip install transformers sentencepiece")
        return False
    try:
        # 检测 GPU
        if torch.cuda.is_available():
            _mt_device = "cuda"
            try:
                print(f"[translate] GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                print("[translate] GPU: cuda")
        else:
            _mt_device = "cpu"
            print("[translate] CPU 模式（未检测到 CUDA）")

        # 优先找本地 opus-mt-en-zh 文件夹
        model_id = "Helsinki-NLP/opus-mt-en-zh"
        load_path = None
        need_files = ["config.json", "source.spm", "target.spm",
                      "tokenizer_config.json", "vocab.json"]
        for d in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
            cand = os.path.join(d, "opus-mt-en-zh")
            print(f"[translate] 查找: {cand}  -> {'存在' if os.path.isdir(cand) else '不存在'}")
            if os.path.isdir(cand):
                files = os.listdir(cand)
                print(f"[translate]   文件: {files}")
                missing = [f for f in need_files if f not in files]
                if missing:
                    print(f"[translate]   缺文件: {missing}")
                    continue
                load_path = cand
                break
        if load_path:
            print(f"[translate] 本地模型: {load_path}")
        else:
            print(f"[translate] 没找到完整模型，加载 {model_id}（首次需联网）...")
            load_path = model_id

        # 详细报错：分别定位 tokenizer 和 model 的问题
        try:
            _mt_tokenizer = MarianTokenizer.from_pretrained(load_path)
            print("[translate] tokenizer OK")
        except Exception as e:
            print(f"[translate] tokenizer 失败: {type(e).__name__}: {e}")
            return False
        try:
            _mt_model = MarianMTModel.from_pretrained(load_path).to(_mt_device)
        except Exception as e:
            print(f"[translate] model 失败: {type(e).__name__}: {e}")
            return False
        if _mt_device == "cuda":
            _mt_model = _mt_model.half()  # FP16 省显存加速
        print("[translate] 模型已就绪")
        return True
    except Exception as e:
        print(f"[translate] 加载失败: {e}")
        return False

def translate_en2zh(text):
    if _mt_model is None or not text:
        return ""
    try:
        inputs = _mt_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(_mt_device)
        outs = _mt_model.generate(**inputs, max_length=128, num_beams=2)
        return _mt_tokenizer.decode(outs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[translate error] {e}")
        return ""

app = Flask(__name__)

PATH_RECORD_FILE = "last_path.txt"

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
    <title>选择数据集目录</title>
    <style>
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
    <button class="theme-toggle" id="themeBtn">💡</button>
    <div class="wrap">
        <h2>选择图片数据集文件夹</h2>
        <p class="tip">点击按钮唤起系统文件资源管理器，挑选存放图片标签的文件夹<br>程序会自动记忆上一次打开的目录</p>
        <form method="post" action="/pick_folder">
            <button class="choose-btn" type="submit">📂 浏览文件夹</button>
        </form>
    </div>

    <script>
        const savedTheme = localStorage.getItem("tagEditorTheme") || "dark";
        const body = document.body;
        const themeBtn = document.getElementById("themeBtn");
        body.className = savedTheme;
        themeBtn.onclick = () => {
            const newTheme = body.classList.contains("dark") ? "light" : "dark";
            body.className = newTheme;
            localStorage.setItem("tagEditorTheme", newTheme);
        }
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
    <title>标签手动编辑工具</title>
    <style>
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
    </style>
</head>
<body class="dark">
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
            <button class="change-dir-btn" onclick="openModal()">切换数据集文件夹</button>
        </div>

        <form method="post" class="img-nav-bar">
            <div class="btn-wrap">
                <button formaction="/prev_img" type="submit">上一张</button>
                <button formaction="/next_img" type="submit">下一张</button>
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
                <button class="btn-translate" id="translateBtn" type="button" onclick="doTranslate()">🌐 中文翻译</button>
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

        // 主题切换
        const savedTheme = localStorage.getItem("tagEditorTheme") || "dark";
        const body = document.body;
        const themeBtn = document.getElementById("themeBtn");
        body.className = savedTheme;
        themeBtn.onclick = () => {
            const newTheme = body.classList.contains("dark") ? "light" : "dark";
            body.className = newTheme;
            localStorage.setItem("tagEditorTheme", newTheme);
        }

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

        // ===== 中文翻译（最小实现）=====
        let _zhOn = false;
        async function doTranslate() {
            const btn = document.getElementById('translateBtn');
            if (_zhOn) {
                // 关闭
                _zhOn = false;
                document.querySelectorAll('.tag-zh').forEach(el => el.remove());
                if (btn) { btn.innerText = '🌐 中文翻译'; btn.classList.remove('active'); }
                return;
            }
            _zhOn = true;
            if (btn) { btn.innerText = '⏳ 翻译中...'; btn.disabled = true; }
            // 收集所有未翻译的英文标签
            const items = document.querySelectorAll('.tag-item');
            const need = [];
            items.forEach(it => {
                if (!it.querySelector('.tag-zh')) {
                    const t = it.dataset.origTag;
                    if (t) need.push(t);
                }
            });
            try {
                if (need.length) {
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
            } catch (e) {
                alert('请求失败：' + e.message);
            } finally {
                if (btn) { btn.disabled = false; btn.innerText = '🌐 关闭翻译'; btn.classList.add('active'); }
            }
        }
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
        return render_template_string(EDIT_HTML, img_path="", now=0, total=0, current_tags=[], tag_count={}, raw_tag_text="")
    tag_counter = count_all_tags_frequency()
    tag_list = get_current_tag_list()
    txt_path = get_txt_path()
    raw_text = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    return render_template_string(EDIT_HTML,
        img_path="/img/" + img_list[current_idx],
        now=current_idx + 1,
        total=total_num,
        current_tags=tag_list,
        tag_count=tag_counter,
        raw_tag_text=raw_text
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
    return send_from_directory(CURRENT_IMG_DIR, fname)

@app.route("/prev_img", methods=["POST"])
def prev():
    global current_idx
    if current_idx > 0: current_idx -= 1
    return redirect(url_for("home"))

@app.route("/next_img", methods=["POST"])
def next():
    global current_idx
    if current_idx < len(img_list) - 1: current_idx += 1
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

@app.route("/api/translate")
def api_translate():
    tags = [t for t in request.args.get("tags", "").split(",") if t.strip()]
    out = {}
    for t in tags:
        zh = translate_en2zh(t)
        if zh:
            out[t] = zh
    return {"ok": _mt_model is not None, "translations": out}

if __name__ == "__main__":
    _load_translator()
    time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:7862")
    app.run(host="0.0.0.0", port=7862, debug=False)
