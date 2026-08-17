from flask import Flask, render_template_string, request, redirect, url_for
import os
from collections import Counter
import webbrowser
import time
import tkinter
from tkinter import filedialog

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
        .btn-replace { background: #fdcb6e; color: #2c2c2e; cursor: pointer; }
        .btn-replace:hover { background: #ffeaa7; }
        
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
    time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:7861")
    app.run(host="0.0.0.0", port=7861, debug=False)
