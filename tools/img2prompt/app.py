# -*- coding: utf-8 -*-
"""
自然语言提示词反推工具（单文件版）
- 上传单张图片或整个文件夹，用本地 Qwen3-VL-4B-Instruct-Uncensored（GGUF）反推文生图提示词（支持中文/英文）
- llama.cpp 推理（llama-cpp-python），主模型 + mmproj 视觉投影，自动 GPU offload
- 模型放在统一启动器根目录 model/ 下，点击页面"加载模型"后才载入显存，卸载即释放
- 系统提示词支持多方案：页面可切换 / 新增 / 删除方案，离线保存在 prompts.json
- 浏览器自动打开 http://127.0.0.1:7867
"""

import base64
import gc
import io
import json
import os
import threading
import time
import traceback
import webbrowser

from PIL import Image
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # 统一启动器根目录（脚本在 tools/img2prompt/ 下，往上两级）
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SYS_FILE = os.path.join(BASE_DIR, "system_prompt.json")   # 系统提示词持久化
PORT = 7867
MAX_NEW_TOKENS = 512
N_CTX = 8192                 # 上下文长度（需容纳图像视觉 token + 提示词 + 输出）
DEFAULT_RES = 1536           # 默认图像长边上限：768 快 / 1024 均衡 / 1536 高清

# 可选视觉模型：自动扫描 model/ 目录，按文件名配对主模型与 mmproj 视觉投影
# 命名约定：mmproj 文件名中 "mmproj" 字段紧跟模型名，如
#   主模型  <模型名>-<量化>.gguf
#   mmproj  <模型名>-mmproj-<量化>.gguf
def scan_models():
    """扫描 model/ 下的 GGUF，返回 {key: {label, model, mmproj}}。key 为主模型文件名（去 .gguf）。"""
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    mmprojs = {}   # 前缀(模型名) -> mmproj 文件名
    mains = []
    for f in os.listdir(MODEL_DIR):
        if not f.lower().endswith(".gguf"):
            continue
        stem = f[:-5]
        idx = stem.lower().find("mmproj")
        if idx > 0:
            mmprojs[stem[:idx].rstrip(".-_ ")] = f
        else:
            mains.append(f)
    for f in sorted(mains):
        stem = f[:-5]
        best = None   # 取最长匹配前缀的 mmproj
        for prefix in mmprojs:
            if stem.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        if not best:
            print(f"[scan] 跳过 {f}（未找到配对的 mmproj 文件）")
            continue
        mp = mmprojs[best]
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024**3
        models[stem] = {
            "label": f"{stem}（≈{size:.1f}G）",
            "model": f,
            "mmproj": mp,
        }
    return models


DEFAULT_SYS_PROMPT = (
    "You are an exceptionally capable, precise, and helpful multimodal AI assistant that excels "
    "at deeply understanding and richly describing images, scenes, and any visual content. "
    "Follow the user's instructions exactly and answer in the same language as the user's request."
)

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu"

# INSTRUCTION_ZH = (
#     "分析这张图片，并将其转换为适合直接用于图像生成的高密度中文提示词。首先概括图片的媒介、视觉风格或艺术表现形式，"
#     "然后描述主体及其关键视觉特征，并补充光线、构图、视角、色彩、环境和其他重要可见细节。"
#     "使用自然但精炼的描述性短语，将信息紧密组合成一整段提示词，不要写解释、故事或分析过程，"
#     "不要使用项目符号、标题或标签列表。只描述图片中实际可见的内容，不要虚构细节。"
# )
INSTRUCTION_ZH = ("""分析图片，并按照下面的固定结构输出中文图像生成提示词：

女性人物：
详细描述女性的外貌、发型、脸部特征、服装、身体姿势、四肢位置、表情，以及她与其他人物和环境之间的空间关系。

男性人物：
详细描述男性的外貌、发型、服装、身体姿势、四肢位置、表情，以及他与其他人物之间的空间关系。

其他人物：
如果画面中存在第三个或更多人物，分别描述其外貌、位置、姿势、动作和与其他人物的关系。

整体画面特征：
描述整体构图、镜头视角、拍摄距离、光线、色彩、环境、背景、材质、氛围、画面风格以及视觉重点。

要求：
1. 必须严格按照上述分类输出。
2. 每个人物单独描述，不要把不同人物的特征混在一起。
3. 重点描述可见的视觉信息、人物位置、姿势、动作和相互关系。
4. 使用具体、连续、自然的中文描述，不要输出 Danbooru 标签。
5. 不要解释分析过程，只输出最终提示词。
6. 图片中无法确认的内容不要自行补充。""")

INSTRUCTION_EN = (
    "Look at this image carefully. Write ONE English text-to-image prompt that could be used with "
    "an AI image generator (such as FLUX.1 Krea) to recreate this image as closely as possible.\n"
    "Requirements:\n"
    "- Describe the main subject with its key visual details, then the scene/background, composition "
    "and camera angle, lighting, color palette, and art style or photographic style.\n"
    "- Use natural, fluent English in a single paragraph, no line breaks, no bullet points, no markdown.\n"
    "- Output ONLY the prompt text itself. No explanations, no quotes, no prefix like 'Prompt:'."
)
INSTRUCTIONS = {"zh": INSTRUCTION_ZH, "en": INSTRUCTION_EN}

# ---------------------------------------------------------------- 模型加载 ---

_model = None
_state = {"status": "unloaded",
          "device": "GPU（llama.cpp 自动 offload）" if DEVICE == "cuda" else "CPU",
          "model": None, "error": None}  # 模型默认不自动加载
_infer_lock = threading.Lock()      # llama.cpp 同一上下文不支持并发推理，必须串行化
_model_op_lock = threading.Lock()   # 串行化 加载/卸载 操作

# ---- 系统提示词多方案（可切换 / 新增 / 删除，离线保存在 prompts.json）----
PRESETS_FILE = os.path.join(BASE_DIR, "prompts.json")
BUILTIN_PRESETS = {          # 出厂方案（"恢复默认"按方案名还原，自定义方案还原为 DEFAULT_SYS_PROMPT）
    "默认人设": DEFAULT_SYS_PROMPT,
    "中文结构化": INSTRUCTION_ZH,
    "英文提示词": INSTRUCTION_EN,
}
_prompt_presets = dict(BUILTIN_PRESETS)   # {方案名: 系统提示词内容}
_prompt_current = "默认人设"


def _load_presets():
    """加载 prompts.json；旧版单方案 system_prompt.json 存在时迁移为默认方案内容"""
    global _prompt_presets, _prompt_current
    _prompt_presets = dict(BUILTIN_PRESETS)
    _prompt_current = "默认人设"
    try:
        if os.path.isfile(PRESETS_FILE):
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            p = d.get("presets")
            if isinstance(p, dict) and p:
                cleaned = {str(k).strip(): str(v)
                           for k, v in p.items() if str(k).strip()}
                if cleaned:
                    _prompt_presets = cleaned
                    cur = str(d.get("current") or "").strip()
                    _prompt_current = cur if cur in cleaned else next(iter(cleaned))
        elif os.path.isfile(SYS_FILE):
            with open(SYS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("prompt"), str) and d["prompt"].strip():
                _prompt_presets["默认人设"] = d["prompt"]
    except Exception as e:
        print(f"[sys] 读取提示词方案失败: {e}")


def _save_presets():
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump({"current": _prompt_current, "presets": _prompt_presets},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[sys] 保存提示词方案失败: {e}")


def _current_sys_prompt():
    return _prompt_presets.get(_prompt_current, DEFAULT_SYS_PROMPT)


_load_presets()


# ---- 显存查询（NVML，不依赖 torch）----
_nvml = None
_nvml_tried = False


def _nvml_mem():
    """返回 (used_gb, free_gb) 或 None。"""
    global _nvml, _nvml_tried
    try:
        if not _nvml_tried:
            _nvml_tried = True
            import ctypes
            import ctypes.util
            path = ctypes.util.find_library("nvml")
            for cand in (path, r"C:\Windows\System32\nvml.dll",
                         r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"):
                if cand and os.path.isfile(cand):
                    _nvml = ctypes.WinDLL(cand)
                    _nvml.nvmlInit_v2()
                    break
        if not _nvml:
            return None

        class _Mem(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong),
                        ("used", ctypes.c_ulonglong)]

        h = ctypes.c_void_p()
        if _nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return None
        m = _Mem()
        if _nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) != 0:
            return None
        return m.used / 1024**3, m.free / 1024**3
    except Exception:
        _nvml = None
        return None


def load_model(key):
    global _model
    try:
        m = scan_models().get(key)
        if not m:
            raise ValueError(f"未知模型: {key}")
        model_path = os.path.join(MODEL_DIR, m["model"])
        mmproj_path = os.path.join(MODEL_DIR, m["mmproj"])
        for p in (model_path, mmproj_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"缺少模型文件: {p}")
        t0 = time.time()
        print(f"[model] device={DEVICE} loading {m['model']} ...")

        from llama_cpp import Llama

        _model = Llama(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_gpu_layers=-1,      # 全部层 offload 到 GPU
            n_ctx=N_CTX,
            flash_attn=(DEVICE == "cuda"),
            verbose=False,
        )
        _state["model"] = key
        _state["status"] = "ready"
        print(f"[model] 就绪，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        _state["status"] = "error"
        _state["model"] = None
        _state["error"] = f"{e.__class__.__name__}: {e}"
        traceback.print_exc()


def unload_model():
    """卸载模型，释放显存，方便切去 ComfyUI 等其他程序。"""
    global _model
    with _model_op_lock:
        if _state["status"] != "ready":
            return False, f"当前状态为 {_state['status']}，无需卸载"
        with _infer_lock:  # 等当前推理跑完再卸
            _model = None
            gc.collect()
        _state["status"] = "unloaded"
        _state["model"] = None
        print("[model] 已卸载，显存已释放")
        return True, "已卸载，显存已释放"


def start_load(key):
    """后台线程重新加载模型。"""
    if _state["status"] in ("loading", "ready"):
        return False, f"当前状态为 {_state['status']}，无需加载"
    with _model_op_lock:
        if _state["status"] in ("loading", "ready"):
            return False, f"当前状态为 {_state['status']}，无需加载"
        _state["status"] = "loading"
        _state["error"] = None
    threading.Thread(target=load_model, args=(key,), daemon=True).start()
    return True, "开始加载模型"


# ---------------------------------------------------------------- 推理 ---

def analyze_image(pil_img: Image.Image, lang: str = "zh", max_side: int = DEFAULT_RES,
                  max_tokens: int = MAX_NEW_TOKENS) -> str:
    pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > max_side:
        pil_img.thumbnail((max_side, max_side))  # 长边缩放，控制视觉 token 数量
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    messages = [
        {"role": "system", "content": _current_sys_prompt()},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": INSTRUCTIONS.get(lang, INSTRUCTION_ZH)},
        ]},
    ]
    t0 = time.time()
    resp = _model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,   # 低温采样：接近确定结果，又避免贪心解码死循环跑满 token
        top_p=0.8,
        top_k=20,
    )
    sec = time.time() - t0
    usage = resp.get("usage") or {}
    n_tok = usage.get("completion_tokens") or 0
    stats = {"sec": round(sec, 1), "tokens": n_tok,
             "tps": round(n_tok / sec, 1) if sec > 0 else 0.0}
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    # 保险：去掉可能出现的思考块和引号
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text.strip('"').strip(), stats


# ---------------------------------------------------------------- Flask ---

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/ping")
def ping():
    return {"ok": True, "app": "image-to-prompt"}

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>自然语言提示词反推</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body.light {
    --bg:#f0f2f5; --card:#ffffff; --inner:#f7f8fa; --border:#e4e6eb;
    --text:#1c1e21; --muted:#65676b; --name:#3a3d42;
    --btn:#ffffff; --btn-border:#d0d3d8; --btn-hover:#f0f2f5; --btn-text:#1c1e21;
    color-scheme: light;
  }
  body.dark {
    --bg:#0f1115; --card:#161a22; --inner:#0f1115; --border:#262b36;
    --text:#e6e8ee; --muted:#8b93a3; --name:#aeb6c6;
    --btn:#2b3040; --btn-border:#3a3f4d; --btn-hover:#353b4e; --btn-text:#e6e8ee;
    color-scheme: dark;
  }
  body { margin:0; padding:24px; font-family:"Segoe UI",system-ui,sans-serif;
         background:var(--bg); color:var(--text); }
  .wrap { max-width: 860px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
  #badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
           background:var(--btn); color:var(--muted); margin-left:8px; }
  #badge.ready { background:#1f5c33; color:#7ee2a0; }
  #badge.loading { background:#5c4a1f; color:#f0d080; }
  #badge.error { background:#5c1f1f; color:#f08080; }
  #badge.unloaded { background:var(--btn); color:var(--muted); }
  #drop { border:2px dashed var(--btn-border); border-radius:12px; padding:36px; text-align:center;
          color:var(--muted); cursor:pointer; transition:.15s; }
  #drop.over { border-color:#6ea8fe; background:var(--card); }
  #drop b { color:var(--text); }
  button { background:var(--btn); color:var(--btn-text); border:1px solid var(--btn-border);
           padding:8px 16px; border-radius:8px; cursor:pointer; font-size:14px; white-space:nowrap; }
  button:hover { background:var(--btn-hover); }
  button.primary { background:#2f6feb; border-color:#2f6feb; color:#fff; }
  button.primary:disabled { opacity:.4; cursor:not-allowed; }
  .btns { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; align-items:center; }
  .btns > span { white-space:nowrap; }
  .sel { background:var(--btn); color:var(--btn-text); border:1px solid var(--btn-border);
         border-radius:6px; padding:4px 8px; font-size:13px; }
  #stats { margin:14px 0 8px; font-size:13px; color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:12px; margin-top:12px; display:flex; gap:12px; }
  .card img { width:120px; height:120px; object-fit:cover; border-radius:8px; flex:none; }
  .card .body { flex:1; min-width:0; display:flex; flex-direction:column; }
  .card .name { font-size:13px; color:var(--name); margin-bottom:6px;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .card .status { font-size:12px; color:var(--muted); }
  .card .status.err { color:#f08080; }
  .card pre { margin:6px 0 8px; padding:10px; background:var(--inner); border-radius:8px;
              font-size:13px; line-height:1.5; white-space:pre-wrap; word-break:break-word;
              flex:1; font-family:inherit; }
  .card .ops { display:flex; gap:8px; }
  .card .ops button { padding:4px 12px; font-size:12px; }
  .syscard { background:var(--card); border:1px solid var(--border); border-radius:12px;
             padding:12px; margin-top:16px; }
  .syshead { display:flex; justify-content:space-between; align-items:center; gap:10px;
             font-size:13px; color:var(--muted); margin-bottom:8px; flex-wrap:wrap; }
  .syshead button { padding:4px 12px; font-size:12px; }
  #sysText { width:100%; box-sizing:border-box; min-height:150px; resize:vertical;
             background:var(--inner); color:var(--text); border:1px solid var(--btn-border);
             border-radius:8px; padding:10px; font-size:13px; line-height:1.5;
             font-family:inherit; }
  .pmask { position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:1000;
           display:flex; align-items:center; justify-content:center; }
  .pmask[hidden] { display:none; }   /* display:flex 会覆盖 hidden 属性，需显式还原 */
  .pmodal { background:var(--card); border:1px solid var(--border); border-radius:12px;
            width:min(560px, 92vw); max-height:70vh; display:flex; flex-direction:column;
            padding:14px; }
  .phead { display:flex; justify-content:space-between; align-items:center;
           font-size:14px; font-weight:600; margin-bottom:10px; }
  #presetList { overflow-y:auto; display:flex; flex-direction:column; gap:6px; }
  .pitem { display:flex; align-items:center; gap:8px; padding:8px 10px; cursor:pointer;
           border:1px solid var(--btn-border); border-radius:8px; font-size:13px; }
  .pitem:hover { background:var(--inner); }
  .pitem.cur { border-color:#4a9eff; }
  .pitem .pname { font-weight:600; white-space:nowrap; }
  .pitem .pprev { color:var(--muted); font-size:12px; overflow:hidden;
                  text-overflow:ellipsis; white-space:nowrap; flex:1; }
  .pitem .pdel { padding:1px 8px; font-size:12px; flex:none; }
  .pnew { display:flex; gap:8px; margin-top:10px; }
  .pnew input { flex:1; background:var(--inner); color:var(--text);
                border:1px solid var(--btn-border); border-radius:8px; padding:7px 10px;
                font-size:13px; font-family:inherit; }
  .theme-toggle { position:fixed; bottom:20px; right:20px; font-size:24px; cursor:pointer;
                  background:transparent; border:none; padding:6px 10px; border-radius:6px;
                  transition:background .2s; z-index:999; }
  body.dark .theme-toggle:hover { background:#38383a; }
  body.light .theme-toggle:hover { background:#e9e9eb; }
</style>
</head>
<body>
<div class="wrap">
  <h1>自然语言提示词反推 <span id="badge" class="unloaded">模型未加载</span></h1>
  <div class="sub">GGUF 视觉模型（llama.cpp）本地推理 · 反推文生图提示词 · 可切换模型与系统提示词</div>


  <div class="btns" style="margin-bottom:14px;">
    <button id="btnLoad">加载模型</button>
    <button id="btnUnload">卸载模型（释放显存）</button>
    <span style="align-self:center;font-size:13px;color:var(--muted);">模型：
      <select id="modelSel" class="sel"></select>
    </span>
  </div>

  <div id="drop">将图片或文件夹拖到这里，或点击选择</div>
  <div class="btns">
    <button class="primary" id="pickFiles">选择图片（可多选）</button>
    <button id="pickDir">选择文件夹</button>
    <button id="clear">清空列表</button>
    <span style="align-self:center;font-size:13px;color:var(--muted);">提示词语言：
      <select id="langSel" class="sel">
        <option value="zh" selected>中文</option>
        <option value="en">English</option>
      </select>
    </span>
    <span style="align-self:center;font-size:13px;color:var(--muted);">图像分辨率：
      <select id="resSel" class="sel">
        <option value="768">768（快速）</option>
        <option value="1024">1024（均衡）</option>
        <option value="1536" selected>1536（高清）</option>
      </select>
    </span>
    <span style="align-self:center;font-size:13px;color:var(--muted);">最大输出长度：
      <select id="maxtSel" class="sel">
        <option value="512">512</option>
        <option value="768">768</option>
        <option value="1024" selected>1024</option>
        <option value="1536">1536</option>
        <option value="2048">2048</option>
      </select>
    </span>
  </div>
  <input type="file" id="fileInput" multiple accept="image/*" hidden>
  <input type="file" id="dirInput" webkitdirectory multiple hidden>
  <div id="stats"></div>
  <div id="list"></div>
  <div class="btns">
    <button id="download">下载全部结果 (.txt)</button>
  </div>

  <div class="syscard">
    <div class="syshead">
      <span>系统提示词（模型人设，每次反推都会生效，切换模型共用）</span>
      <span>
        <button id="presetBtn" title="点击展开提示词方案列表">方案 ▾</button>
        <button id="sysSave">保存</button>
        <button id="sysReset">恢复默认</button>
      </span>
    </div>
    <textarea id="sysText" spellcheck="false"></textarea>
  </div>
</div>

<div id="presetMask" class="pmask" hidden>
  <div class="pmodal">
    <div class="phead">
      <span>提示词方案</span>
      <button id="presetClose">✕</button>
    </div>
    <div id="presetList"></div>
    <div class="pnew">
      <input id="presetName" placeholder="输入新方案名称">
      <button id="presetAdd" title="以当前编辑框内容创建新方案">新增（取当前编辑内容）</button>
    </div>
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

const EXT = /\.(jpe?g|png|webp|bmp|gif|tiff?)$/i;
let items = [];      // {id, file, url, status, prompt}
let running = false;
let doneCount = 0;
const $ = id => document.getElementById(id);
const listEl = $('list'), statsEl = $('stats'), dropEl = $('drop');

// ---- 模型状态轮询 ----
async function poll() {
  try {
    const r = await fetch('/api/status'); const s = await r.json();
    const b = $('badge');
    if (s.status === 'ready') {
      b.textContent = '模型就绪' + (s.vram != null ? ' · 显存 ' + s.vram + 'G · 空闲 ' + s.free + 'G' : '')
                    + (s.model && $('modelSel').selectedOptions[0] ? '（' + $('modelSel').selectedOptions[0].text.split('（')[0] + '）' : '');
      b.className = 'ready'; b.title = '';
    } else if (s.status === 'error') {
      b.textContent = '加载失败（可重试）'; b.className = 'error'; b.title = s.error;
    } else if (s.status === 'unloaded') {
      b.textContent = '模型未加载 · 显存已释放'; b.className = 'unloaded'; b.title = '点击上方"加载模型"按钮后再使用';
    } else {
      b.textContent = '模型加载中…'; b.className = 'loading';
    }
    if (s.model) $('modelSel').value = s.model;
    $('modelSel').disabled = (s.status !== 'unloaded');  // 需先卸载才能切换模型
    $('btnLoad').disabled = (s.status === 'loading' || s.status === 'ready');
    $('btnUnload').disabled = (s.status !== 'ready');
  } catch (e) {}
}
setInterval(poll, 1500); poll();

// ---- 模型下拉初始化 ----
(async function initModels() {
  try {
    const r = await fetch('/api/models'); const j = await r.json();
    const sel = $('modelSel'); sel.innerHTML = '';
    for (const [k, m] of Object.entries(j.models)) {
      const o = document.createElement('option'); o.value = k; o.textContent = m.label;
      sel.appendChild(o);
    }
    if (j.current) sel.value = j.current; else sel.value = j.default;
    sel.disabled = (j.status !== 'unloaded');
  } catch (e) {}
})();

// ---- 系统提示词多方案 ----
async function loadPresets() {
  const r = await fetch('/api/prompts'); const j = await r.json();
  if (!j.ok) return;
  const cur = (j.presets || []).find(p => p.current);
  if (cur) $('sysText').value = cur.prompt;
  $('presetBtn').textContent = '方案：' + (cur ? cur.name : '—') + ' ▾';
}
(async function initSys() { try { await loadPresets(); } catch (e) {} })();

async function presetAction(body) {
  const r = await fetch('/api/prompts', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return await r.json();
}
$('sysSave').onclick = async () => {
  const btn = $('sysSave');
  try {
    const j = await presetAction({ action: 'save', prompt: $('sysText').value });
    btn.textContent = j.ok ? '已保存' : (j.error || '保存失败');
  } catch (e) { btn.textContent = '保存失败'; }
  setTimeout(() => btn.textContent = '保存', 1200);
};
$('sysReset').onclick = async () => {
  const btn = $('sysReset');
  try {
    const j = await presetAction({ action: 'reset' });
    if (j.ok) { $('sysText').value = j.prompt; await loadPresets(); }
    btn.textContent = j.ok ? '已恢复' : (j.error || '失败');
  } catch (e) { btn.textContent = '失败'; }
  setTimeout(() => btn.textContent = '恢复默认', 1200);
};

// ---- 方案弹窗 ----
function renderPresetList() {
  const wrap = $('presetList'); wrap.innerHTML = '';
  for (const p of window.__presets || []) {
    const row = document.createElement('div');
    row.className = 'pitem' + (p.current ? ' cur' : '');
    const nm = document.createElement('span'); nm.className = 'pname';
    nm.textContent = (p.current ? '✔ ' : '') + p.name;
    const pv = document.createElement('span'); pv.className = 'pprev';
    pv.textContent = p.prompt.replace(/\s+/g, ' ').slice(0, 60);
    row.append(nm, pv);
    row.onclick = async () => {
      const j = await presetAction({ action: 'select', name: p.name });
      if (j.ok) {
        $('sysText').value = j.prompt;
        await loadPresets(); renderPresetList();
      }
    };
    if (!p.current) {
      const del = document.createElement('button'); del.className = 'pdel';
      del.textContent = '删除';
      del.onclick = async e => {
        e.stopPropagation();
        if (!confirm('删除方案「' + p.name + '」？')) return;
        const j = await presetAction({ action: 'delete', name: p.name });
        if (j.ok) { await loadPresets(); renderPresetList(); }
        else alert(j.error || '删除失败');
      };
      row.appendChild(del);
    }
    wrap.appendChild(row);
  }
}
$('presetBtn').onclick = async () => {
  try {
    const r = await fetch('/api/prompts'); const j = await r.json();
    window.__presets = j.presets || [];
    renderPresetList();
    $('presetMask').hidden = false;
  } catch (e) {}
};
$('presetClose').onclick = () => { $('presetMask').hidden = true; };
$('presetMask').onclick = e => { if (e.target === $('presetMask')) $('presetMask').hidden = true; };
$('presetAdd').onclick = async () => {
  const name = $('presetName').value.trim();
  if (!name) { alert('请先输入新方案名称'); return; }
  const j = await presetAction({ action: 'new', name: name, prompt: $('sysText').value });
  if (j.ok) {
    $('presetName').value = '';
    $('sysText').value = j.prompt;
    await loadPresets(); renderPresetList();
  } else alert(j.error || '新增失败');
};

// ---- 加载 / 卸载模型 ----
$('btnLoad').onclick = async () => {
  $('btnLoad').disabled = true;
  await fetch('/api/load', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: $('modelSel').value }) });
  poll();
};
$('btnUnload').onclick = async () => {
  $('btnUnload').disabled = true;
  $('btnUnload').textContent = '卸载中…';
  try { await fetch('/api/unload', { method: 'POST' }); } catch (e) {}
  $('btnUnload').textContent = '卸载模型（释放显存）';
  poll();
};

// ---- 选择文件 ----
$('pickFiles').onclick = () => $('fileInput').click();
$('pickDir').onclick = () => $('dirInput').click();
$('clear').onclick = () => { items = []; doneCount = 0; render(); updateStats(); };
$('fileInput').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
// 文件夹模式：只取所选文件夹第一层的文件，不递归子目录（webkitRelativePath 形如 "文件夹/文件名"）
$('dirInput').onchange = e => {
  addFiles([...e.target.files].filter(f => f.webkitRelativePath.split('/').length === 2));
  e.target.value = '';
};

dropEl.onclick = () => $('fileInput').click();
dropEl.ondragover = e => { e.preventDefault(); dropEl.classList.add('over'); };
dropEl.ondragleave = () => dropEl.classList.remove('over');
dropEl.ondrop = async e => {
  e.preventDefault(); dropEl.classList.remove('over');
  const files = [];
  const entries = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry());
  for (const en of entries) { if (en) await walkEntry(en, files); }
  if (!files.length && e.dataTransfer.files.length) files.push(...e.dataTransfer.files);
  addFiles(files);
};
// 拖拽文件夹：同样只收该文件夹第一层的文件，不进入子目录
async function walkEntry(entry, out) {
  if (entry.isFile) {
    const f = await new Promise(res => entry.file(res));
    out.push(f);
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    let batch;
    do {
      batch = await new Promise(res => reader.readEntries(res));
      for (const c of batch) {
        if (c.isFile) out.push(await new Promise(res => c.file(res)));
      }
    } while (batch.length);
  }
}

function addFiles(files) {
  for (const f of files) {
    if (!EXT.test(f.name)) continue;
    items.push({ id: crypto.randomUUID(), file: f, url: URL.createObjectURL(f),
                 status: 'pending', prompt: '' });
  }
  render(); updateStats(); pump();
}

// ---- 渲染 ----
function render() {
  listEl.innerHTML = '';
  for (const it of items) {
    const card = document.createElement('div'); card.className = 'card';
    const img = document.createElement('img'); img.src = it.url; card.appendChild(img);
    const body = document.createElement('div'); body.className = 'body';
    const name = document.createElement('div'); name.className = 'name'; name.textContent = it.file.name;
    const pre = document.createElement('pre');
    const st = document.createElement('div'); st.className = 'status';
    if (it.status === 'pending') st.textContent = '排队中…';
    else if (it.status === 'running') st.textContent = '反推中…';
    else if (it.status === 'done') {
      const s2 = it.stats;
      st.textContent = '完成' + (s2 ? ' · ' + s2.sec + 's · ' + s2.tokens + ' tok · ' + s2.tps + ' tok/s' : '');
    }
    else { st.textContent = '失败: ' + it.error; st.className = 'status err'; }
    pre.textContent = it.prompt;
    body.append(name, st, pre);
    if (it.status === 'done') {
      const ops = document.createElement('div'); ops.className = 'ops';
      const cp = document.createElement('button'); cp.textContent = '复制提示词';
      cp.onclick = () => { navigator.clipboard.writeText(it.prompt); cp.textContent = '已复制';
                           setTimeout(() => cp.textContent = '复制提示词', 1200); };
      ops.appendChild(cp); body.appendChild(ops);
    }
    card.appendChild(body); listEl.appendChild(card);
  }
}
function updateStats() {
  const done = items.filter(i => i.status === 'done').length;
  const fail = items.filter(i => i.status === 'error').length;
  statsEl.textContent = items.length ? `共 ${items.length} 张 · 完成 ${done} · 失败 ${fail}` : '';
}
function updateById(id, patch) {
  const it = items.find(i => i.id === id); if (!it) return;
  Object.assign(it, patch); render(); updateStats();
}

// ---- 串行处理（模型一次处理一张） ----
function pump() {
  if (running) return;
  const next = items.find(i => i.status === 'pending');
  if (!next) return;
  running = true;
  next.status = 'running'; render();
  const fd = new FormData(); fd.append('image', next.file, next.file.name);
  fd.append('lang', $('langSel').value);
  fd.append('res', $('resSel').value);
  fd.append('maxt', $('maxtSel').value);
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 300000);  // 单张 5 分钟超时，防止假死卡住队列
  fetch('/api/infer', { method: 'POST', body: fd, signal: ctl.signal })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      updateById(next.id, ok && j.ok
        ? { status: 'done', prompt: j.prompt, stats: j.stats || null }
        : { status: 'error', error: (j && j.error) || ('HTTP ' + ok) });
      doneCount++;
    })
    .catch(e => updateById(next.id, { status: 'error',
                 error: e.name === 'AbortError' ? '请求超时(5分钟)' : String(e) }))
    .finally(() => { clearTimeout(timer); running = false; pump(); });
}

// ---- 下载全部 ----
$('download').onclick = () => {
  const lines = items.filter(i => i.status === 'done')
                     .map(i => i.file.name + '\n' + i.prompt + '\n');
  if (!lines.length) return alert('还没有完成的结果');
  const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'krea_prompts_' + new Date().toISOString().slice(0, 10) + '.txt';
  a.click();
};
</script>
</body>
</html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.get("/api/status")
def status():
    s = dict(_state)
    mem = _nvml_mem()
    if mem:
        s["vram"] = round(mem[0], 1)   # 全机已用显存
        s["free"] = round(mem[1], 1)   # 全机空闲显存
    return jsonify(s)


@app.get("/api/models")
def api_models():
    models = scan_models()   # 每次请求重新扫描，放入新模型文件后刷新页面即可看到
    default = next(iter(models), None)
    return jsonify({"ok": True, "models": models, "current": _state.get("model"),
                    "default": default, "status": _state["status"]})


@app.post("/api/load")
def api_load():
    data = request.get_json(silent=True) or {}
    key = data.get("model") or next(iter(scan_models()), None)
    if not key or key not in scan_models():
        return jsonify({"ok": False, "message": f"未知模型: {key}"}), 400
    if _state["status"] == "ready" and _state.get("model") != key:
        unload_model()  # 切换模型：先释放当前模型再加载新的
    ok, msg = start_load(key)
    return jsonify({"ok": ok, "message": msg})


@app.post("/api/unload")
def api_unload():
    ok, msg = unload_model()
    return jsonify({"ok": ok, "message": msg})


@app.get("/api/prompts")
def api_prompts():
    """列出全部提示词方案（含当前选中项）"""
    return jsonify({"ok": True, "current": _prompt_current,
                    "presets": [{"name": n, "prompt": p, "current": n == _prompt_current}
                                for n, p in _prompt_presets.items()]})


@app.post("/api/prompts")
def api_prompts_set():
    """提示词方案操作：select 切换 / save 保存当前内容 / new 新增 / delete 删除 / reset 恢复出厂"""
    global _prompt_current, _prompt_presets
    d = request.get_json(silent=True) or {}
    act = d.get("action")
    if act == "select":
        name = (d.get("name") or "").strip()
        if name not in _prompt_presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        _prompt_current = name
    elif act == "save":
        p = (d.get("prompt") or "").strip()
        if not p:
            return jsonify({"ok": False, "error": "提示词不能为空"}), 400
        _prompt_presets[_prompt_current] = p
    elif act == "new":
        name = (d.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "方案名不能为空"}), 400
        if name in _prompt_presets:
            return jsonify({"ok": False, "error": "方案名已存在"}), 400
        _prompt_presets[name] = (d.get("prompt") or "").strip() or DEFAULT_SYS_PROMPT
        _prompt_current = name
    elif act == "delete":
        name = (d.get("name") or "").strip()
        if name not in _prompt_presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        if name == _prompt_current:
            return jsonify({"ok": False, "error": "不能删除当前选中的方案"}), 400
        del _prompt_presets[name]
    elif act == "reset":
        _prompt_presets[_prompt_current] = BUILTIN_PRESETS.get(_prompt_current, DEFAULT_SYS_PROMPT)
    else:
        return jsonify({"ok": False, "error": "未知操作"}), 400
    _save_presets()
    return jsonify({"ok": True, "current": _prompt_current,
                    "prompt": _prompt_presets[_prompt_current]})


@app.post("/api/infer")
def infer():
    if _state["status"] != "ready":
        if _state["status"] == "unloaded":
            err = "模型已卸载，请先点击「加载模型」"
        elif _state["status"] == "loading":
            err = "模型加载中，请稍候"
        else:
            err = f"模型不可用（{_state['error'] or _state['status']}）"
        return jsonify({"ok": False, "error": err}), 503
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "缺少图片文件"}), 400
    try:
        pil = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"ok": False, "error": f"无法读取图片: {e}"}), 400

    with _infer_lock:  # GPU 串行
        try:
            try:
                res = int(request.form.get("res", DEFAULT_RES))
            except ValueError:
                res = DEFAULT_RES
            res = max(512, min(res, 4096))
            try:
                maxt = int(request.form.get("maxt", MAX_NEW_TOKENS))
            except ValueError:
                maxt = MAX_NEW_TOKENS
            maxt = max(128, min(maxt, 4096))
            prompt, stats = analyze_image(pil, request.form.get("lang", "zh"), res, maxt)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}"}), 500
    return jsonify({"ok": True, "prompt": prompt, "stats": stats})


# ---------------------------------------------------------------- 入口 ---

if __name__ == "__main__":
    print(f"[app] model dir: {MODEL_DIR}")
    for k, m in scan_models().items():
        print(f"[app]   - {k}: {m['model']} + {m['mmproj']}")
    # 模型不再启动时自动加载，等用户在页面上点"加载模型"再载入（省显存）
    print(f"[app] model will NOT auto-load; click the load button on the page when needed")

    def _open_browser():
        if os.environ.get("TOOL_NO_BROWSER") == "1":  # 被门户拉起时不自动开浏览器
            return
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"[app] serving at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
