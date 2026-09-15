# -*- coding: utf-8 -*-
"""
Krea2 提示词反推工具（单文件版）
- 上传单张图片或整个文件夹，用本地 Qwen3-VL-4B-Instruct-FP8 模型反推文生图提示词（支持中文/英文，适配 FLUX.1 Krea）
- 不依赖 vllm / llama.cpp，纯 transformers + torch
- 首次启动把 FP8 权重反量化为 BF16 并缓存到模型目录下 _bf16_cache/，之后启动直接加载缓存（更快、内存占用更低）
- 运行方式:
    conda activate pytorch-study
    python image_to_prompt.py
- 浏览器自动打开 http://127.0.0.1:7863
"""

import gc
import io
import json
import os
import shutil
import threading
import time
import traceback
import webbrowser

import torch
from PIL import Image
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # 统一启动器根目录（脚本在 tools/img2prompt/ 下，往上两级）
# 模型统一放在根目录的 model 文件夹
MODEL_DIR = os.path.join(ROOT_DIR, "model", "Qwen3-VL-4B-Instruct-FP8")
CACHE_DIR = os.path.join(MODEL_DIR, "_bf16_cache")
PORT = 7867
MAX_NEW_TOKENS = 512
DEFAULT_RES = 1536          # 默认图像长边上限：768 快 / 1024 均衡 / 1536 高清
PROC_MAX_PIXELS = 1536 * 1536  # 处理器兜底像素上限（总像素）

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

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
_processor = None
_state = {"status": "unloaded", "device": DEVICE, "error": None}  # 模型默认不自动加载
_infer_lock = threading.Lock()
_model_op_lock = threading.Lock()  # 串行化 加载/卸载 操作


def _dequant_block_fp8(w_fp8, scale):
    """FP8 块缩放(128x128) -> BF16。按行块分块处理，避免 fp32 全尺寸中间张量撑爆内存。"""
    n_ob, n_ib = scale.shape
    in_d = w_fp8.shape[1]
    rows = []
    for i in range(n_ob):
        blk = w_fp8[i * 128:(i + 1) * 128].to(torch.float32)          # [128, in_d]
        blk *= scale[i].repeat_interleave(in_d // n_ib)[None, :]
        rows.append(blk.to(torch.bfloat16))
    return torch.cat(rows, dim=0)


def _build_bf16_cache(model_dir, cache_dir):
    """把 FP8 checkpoint 反量化为 BF16 并保存到磁盘缓存，返回缓存目录。"""
    import safetensors.torch

    marker = os.path.join(cache_dir, ".complete")
    if os.path.isfile(marker):
        return cache_dir
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)  # 上次可能中途失败，清掉重建
    os.makedirs(cache_dir)

    t0 = time.time()
    print("[model] 首次运行：正在把 FP8 权重反量化为 BF16 并缓存（约 1-3 分钟，之后启动会快很多）...")
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))

    for shard in shards:
        sd = safetensors.torch.load_file(os.path.join(model_dir, shard))
        out = {}
        for name, tensor in sd.items():
            if name.endswith(".weight_scale_inv") or name.endswith(".weight_scale"):
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                scale_key = name[: -len(".weight")] + ".weight_scale_inv"
                if scale_key not in sd:
                    raise RuntimeError(f"找不到 {scale_key} 的缩放系数")
                out[name] = _dequant_block_fp8(tensor, sd[scale_key].to(torch.float32))
            elif tensor.is_floating_point():
                out[name] = tensor.to(torch.bfloat16)
            else:
                out[name] = tensor
        safetensors.torch.save_file(out, os.path.join(cache_dir, shard), metadata={"format": "pt"})
        del sd, out

    # 写入去掉量化配置的 config.json，并复制其余小文件，让缓存目录自成完整模型
    index["weight_map"] = {k: v for k, v in index["weight_map"].items()
                           if not k.endswith(".weight_scale_inv")
                           and not k.endswith(".weight_scale")}
    with open(os.path.join(cache_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    with open(os.path.join(cache_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    for fname in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "chat_template.json", "preprocessor_config.json",
                  "video_preprocessor_config.json"):
        src = os.path.join(model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(cache_dir, fname))
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")
    print(f"[model] BF16 缓存完成，耗时 {time.time() - t0:.0f}s -> {cache_dir}")
    return cache_dir


def load_model():
    global _model, _processor
    try:
        t0 = time.time()
        print(f"[model] device={DEVICE} ...")

        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        cache_dir = _build_bf16_cache(MODEL_DIR, CACHE_DIR)

        # device_map 流式加载：逐张量上卡，内存峰值低
        _model = ModelCls.from_pretrained(
            cache_dir,
            dtype=DTYPE,
            device_map={"": 0 if DEVICE == "cuda" else "cpu"},
            attn_implementation="sdpa",
        )
        _model.eval()

        # 关键：Qwen3-VL 处理器以 size.longest_edge 为像素上限（默认 1677 万！，
        # 传 max_pixels 不生效）。必须用 size 覆盖，否则大图会产生近万视觉 token 直接 OOM
        _processor = AutoProcessor.from_pretrained(
            MODEL_DIR,
            size={"shortest_edge": 32 * 32, "longest_edge": PROC_MAX_PIXELS},
        )

        _state["status"] = "ready"
        vram = (f", VRAM={torch.cuda.memory_allocated() / 1024**3:.1f}GB"
                if DEVICE == "cuda" else "")
        print(f"[model] 就绪，耗时 {time.time() - t0:.1f}s{vram}")
    except Exception as e:
        _state["status"] = "error"
        _state["error"] = f"{e.__class__.__name__}: {e}"
        traceback.print_exc()


def unload_model():
    """卸载模型，释放显存，方便切去 ComfyUI 等其他程序。"""
    global _model, _processor
    with _model_op_lock:
        if _state["status"] != "ready":
            return False, f"当前状态为 {_state['status']}，无需卸载"
        with _infer_lock:  # 等当前推理跑完再卸
            _model = None
            _processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _state["status"] = "unloaded"
        print("[model] 已卸载，显存已释放")
        return True, "已卸载，显存已释放"


def start_load():
    """后台线程重新加载模型。"""
    if _state["status"] in ("loading", "ready"):
        return False, f"当前状态为 {_state['status']}，无需加载"
    with _model_op_lock:
        if _state["status"] in ("loading", "ready"):
            return False, f"当前状态为 {_state['status']}，无需加载"
        _state["status"] = "loading"
        _state["error"] = None
    threading.Thread(target=load_model, daemon=True).start()
    return True, "开始加载模型"


# ---------------------------------------------------------------- 推理 ---

def analyze_image(pil_img: Image.Image, lang: str = "zh", max_side: int = DEFAULT_RES) -> str:
    pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > max_side:
        pil_img.thumbnail((max_side, max_side))  # 长边缩放，控制视觉 token 数量
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": INSTRUCTIONS.get(lang, INSTRUCTION_ZH)},
        ],
    }]
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,   # 低温采样：接近确定结果，又避免贪心解码死循环跑满 token
            top_p=0.8,
            top_k=20,
        )
    text = _processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    # 保险：去掉可能出现的思考块和引号
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text.strip('"').strip()


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
<title>Krea2提示词反推</title>
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
  .btns { display:flex; gap:10px; margin-top:14px; }
  button { background:var(--btn); color:var(--btn-text); border:1px solid var(--btn-border);
           padding:8px 16px; border-radius:8px; cursor:pointer; font-size:14px; }
  button:hover { background:var(--btn-hover); }
  button.primary { background:#2f6feb; border-color:#2f6feb; color:#fff; }
  button.primary:disabled { opacity:.4; cursor:not-allowed; }
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
  .theme-toggle { position:fixed; bottom:20px; right:20px; font-size:24px; cursor:pointer;
                  background:transparent; border:none; padding:6px 10px; border-radius:6px;
                  transition:background .2s; z-index:999; }
  body.dark .theme-toggle:hover { background:#38383a; }
  body.light .theme-toggle:hover { background:#e9e9eb; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Krea2提示词反推 <span id="badge" class="unloaded">模型未加载</span></h1>
  <div class="sub">Qwen3-VL-4B-Instruct-FP8 本地推理 · 反推文生图提示词（Krea-2）</div>


  <div class="btns" style="margin-bottom:14px;">
    <button id="btnLoad">加载模型</button>
    <button id="btnUnload">卸载模型（释放显存）</button>
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
  </div>
  <input type="file" id="fileInput" multiple accept="image/*" hidden>
  <input type="file" id="dirInput" webkitdirectory multiple hidden>
  <div id="stats"></div>
  <div id="list"></div>
  <div class="btns">
    <button id="download">下载全部结果 (.txt)</button>
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
      b.textContent = '模型就绪 · 显存 ' + (s.vram != null ? s.vram + 'G' : '?')
                    + (s.free_vram != null ? ' · 空闲 ' + s.free_vram + 'G' : '');
      b.className = 'ready'; b.title = '';
    } else if (s.status === 'error') {
      b.textContent = '加载失败（可重试）'; b.className = 'error'; b.title = s.error;
    } else if (s.status === 'unloaded') {
      b.textContent = '模型未加载 · 显存已释放'; b.className = 'unloaded'; b.title = '点击上方"加载模型"按钮后再使用';
    } else {
      b.textContent = '模型加载中…'; b.className = 'loading';
    }
    $('btnLoad').disabled = (s.status === 'loading' || s.status === 'ready');
    $('btnUnload').disabled = (s.status !== 'ready');
  } catch (e) {}
}
setInterval(poll, 1500); poll();

// ---- 加载 / 卸载模型 ----
$('btnLoad').onclick = async () => {
  $('btnLoad').disabled = true;
  await fetch('/api/load', { method: 'POST' });
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
    else if (it.status === 'done') st.textContent = '完成';
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
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 300000);  // 单张 5 分钟超时，防止假死卡住队列
  fetch('/api/infer', { method: 'POST', body: fd, signal: ctl.signal })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      updateById(next.id, ok && j.ok
        ? { status: 'done', prompt: j.prompt }
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
    if DEVICE == "cuda":
        s["torch_allocated"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        s["torch_reserved"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
        free_b, _total = torch.cuda.mem_get_info()
        s["free_vram"] = round(free_b / 1024**3, 1)
        if _state["status"] == "ready":
            s["vram"] = s["torch_allocated"]
    return jsonify(s)


@app.post("/api/load")
def api_load():
    ok, msg = start_load()
    return jsonify({"ok": ok, "message": msg})


@app.post("/api/unload")
def api_unload():
    ok, msg = unload_model()
    return jsonify({"ok": ok, "message": msg})


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
            if DEVICE == "cuda":
                free_b, _ = torch.cuda.mem_get_info()
                if free_b < 2.5 * 1024**3:
                    return jsonify({"ok": False,
                                    "error": f"显存空闲仅 {free_b / 1024**3:.1f}G，不够安全推理，"
                                             "硬跑会溢出到内存导致假死；请先释放显存"
                                             "（如在 ComfyUI 里卸载模型）再试"}), 503
            try:
                res = int(request.form.get("res", DEFAULT_RES))
            except ValueError:
                res = DEFAULT_RES
            res = max(512, min(res, 4096))
            prompt = analyze_image(pil, request.form.get("lang", "zh"), res)
        except torch.cuda.OutOfMemoryError:
            return jsonify({"ok": False,
                            "error": "显存不足(OOM)，请关闭其他占用显存的程序（如 ComfyUI）后重试"}), 500
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}"}), 500
        finally:
            # 每次推理后清掉缓存块：不同图片缩放尺寸不同会碎片化缓存，
            # 累积后会导致下一张分配失败（表现为卡死、显存暴涨）
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    return jsonify({"ok": True, "prompt": prompt})


# ---------------------------------------------------------------- 入口 ---

if __name__ == "__main__":
    print(f"[app] model dir: {MODEL_DIR}")
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
