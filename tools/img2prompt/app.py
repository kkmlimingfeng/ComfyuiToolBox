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
import math
import os
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser

from PIL import Image
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # 统一启动器根目录（脚本在 tools/img2prompt/ 下，往上两级）
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SYS_FILE = os.path.join(BASE_DIR, "system_prompt.json")   # 系统提示词持久化
PORT = 7867
MAX_NEW_TOKENS = 512
N_CTX = 16384                # 默认上下文长度，兼顾 27B 模型显存占用
VIDEO_CTX_OPTIONS = set(range(4096, 32769, 4096))
DEFAULT_RES = 1536           # 默认图像长边上限：768 快 / 1024 均衡 / 1536 高清
VIDEO_MAX_FPS = 30
VIDEO_MAX_SAMPLES = 600
VIDEO_MAX_BYTES = 1024 * 1024 * 1024
VIDEO_PROMPT_FILE = os.path.join(BASE_DIR, "video_prompt.json")
KV_CACHE_OPTIONS = {"q8_0", "f16"}

# JoyCaption's embedded ChatML template concatenates message content as a string,
# which fails when llama.cpp receives the image_url/text content list.
JOYCAPTION_CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{%- set ns = namespace(content='') -%}"
    "{%- if message['content'] is string -%}"
    "{%- set ns.content = message['content'] -%}"
    "{%- else -%}"
    "{%- for item in message['content'] -%}"
    "{%- if item['type'] == 'text' -%}"
    "{%- set ns.content = ns.content ~ item['text'] -%}"
    "{%- elif item['type'] == 'image_url' -%}"
    "{%- set ns.content = ns.content ~ '<image>' -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + ns.content + eos_token -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}"
    "{%- endif -%}"
)

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

VIDEO_SYS_PROMPT = (
    "You are a precise multimodal video analyst. Infer the visible action and temporal progression "
    "from the supplied video frames, then follow the user's instructions to write a useful "
    "text-to-video generation prompt. Do not invent details that are not visible."
)
VIDEO_INSTRUCTION_ZH = (
    "按时间顺序分析这段视频抽取出的画面，并写成可用于视频生成模型的中文提示词。"
    "描述主体、场景、连续动作与动作变化、镜头运动、构图、光线、色彩和视觉风格；"
    "保持主体与环境在各帧之间的一致性。只写画面可见的信息，不推测声音或不可见内容。"
    "用连贯的一段文字输出最终提示词，不要写分析过程、时间戳、标题或列表。"
)
VIDEO_INSTRUCTION_EN = (
    "Analyze the supplied video frames in chronological order and write one text-to-video prompt. "
    "Describe the subject, setting, continuous action and its progression, camera movement, "
    "composition, lighting, colors, and visual style while keeping subjects and surroundings "
    "consistent across frames. Include only visible information; do not infer audio or unseen events. "
    "Output one coherent paragraph containing only the final prompt, with no analysis, timestamps, "
    "heading, or list."
)
VIDEO_BUILTIN_PRESETS = {
    "默认提示词": VIDEO_SYS_PROMPT,
    "动作时间线": (
        "Analyze the supplied video frames chronologically. Track each subject consistently and "
        "describe visible actions as they develop, along with meaningful camera movement, framing, "
        "lighting, color, and setting. Return one fluent text-to-video prompt in the requested "
        "language. Do not invent audio or events outside the visible frames."
    ),
    "分镜描述": (
        "Analyze the supplied video frames in chronological order and organize the visible content "
        "into concise shots. For each shot, describe the composition, subject positions, action, "
        "camera movement, and transition while preserving visual continuity. Use only evidence from "
        "the frames and follow the user's requested output language and format."
    ),
}
VIDEO_DEFAULT_PRESET = "默认提示词"

# ---------------------------------------------------------------- 模型加载 ---

_model = None
_state = {"status": "unloaded",
          "device": "GPU（llama.cpp 自动 offload）" if DEVICE == "cuda" else "CPU",
          "model": None, "n_ctx": None, "kv_cache": None,
          "error": None}  # 模型默认不自动加载
_infer_lock = threading.Lock()      # llama.cpp 同一上下文不支持并发推理，必须串行化
_model_op_lock = threading.Lock()   # 串行化 加载/卸载 操作
_video_jobs = {}
_video_jobs_lock = threading.Lock()


def _video_job_id(value=None):
    try:
        return uuid.UUID(str(value)).hex
    except (ValueError, TypeError, AttributeError):
        return uuid.uuid4().hex


def _set_video_job(job_id, **values):
    with _video_jobs_lock:
        job = _video_jobs.setdefault(job_id, {})
        job.update(values)
        job["updated"] = time.time()
        expired = [key for key, value in _video_jobs.items()
                   if time.time() - value.get("updated", 0) > 3600]
        for key in expired:
            _video_jobs.pop(key, None)


def _get_video_job(job_id):
    with _video_jobs_lock:
        job = _video_jobs.get(job_id)
        return dict(job) if job else None

# ---- 系统提示词多方案（可切换 / 新增 / 删除，离线保存在 prompts.json）----
PRESETS_FILE = os.path.join(BASE_DIR, "prompts.json")
BUILTIN_PRESETS = {          # 出厂方案（"恢复默认"按方案名还原，自定义方案还原为 DEFAULT_SYS_PROMPT）
    "默认人设": DEFAULT_SYS_PROMPT,
    "中文结构化": INSTRUCTION_ZH,
    "英文提示词": INSTRUCTION_EN,
}
_prompt_presets = dict(BUILTIN_PRESETS)   # {方案名: 系统提示词内容}
_prompt_current = "默认人设"
_video_presets = dict(VIDEO_BUILTIN_PRESETS)
_video_prompt_current = VIDEO_DEFAULT_PRESET
_video_sys_prompt = VIDEO_SYS_PROMPT


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
        return True
    except Exception as e:
        print(f"[sys] 保存提示词方案失败: {e}")
        return False


def _current_sys_prompt():
    return _prompt_presets.get(_prompt_current, DEFAULT_SYS_PROMPT)


def _load_video_presets():
    global _video_presets, _video_prompt_current, _video_sys_prompt
    _video_presets = dict(VIDEO_BUILTIN_PRESETS)
    _video_prompt_current = VIDEO_DEFAULT_PRESET
    try:
        with open(VIDEO_PROMPT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_presets = data.get("presets")
        if isinstance(saved_presets, dict):
            cleaned = {str(k).strip(): str(v)
                       for k, v in saved_presets.items() if str(k).strip()}
            if cleaned:
                _video_presets = cleaned
                current = str(data.get("current") or "").strip()
                _video_prompt_current = (current if current in cleaned
                                         else next(iter(cleaned)))
        else:
            prompt = data.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                _video_presets[VIDEO_DEFAULT_PRESET] = prompt.strip()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[video] 读取提示词方案失败: {e}")
    _video_sys_prompt = _video_presets.get(_video_prompt_current, VIDEO_SYS_PROMPT)


def _save_video_presets():
    try:
        with open(VIDEO_PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"current": _video_prompt_current, "presets": _video_presets},
                      f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[video] 保存提示词方案失败: {e}")
        return False


def _save_video_sys_prompt(prompt):
    global _video_sys_prompt, _video_presets
    prompt = str(prompt or "").strip()
    if not prompt:
        return False
    _video_presets[_video_prompt_current] = prompt
    _video_sys_prompt = prompt
    return _save_video_presets()


_load_presets()
_load_video_presets()


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


def load_model(key, n_ctx=N_CTX, kv_cache="q8_0"):
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
        from llama_cpp._ggml import GGMLType

        cache_type = {
            "q8_0": int(GGMLType.GGML_TYPE_Q8_0),
            "f16": int(GGMLType.GGML_TYPE_F16),
        }[kv_cache]

        _model = Llama(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_gpu_layers=-1,      # 全部层 offload 到 GPU
            n_ctx=n_ctx,
            type_k=cache_type,
            type_v=cache_type,
            flash_attn=(DEVICE == "cuda"),
            verbose=False,
        )
        if "joycaption" in m["model"].lower():
            handler = _model.chat_handler
            handler.chat_format = JOYCAPTION_CHAT_TEMPLATE
            handler._change_chat_template(JOYCAPTION_CHAT_TEMPLATE)
            handler._template_initialized = False
            print("[model] applied multimodal JoyCaption chat template")
        _state["model"] = key
        _state["n_ctx"] = n_ctx
        _state["kv_cache"] = kv_cache
        _state["status"] = "ready"
        print(f"[model] 就绪，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        _state["status"] = "error"
        _state["model"] = None
        _state["n_ctx"] = None
        _state["kv_cache"] = None
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
        _state["n_ctx"] = None
        _state["kv_cache"] = None
        print("[model] 已卸载，显存已释放")
        return True, "已卸载，显存已释放"


def start_load(key, n_ctx=N_CTX, kv_cache="q8_0"):
    """后台线程重新加载模型。"""
    if _state["status"] in ("loading", "ready"):
        return False, f"当前状态为 {_state['status']}，无需加载"
    with _model_op_lock:
        if _state["status"] in ("loading", "ready"):
            return False, f"当前状态为 {_state['status']}，无需加载"
        _state["status"] = "loading"
        _state["error"] = None
        _state["n_ctx"] = n_ctx
        _state["kv_cache"] = kv_cache
    threading.Thread(target=load_model, args=(key, n_ctx, kv_cache), daemon=True).start()
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


def extract_video_frames(video_path, sample_fps, max_side):
    """按目标帧率取样，缩放后以 JPEG 暂存，控制长视频的内存占用。"""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(video_path, pix_fmt="rgb24")
    try:
        meta = next(reader)
        width, height = meta["size"]
        fps = float(meta.get("fps") or 0)
        duration = float(meta.get("duration") or 0)
        if width <= 0 or height <= 0 or fps <= 0 or duration <= 0:
            raise ValueError("无法读取视频时长或帧率")

        total_frames = max(1, int(round(duration * fps)))
        requested_samples = max(1, math.ceil(duration * sample_fps))
        if requested_samples > VIDEO_MAX_SAMPLES:
            raise ValueError(
                f"按 {sample_fps} FPS 预计抽取 {requested_samples} 帧，超过单次 {VIDEO_MAX_SAMPLES} 帧上限；"
                "请降低采样帧率或缩短视频"
            )
        if requested_samples >= total_frames:
            targets = list(range(total_frames))
        else:
            targets = sorted({int(round(i * fps / sample_fps))
                              for i in range(requested_samples)})
        target_set = set(targets)
        extracted = []
        last_frame = None
        last_index = -1
        for index, raw in enumerate(reader):
            last_index, last_frame = index, raw
            if index not in target_set:
                continue
            image = Image.frombytes("RGB", (width, height), raw)
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            extracted.append((index / fps, buffer.getvalue(), image.size))
            if index >= targets[-1]:
                break

        if last_frame is None:
            raise ValueError("视频中没有可读取的画面")
        if len(extracted) < len(targets):
            image = Image.frombytes("RGB", (width, height), last_frame)
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            extracted.append((min(duration, last_index / fps), buffer.getvalue(), image.size))
        return extracted, duration
    finally:
        reader.close()


def analyze_video_frames(frames, lang="zh", max_tokens=4096,
                         segment_info=None, previous_segments=None):
    if segment_info:
        idx, total, start, end = segment_info
        if lang == "en":
            opening = (f"This is segment {idx} of {total}, covering {start:.2f} to "
                       f"{end:.2f} seconds of the video. Analyze its frames in order.")
            instruction = VIDEO_INSTRUCTION_EN + (
                " Describe this segment only, with enough concrete action and visual details "
                "to combine it with the other segments later."
            )
        else:
            opening = (f"这是视频第 {idx}/{total} 段，时间范围 {start:.2f}–{end:.2f} 秒。"
                       "请按顺序分析本段画面。")
            instruction = VIDEO_INSTRUCTION_ZH + (
                "只描述本段内容，保留足够具体的动作与视觉细节，以便之后与其他分段合并。"
            )
    else:
        opening = (f"Analyze {len(frames)} frames sampled from this video in chronological order. "
                   "Each timestamp is relative to the start of the video.")
        instruction = VIDEO_INSTRUCTION_EN if lang == "en" else VIDEO_INSTRUCTION_ZH
    content = [{"type": "text", "text": opening}]
    if previous_segments:
        if lang == "en":
            previous_header = "Observations from earlier segments (use only for continuity):"
        else:
            previous_header = "前面分段的观察（仅用于保持连续性）："
        content.append({"type": "text", "text": previous_header + "\n" + previous_segments})
    for timestamp, image_data, _size in frames:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content.append({"type": "text", "text": f"Frame at {timestamp:.2f} seconds:"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{b64}"
        }})
    content.append({"type": "text", "text": instruction})

    t0 = time.time()
    resp = _model.create_chat_completion(
        messages=[
            {"role": "system", "content": _video_sys_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.8,
        top_k=20,
    )
    sec = time.time() - t0
    usage = resp.get("usage") or {}
    n_tok = usage.get("completion_tokens") or 0
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    stats = {"sec": round(sec, 1), "tokens": n_tok,
             "tps": round(n_tok / sec, 1) if sec > 0 else 0.0}
    return text.strip('"').strip(), stats


def synthesize_video_prompt(segments, lang="zh", max_tokens=4096):
    if lang == "en":
        instruction = (
            "Combine the chronological segment descriptions into one coherent text-to-video prompt. "
            "Preserve the order of actions and camera changes, maintain subject and scene continuity, "
            "remove repetition, and include only details supported by the descriptions. Output only "
            "one fluent paragraph with no preface or explanation."
        )
        header = "Chronological segment descriptions:"
    else:
        instruction = (
            "将这些按时间顺序排列的分段描述整合为一条连贯的视频生成提示词。保留动作和镜头变化的先后顺序，"
            "保持主体与场景一致，删去重复内容，只保留分段描述支持的细节。最终只输出一段流畅的提示词，"
            "不要添加前言或解释。"
        )
        header = "按时间顺序排列的分段描述："
    content = [{"type": "text", "text": header}]
    for segment in segments:
        content.append({"type": "text", "text": (
            f"Segment {segment['index']} ({segment['start']:.2f}-{segment['end']:.2f}s): "
            f"{segment['prompt']}"
        )})
    content.append({"type": "text", "text": instruction})
    t0 = time.time()
    resp = _model.create_chat_completion(
        messages=[
            {"role": "system", "content": _video_sys_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.8,
        top_k=20,
    )
    sec = time.time() - t0
    usage = resp.get("usage") or {}
    n_tok = usage.get("completion_tokens") or 0
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    stats = {"sec": round(sec, 1), "tokens": n_tok,
             "tps": round(n_tok / sec, 1) if sec > 0 else 0.0}
    return text.strip('"').strip(), stats


# ---------------------------------------------------------------- Flask ---

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = VIDEO_MAX_BYTES


@app.after_request
def no_store(resp):
    # iframe 内嵌时禁止浏览器缓存，保证门户主题参数与代码更新即时生效
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"ok": False, "error": "文件超过 1 GB 上传限制"}), 413

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
  .mode-switch { display:flex; gap:8px; margin:10px 0 8px; }
  .mode-switch button { min-width:112px; }
  .mode-switch button.active { border-color:#2f6feb; background:#2f6feb; color:#fff; }
  .mode-panel[hidden] { display:none; }
  .field { display:flex; align-items:center; gap:7px; color:var(--muted); font-size:13px; }
  .video-drop { border:2px dashed var(--btn-border); border-radius:8px; padding:20px;
                text-align:center; color:var(--muted); cursor:pointer; }
  .video-drop.over { border-color:#6ea8fe; background:var(--card); }
  .video-file { display:flex; flex-direction:column; align-items:center; gap:12px; margin-top:12px; }
  .video-file[hidden] { display:none; }
  .video-file video { display:block; width:100%; max-width:640px; max-height:360px;
                      object-fit:contain; background:#000; }
  .video-file-info { display:flex; flex-direction:column; align-items:center; gap:8px; min-width:0; }
  .video-file-info span { display:block; overflow-wrap:anywhere; color:var(--muted); font-size:13px; text-align:center; }
  .video-actions { margin-top:12px; }
  .video-settings-table { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr));
                          gap:14px 16px; margin-top:14px; padding:4px 0;
                          align-items:center; justify-items:center; }
  .video-settings-table.cols-3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
  .video-setting { display:flex; flex-direction:column; align-items:center; justify-content:center;
                   gap:6px; min-width:0; width:100%; color:var(--muted); font-size:13px;
                   text-align:center; }
  .video-setting-title { display:flex; align-items:center; justify-content:center;
                         gap:4px; line-height:1.35; }
  .segment-toggle { display:flex; align-items:center; gap:5px; white-space:nowrap; }
  .segment-setting.is-disabled .video-setting-title,
  .segment-setting.is-disabled .video-stepper span { color:var(--muted); opacity:.55; }
  .video-setting .sel, .video-setting input { width:100%; min-width:0; height:36px; }
  .video-setting .segment-toggle input { width:auto; height:auto; }
  .video-stepper { display:flex; align-items:center; gap:8px; }
  .video-stepper input { max-width:110px; }
  .video-stepper input::-webkit-outer-spin-button,
  .video-stepper input::-webkit-inner-spin-button { opacity:1; -webkit-appearance:inner-spin-button; }
  .video-stepper span { color:var(--muted); white-space:nowrap; }
  #videoFrameEstimate { margin-top:8px; color:var(--muted); font-size:12px; }
  #videoResult { display:none; }
  #videoResult pre { min-height:120px; }
  #videoSegmentsWrap { margin-top:10px; border-top:1px solid var(--border); padding-top:8px; }
  #videoSegmentsWrap summary { cursor:pointer; color:var(--muted); font-size:13px; }
  .segment-result { padding:8px 0; border-bottom:1px solid var(--border); }
  .segment-result:last-child { border-bottom:0; }
  .segment-title { font-size:12px; color:var(--muted); margin-bottom:4px; }
  .segment-text { font-size:13px; line-height:1.5; white-space:pre-wrap; }
  @media (max-width:900px) {
    .video-settings-table { grid-template-columns:repeat(3, minmax(0, 1fr)); }
  }
  @media (max-width:620px) {
    body { padding:16px; }
    .video-settings-table { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .video-settings-table.cols-3 { grid-template-columns:repeat(1, minmax(0, 1fr)); }
  }
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
  .syscard { display:flex; justify-content:space-between; align-items:center; gap:12px;
             margin-top:14px; padding:12px 0; border-top:1px solid var(--border); }
  .syshead { display:flex; flex-direction:column; gap:4px; min-width:0;
             font-size:13px; color:var(--muted); }
  .prompt-current { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #videoStatus { margin-top:12px; color:var(--muted); font-size:13px; }
  .pmask { position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:1000;
           display:flex; align-items:center; justify-content:center; }
  .pmask[hidden] { display:none; }   /* display:flex 会覆盖 hidden 属性，需显式还原 */
  .pmodal { background:var(--card); border:1px solid var(--border); border-radius:12px;
            width:min(1120px, 96vw); height:min(86vh, 900px); max-height:900px;
            display:flex; flex-direction:column; gap:12px; padding:20px; }
  .phead { display:flex; justify-content:space-between; align-items:center;
           font-size:16px; font-weight:600; }
  .phead button { min-width:36px; min-height:36px; padding:4px; }
  .prompt-assignments { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr));
                        gap:12px; padding-bottom:12px; border-bottom:1px solid var(--border); }
  .prompt-assignment { display:flex; align-items:center; gap:10px; min-width:0;
                       color:var(--muted); font-size:13px; }
  .prompt-assignment select { flex:1; min-width:0; height:36px; }
  .prompt-manager-layout { display:grid; grid-template-columns:minmax(220px, 280px) minmax(0, 1fr);
                           gap:16px; flex:1; min-height:0; }
  .prompt-sidebar, .prompt-editor { display:flex; flex-direction:column; min-width:0; min-height:0; }
  .prompt-sidebar-head, .prompt-editor-head, .prompt-editor-actions {
    display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .prompt-sidebar-head, .prompt-editor-head { min-height:36px; color:var(--muted); font-size:13px; }
  .prompt-scope { display:flex; gap:6px; margin:8px 0; }
  .prompt-scope button { flex:1; padding:6px 8px; font-size:12px; }
  .prompt-scope button.active { background:#2f6feb; border-color:#2f6feb; color:#fff; }
  #presetList { overflow-y:auto; display:flex; flex-direction:column; gap:6px; flex:1; min-height:0; }
  .pitem { display:flex; flex-direction:column; align-items:stretch; gap:4px; padding:9px 10px;
           cursor:pointer; text-align:left; white-space:normal;
           border:1px solid var(--btn-border); border-radius:8px; font-size:13px; }
  .pitem:hover { background:var(--inner); }
  .pitem.cur { border-color:#4a9eff; }
  .pitem-top { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .pitem .pname { font-weight:600; overflow-wrap:anywhere; }
  .pitem .pcurrent { color:#4a9eff; font-size:11px; white-space:nowrap; }
  .pitem .pprev { color:var(--muted); font-size:12px; overflow:hidden;
                  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .pnew { display:flex; gap:8px; margin-top:10px; }
  .pnew input { flex:1; min-width:0; background:var(--inner); color:var(--text);
                border:1px solid var(--btn-border); border-radius:6px; padding:7px 10px;
                font-size:13px; font-family:inherit; }
  .prompt-editor { gap:8px; }
  #promptEditor { flex:1; width:100%; min-height:0; resize:none; box-sizing:border-box;
                  background:var(--inner); color:var(--text); border:1px solid var(--btn-border);
                  border-radius:6px; padding:12px; font-family:inherit; font-size:13px;
                  line-height:1.55; }
  #promptStatus { min-height:18px; color:var(--muted); font-size:12px; }
  #promptStatus.error { color:#f08080; }
  .prompt-editor-actions { flex-wrap:wrap; }
  .prompt-editor-actions .right { display:flex; gap:8px; margin-left:auto; }
  .prompt-editor-actions button { padding:7px 12px; }
  .prompt-editor-actions button:disabled { opacity:.45; cursor:not-allowed; }
  @media (max-width:700px) {
    .pmodal { width:96vw; height:92vh; padding:14px; }
    .prompt-assignments { grid-template-columns:1fr; gap:8px; }
    .prompt-manager-layout { grid-template-columns:minmax(0, 1fr); grid-template-rows:minmax(150px, .65fr) minmax(260px, 1.35fr); }
    .prompt-sidebar { min-height:0; }
    #presetList { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); align-content:start; }
    .pnew { margin-top:6px; }
  }
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
  <div class="mode-switch" role="tablist" aria-label="反推类型">
    <button id="modeImage" class="active" role="tab" aria-selected="true">图片反推</button>
    <button id="modeVideo" role="tab" aria-selected="false">视频反推</button>
  </div>
  <div class="sub">GGUF 视觉模型（llama.cpp）本地推理 · 图片与视频提示词反推</div>


  <div class="btns" style="margin-bottom:14px;">
    <button id="btnLoad">加载模型</button>
    <button id="btnUnload">卸载模型（释放显存）</button>
    <span style="align-self:center;font-size:13px;color:var(--muted);">模型：
      <select id="modelSel" class="sel"></select>
    </span>
    <span style="align-self:center;font-size:13px;color:var(--muted);">上下文长度：
      <select id="ctxSel" class="sel" title="视频多帧分析建议使用较大上下文">
        <option value="4096">4K</option>
        <option value="8192">8K</option>
        <option value="12288">12K</option>
        <option value="16384" selected>16K</option>
        <option value="20480">20K</option>
        <option value="24576">24K</option>
        <option value="28672">28K</option>
        <option value="32768">32K</option>
      </select>
    </span>
    <span style="align-self:center;font-size:13px;color:var(--muted);">KV 缓存：
      <select id="kvCacheSel" class="sel" title="Q8_0 可减少缓存显存占用，F16 兼容性更广">
        <option value="q8_0" selected>Q8_0（8-bit，省显存）</option>
        <option value="f16">F16（标准）</option>
      </select>
    </span>
  </div>

  <section id="imagePanel" class="mode-panel">
  <div id="drop">将图片或文件夹拖到这里，或点击选择</div>
  <div class="btns">
    <button class="primary" id="pickFiles">选择图片（可多选）</button>
    <button id="pickDir">选择文件夹</button>
    <button id="clear">清空列表</button>
  </div>
  <div class="video-settings-table cols-3" role="group" aria-label="图片反推参数">
    <label class="video-setting"><span class="video-setting-title">最长边分辨率</span>
      <select id="resSel" class="sel">
        <option value="768">768（快速）</option>
        <option value="1024">1024（均衡）</option>
        <option value="1536" selected>1536（高清）</option>
      </select>
    </label>
    <label class="video-setting"><span class="video-setting-title">最大输出长度</span>
      <select id="maxtSel" class="sel">
        <option value="512">512</option>
        <option value="768">768</option>
        <option value="1024" selected>1024</option>
        <option value="1536">1536</option>
        <option value="2048">2048</option>
      </select>
    </label>
    <label class="video-setting"><span class="video-setting-title">提示词语言</span>
      <select id="langSel" class="sel">
        <option value="zh" selected>中文</option>
        <option value="en">English</option>
      </select>
    </label>
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
      <span>图片系统提示词</span>
      <span id="imagePromptCurrent" class="prompt-current">当前预设：加载中</span>
    </div>
    <button id="manageImagePrompt">管理提示词</button>
  </div>
  </section>

  <section id="videoPanel" class="mode-panel" hidden>
    <div id="videoDrop" class="video-drop">将 MP4 视频拖到这里，或点击选择</div>
    <input type="file" id="videoInput" accept="video/mp4,.mp4" hidden>
    <div id="videoFile" class="video-file" hidden>
      <video id="videoPreview" controls preload="metadata"></video>
      <div class="video-file-info">
        <span id="videoName"></span>
        <button id="videoRemove" title="移除视频">移除视频</button>
      </div>
    </div>
    <div class="btns video-actions">
      <button class="primary" id="videoPick">选择 MP4</button>
      <button id="videoRun" disabled>开始视频反推</button>
    </div>
    <div class="video-settings-table" role="group" aria-label="视频反推参数">
      <div class="video-setting segment-setting" id="segmentSetting">
        <div class="video-setting-title">
          <label class="segment-toggle" for="segmentEnabled">
            <input type="checkbox" id="segmentEnabled"><span>分段反推 (省显存)</span>
          </label>
        </div>
        <span class="video-stepper">
          <input type="number" id="framesPerSegment" class="sel" min="1" max="60" step="1" value="8">
          <span>帧/段</span>
        </span>
      </div>
      <label class="video-setting"><span class="video-setting-title">采样帧率（每秒帧数）</span>
        <span class="video-stepper">
          <input type="number" id="videoFps" class="sel" min="1" max="30" step="1" value="1">
          <span>FPS</span>
        </span>
      </label>
      <label class="video-setting"><span class="video-setting-title">最长边分辨率</span>
        <select id="videoRes" class="sel">
          <option value="512">512</option><option value="768" selected>768</option>
          <option value="1024">1024</option><option value="1536">1536</option>
        </select>
      </label>
      <label class="video-setting"><span class="video-setting-title">最大输出长度</span>
        <select id="videoMaxTokens" class="sel">
          <option value="512">512</option><option value="1024">1024</option>
          <option value="2048">2048</option><option value="4096" selected>4096</option>
          <option value="6144">6144</option><option value="8192">8192</option>
        </select>
      </label>
      <label class="video-setting"><span class="video-setting-title">提示词语言</span>
        <select id="videoLang" class="sel">
          <option value="zh" selected>中文</option><option value="en">English</option>
        </select>
      </label>
    </div>
    <div id="videoFrameEstimate"></div>
    <div id="videoStatus" class="status"></div>
    <div id="videoResult" class="card">
      <div class="body">
        <div id="videoStats" class="status"></div>
        <pre id="videoPrompt"></pre>
        <div class="ops">
          <button id="videoCopy">复制提示词</button>
          <button id="videoDownload">下载提示词</button>
        </div>
        <details id="videoSegmentsWrap" hidden>
          <summary id="videoSegmentsSummary">分段分析结果</summary>
          <div id="videoSegments"></div>
        </details>
      </div>
    </div>
    <div class="syscard">
      <div class="syshead">
        <span>视频系统提示词</span>
        <span id="videoPromptCurrent" class="prompt-current">当前预设：加载中</span>
      </div>
      <button id="manageVideoPrompt">管理提示词</button>
    </div>
  </section>
</div>

<div id="presetMask" class="pmask" hidden>
  <section class="pmodal" role="dialog" aria-modal="true" aria-labelledby="promptManagerTitle">
    <header class="phead">
      <span id="promptManagerTitle">系统提示词管理</span>
      <button id="promptClose" aria-label="关闭">✕</button>
    </header>
    <div class="prompt-assignments">
      <label class="prompt-assignment" for="imagePresetSelect">图片使用
        <select id="imagePresetSelect" class="sel"></select>
      </label>
      <label class="prompt-assignment" for="videoPresetSelect">视频使用
        <select id="videoPresetSelect" class="sel"></select>
      </label>
    </div>
    <div class="prompt-manager-layout">
      <aside class="prompt-sidebar">
        <div class="prompt-sidebar-head"><span id="promptListTitle">图片预设</span></div>
        <div class="prompt-scope" role="tablist" aria-label="预设类型">
          <button id="promptScopeImage" role="tab" aria-selected="true">图片</button>
          <button id="promptScopeVideo" role="tab" aria-selected="false">视频</button>
        </div>
        <div id="presetList"></div>
        <div class="pnew">
          <input id="presetName" placeholder="新预设名称" aria-label="新预设名称">
          <button id="presetAdd">新建</button>
        </div>
      </aside>
      <section class="prompt-editor">
        <div class="prompt-editor-head">
          <span id="presetEditingName">选择一个预设</span>
          <button id="presetDelete" title="删除所选预设">删除预设</button>
        </div>
        <textarea id="promptEditor" spellcheck="false" aria-label="系统提示词内容"></textarea>
        <div id="promptStatus" role="status"></div>
        <div class="prompt-editor-actions">
          <button id="presetReset">恢复预设默认</button>
          <div class="right">
            <button id="presetSave" disabled>保存更改</button>
            <button id="promptDone" class="primary">完成</button>
          </div>
        </div>
      </section>
    </div>
  </section>
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
let videoFileObj = null;
let videoObjectUrl = null;
let videoRunning = false;
let videoProgressTimer = null;
let videoDuration = 0;
let videoSampleLimitExceeded = false;
let modelReady = false;
const $ = id => document.getElementById(id);
const listEl = $('list'), statsEl = $('stats'), dropEl = $('drop');

function setMode(mode) {
  const image = mode === 'image';
  $('imagePanel').hidden = !image;
  $('videoPanel').hidden = image;
  $('modeImage').classList.toggle('active', image);
  $('modeVideo').classList.toggle('active', !image);
  $('modeImage').setAttribute('aria-selected', String(image));
  $('modeVideo').setAttribute('aria-selected', String(!image));
}
$('modeImage').onclick = () => setMode('image');
$('modeVideo').onclick = () => setMode('video');

// ---- 模型状态轮询 ----
async function poll() {
  try {
    const r = await fetch('/api/status'); const s = await r.json();
    const b = $('badge');
    if (s.status === 'ready') {
      b.textContent = '模型就绪' + (s.vram != null ? ' · 显存 ' + s.vram + 'G · 空闲 ' + s.free + 'G' : '')
                    + (s.n_ctx ? ' · 上下文 ' + Math.round(s.n_ctx / 1024) + 'K' : '')
                    + (s.kv_cache ? ' · KV ' + s.kv_cache.toUpperCase() : '')
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
    if (s.n_ctx) $('ctxSel').value = String(s.n_ctx);
    if (s.kv_cache) $('kvCacheSel').value = s.kv_cache;
    const canConfigureModel = (s.status === 'unloaded' || s.status === 'error');
    $('modelSel').disabled = !canConfigureModel;
    $('ctxSel').disabled = !canConfigureModel;
    $('kvCacheSel').disabled = !canConfigureModel;
    $('btnLoad').disabled = (s.status === 'loading' || s.status === 'ready');
    $('btnUnload').disabled = (s.status !== 'ready' || videoRunning);
    modelReady = s.status === 'ready';
    syncVideoButton();
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
    sel.disabled = (j.status !== 'unloaded' && j.status !== 'error');
  } catch (e) {}
})();

// ---- 统一提示词管理 ----
let promptMode = 'image';
let selectedPresetName = null;
let promptCollections = {
  image: { current: null, presets: [] },
  video: { current: null, presets: [] },
};
function promptCollection(mode=promptMode) { return promptCollections[mode]; }
function selectedPromptPreset() {
  return promptCollection().presets.find(p => p.name === selectedPresetName) || null;
}
function editorHasUnsavedChanges() {
  const preset = selectedPromptPreset();
  return Boolean(preset && $('promptEditor').value !== preset.prompt);
}
function setPromptStatus(message='', error=false) {
  $('promptStatus').textContent = message;
  $('promptStatus').classList.toggle('error', error);
}
function syncPromptEditorState() {
  const preset = selectedPromptPreset();
  $('presetEditingName').textContent = preset
    ? preset.name + (editorHasUnsavedChanges() ? ' · 未保存' : '')
    : '选择一个预设';
  $('presetSave').disabled = !preset || !editorHasUnsavedChanges();
  $('presetDelete').disabled = !preset || promptCollection().presets.length <= 1;
  $('presetReset').disabled = !preset;
}
function renderPromptAssignments() {
  for (const mode of ['image', 'video']) {
    const collection = promptCollection(mode);
    const select = $(mode === 'image' ? 'imagePresetSelect' : 'videoPresetSelect');
    select.innerHTML = '';
    for (const preset of collection.presets) {
      const option = document.createElement('option');
      option.value = preset.name;
      option.textContent = preset.name;
      select.appendChild(option);
    }
    select.value = collection.current || '';
    const current = collection.presets.find(p => p.name === collection.current);
    $(mode === 'image' ? 'imagePromptCurrent' : 'videoPromptCurrent').textContent =
      '当前预设：' + (current ? current.name : '—');
  }
}
function renderPresetList() {
  const collection = promptCollection();
  if (!collection.presets.some(p => p.name === selectedPresetName)) {
    selectedPresetName = collection.current || (collection.presets[0] && collection.presets[0].name);
  }
  $('promptListTitle').textContent = promptMode === 'image' ? '图片预设' : '视频预设';
  $('promptScopeImage').classList.toggle('active', promptMode === 'image');
  $('promptScopeVideo').classList.toggle('active', promptMode === 'video');
  $('promptScopeImage').setAttribute('aria-selected', String(promptMode === 'image'));
  $('promptScopeVideo').setAttribute('aria-selected', String(promptMode === 'video'));
  const wrap = $('presetList');
  wrap.innerHTML = '';
  for (const preset of collection.presets) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'pitem' + (preset.name === selectedPresetName ? ' cur' : '');
    const top = document.createElement('span'); top.className = 'pitem-top';
    const name = document.createElement('span'); name.className = 'pname'; name.textContent = preset.name;
    top.appendChild(name);
    if (preset.name === collection.current) {
      const current = document.createElement('span'); current.className = 'pcurrent';
      current.textContent = '使用中'; top.appendChild(current);
    }
    const preview = document.createElement('span'); preview.className = 'pprev';
    preview.textContent = preset.prompt.replace(/\s+/g, ' ').slice(0, 120);
    row.append(top, preview);
    row.onclick = () => {
      if (preset.name !== selectedPresetName && editorHasUnsavedChanges() &&
          !confirm('当前提示词有未保存的更改，切换后将丢弃这些更改。继续？')) return;
      selectedPresetName = preset.name;
      $('promptEditor').value = preset.prompt;
      setPromptStatus();
      renderPresetList();
      syncPromptEditorState();
    };
    wrap.appendChild(row);
  }
  syncPromptEditorState();
}
async function refreshPromptCollections() {
  const r = await fetch('/api/prompts');
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || '加载预设失败');
  promptCollections = {
    image: j.image || { current: j.current, presets: j.presets || [] },
    video: j.video || { current: null, presets: [] },
  };
  renderPromptAssignments();
  renderPresetList();
}
async function promptPresetAction(mode, action, values={}) {
  const r = await fetch('/api/prompts', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, action, ...values }) });
  return await r.json();
}
function loadSelectedPreset() {
  const preset = selectedPromptPreset();
  $('promptEditor').value = preset ? preset.prompt : '';
  setPromptStatus();
  syncPromptEditorState();
  renderPresetList();
}
async function openPromptManager(mode) {
  promptMode = mode;
  try {
    await refreshPromptCollections();
    selectedPresetName = promptCollection().current;
    loadSelectedPreset();
    $('presetMask').hidden = false;
    $('presetName').value = '';
  } catch (e) { setPromptStatus(e.message || '加载预设失败', true); }
}
function closePromptManager() {
  if (editorHasUnsavedChanges() &&
      !confirm('当前提示词有未保存的更改，关闭后将丢弃这些更改。继续？')) return;
  $('presetMask').hidden = true;
}
async function changePromptScope(mode) {
  if (mode === promptMode) return;
  if (editorHasUnsavedChanges() &&
      !confirm('当前提示词有未保存的更改，切换类型后将丢弃这些更改。继续？')) return;
  promptMode = mode;
  selectedPresetName = promptCollection().current;
  loadSelectedPreset();
}
async function changePromptAssignment(mode, name) {
  try {
    const result = await promptPresetAction(mode, 'select', { name });
    if (!result.ok) throw new Error(result.error || '切换预设失败');
    await refreshPromptCollections();
    setPromptStatus((mode === 'image' ? '图片' : '视频') + '已使用「' + name + '」');
  } catch (e) { setPromptStatus(e.message || '切换预设失败', true); }
}
$('manageImagePrompt').onclick = () => openPromptManager('image');
$('manageVideoPrompt').onclick = () => openPromptManager('video');
$('promptScopeImage').onclick = () => changePromptScope('image');
$('promptScopeVideo').onclick = () => changePromptScope('video');
$('imagePresetSelect').onchange = e => changePromptAssignment('image', e.target.value);
$('videoPresetSelect').onchange = e => changePromptAssignment('video', e.target.value);
$('promptEditor').oninput = () => {
  syncPromptEditorState();
  setPromptStatus(editorHasUnsavedChanges() ? '有未保存的更改' : '');
};
$('promptClose').onclick = closePromptManager;
$('promptDone').onclick = closePromptManager;
$('presetMask').onclick = e => { if (e.target === $('presetMask')) closePromptManager(); };
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !$('presetMask').hidden) closePromptManager();
});
$('presetSave').onclick = async () => {
  const preset = selectedPromptPreset();
  if (!preset) return;
  try {
    const result = await promptPresetAction(promptMode, 'save', {
      name: selectedPresetName, prompt: $('promptEditor').value,
    });
    if (!result.ok) throw new Error(result.error || '保存失败');
    await refreshPromptCollections();
    $('promptEditor').value = selectedPromptPreset()?.prompt || '';
    syncPromptEditorState();
    setPromptStatus('更改已保存');
  } catch (e) { setPromptStatus(e.message || '保存失败', true); }
};
$('presetAdd').onclick = async () => {
  const name = $('presetName').value.trim();
  if (!name) { setPromptStatus('请输入预设名称', true); return; }
  try {
    const result = await promptPresetAction(promptMode, 'new', {
      name, prompt: $('promptEditor').value,
    });
    if (!result.ok) throw new Error(result.error || '新建失败');
    $('presetName').value = '';
    await refreshPromptCollections();
    selectedPresetName = name;
    loadSelectedPreset();
    setPromptStatus('预设已创建；可在上方分别指定图片或视频使用');
  } catch (e) { setPromptStatus(e.message || '新建失败', true); }
};
$('presetDelete').onclick = async () => {
  const preset = selectedPromptPreset();
  if (!preset || !confirm('删除预设「' + preset.name + '」？')) return;
  try {
    const result = await promptPresetAction(promptMode, 'delete', { name: preset.name });
    if (!result.ok) throw new Error(result.error || '删除失败');
    selectedPresetName = result.current;
    await refreshPromptCollections();
    loadSelectedPreset();
    setPromptStatus('预设已删除');
  } catch (e) { setPromptStatus(e.message || '删除失败', true); }
};
$('presetReset').onclick = async () => {
  const preset = selectedPromptPreset();
  if (!preset || !confirm('将「' + preset.name + '」恢复为内置内容？')) return;
  try {
    const result = await promptPresetAction(promptMode, 'reset', { name: preset.name });
    if (!result.ok) throw new Error(result.error || '恢复失败');
    await refreshPromptCollections();
    loadSelectedPreset();
    setPromptStatus('已恢复内置内容');
  } catch (e) { setPromptStatus(e.message || '恢复失败', true); }
};
(async function initPromptManager() {
  try { await refreshPromptCollections(); } catch (e) { console.warn(e); }
})();

// ---- 加载 / 卸载模型 ----
$('btnLoad').onclick = async () => {
  $('btnLoad').disabled = true;
  await fetch('/api/load', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: $('modelSel').value, n_ctx: Number($('ctxSel').value),
                           kv_cache: $('kvCacheSel').value }) });
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

// ---- 视频反推 ----
function syncVideoButton() {
  const btn = $('videoRun');
  if (btn) btn.disabled = !videoFileObj || !modelReady || videoRunning || videoSampleLimitExceeded;
  const unload = $('btnUnload');
  if (unload) unload.disabled = !modelReady || videoRunning;
}
function setVideoStatus(message, error=false) {
  $('videoStatus').textContent = message;
  $('videoStatus').style.color = error ? '#f08080' : '';
}
function updateSegmentControls() {
  const input = $('videoFps');
  const sampleFps = Math.max(1, Math.min(30, Math.round(Number(input.value) || 1)));
  input.value = sampleFps;
  const estimatedFrames = videoDuration > 0 ? Math.ceil(videoDuration * sampleFps) : 8;
  videoSampleLimitExceeded = estimatedFrames > 600;
  $('videoFrameEstimate').textContent = videoDuration > 0
    ? `预计抽取约 ${estimatedFrames} 帧${videoSampleLimitExceeded ? '，超过 600 帧上限，请降低帧率或缩短视频' : ''}`
    : '选择视频后显示预计抽帧数';
  $('videoFrameEstimate').style.color = videoSampleLimitExceeded ? '#f08080' : '';
  const toggle = $('segmentEnabled');
  $('framesPerSegment').disabled = !toggle.checked;
  $('segmentSetting').classList.toggle('is-disabled', !toggle.checked);
  syncVideoButton();
}
$('segmentEnabled').onchange = updateSegmentControls;
$('videoFps').oninput = updateSegmentControls;
$('videoFps').onchange = updateSegmentControls;
$('framesPerSegment').onchange = () => {
  const input = $('framesPerSegment');
  input.value = Math.max(1, Math.min(60, Math.round(Number(input.value) || 1)));
};
$('videoPreview').onloadedmetadata = () => {
  videoDuration = Number.isFinite($('videoPreview').duration) ? $('videoPreview').duration : 0;
  updateSegmentControls();
};
updateSegmentControls();
function setVideoFile(file) {
  if (!file) return;
  if (!/\.mp4$/i.test(file.name)) return setVideoStatus('仅支持 MP4 视频', true);
  if (file.size > 1024 * 1024 * 1024) return setVideoStatus('视频超过 1 GB 上传限制', true);
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
  videoFileObj = file;
  videoDuration = 0;
  videoObjectUrl = URL.createObjectURL(file);
  $('videoPreview').src = videoObjectUrl;
  $('videoName').textContent = file.name + ' · ' + (file.size / 1024 / 1024).toFixed(1) + ' MB';
  $('videoFile').hidden = false;
  $('videoResult').style.display = 'none';
  updateSegmentControls();
  setVideoStatus(modelReady ? '视频已就绪，可以开始反推' : '视频已就绪，请先加载模型');
  syncVideoButton();
}
$('videoPick').onclick = () => $('videoInput').click();
$('videoDrop').onclick = () => $('videoInput').click();
$('videoInput').onchange = e => { setVideoFile(e.target.files[0]); e.target.value = ''; };
$('videoDrop').ondragover = e => { e.preventDefault(); $('videoDrop').classList.add('over'); };
$('videoDrop').ondragleave = () => $('videoDrop').classList.remove('over');
$('videoDrop').ondrop = e => {
  e.preventDefault(); $('videoDrop').classList.remove('over');
  setVideoFile(e.dataTransfer.files[0]);
};
$('videoRemove').onclick = () => {
  if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
  videoObjectUrl = null; videoFileObj = null; videoDuration = 0;
  $('videoPreview').removeAttribute('src'); $('videoPreview').load();
  $('videoFile').hidden = true; $('videoResult').style.display = 'none';
  updateSegmentControls();
  setVideoStatus('选择一个 MP4 视频'); syncVideoButton();
};
$('videoRun').onclick = async () => {
  if (!videoFileObj || !modelReady || videoRunning) return;
  videoRunning = true; syncVideoButton();
  $('videoResult').style.display = 'none';
  setVideoStatus('正在上传视频、提取画面并分析，请稍候…');
  const fd = new FormData();
  fd.append('video', videoFileObj, videoFileObj.name);
  fd.append('fps', $('videoFps').value);
  fd.append('res', $('videoRes').value);
  fd.append('maxt', $('videoMaxTokens').value);
  fd.append('lang', $('videoLang').value);
  const jobId = crypto.randomUUID();
  fd.append('segment_enabled', $('segmentEnabled').checked ? '1' : '0');
  fd.append('frames_per_segment', $('framesPerSegment').value);
  fd.append('job_id', jobId);
  videoProgressTimer = setInterval(() => pollVideoProgress(jobId), 700);
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 1800000);
  try {
    const r = await fetch('/api/infer-video', { method: 'POST', body: fd, signal: ctl.signal });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ('HTTP ' + r.status));
    $('videoPrompt').textContent = j.prompt || '';
    const s = j.stats || {};
    $('videoStats').textContent = `${s.frames || 0} 帧（${s.fps || 0} FPS） · ${s.segments > 1 ? s.segments + ' 段顺序分析 · ' : ''}视频 ${s.duration || 0}s · 提取 ${s.extract_sec || 0}s · 推理 ${s.sec || 0}s · ${s.tokens || 0} tok · 上下文 ${(s.n_ctx || 0) / 1024}K`;
    renderVideoSegments(j.segments || []);
    $('videoResult').style.display = 'flex';
    setVideoStatus('视频反推完成');
  } catch (e) {
    setVideoStatus(e.name === 'AbortError' ? '请求超时（30 分钟）' : String(e.message || e), true);
  } finally {
    clearTimeout(timer); clearInterval(videoProgressTimer); videoProgressTimer = null;
    videoRunning = false; syncVideoButton();
  }
};
async function pollVideoProgress(jobId) {
  try {
    const r = await fetch('/api/video-progress/' + encodeURIComponent(jobId));
    if (!r.ok) return;
    const j = await r.json();
    const p = j.progress || {};
    if (p.phase === 'extracting') setVideoStatus('正在服务端解码视频并提取画面…');
    else if (p.phase === 'segment') setVideoStatus(`正在反推第 ${p.segment}/${p.total} 段…`);
    else if (p.phase === 'synthesizing') setVideoStatus(`正在整合分段内容（${p.segment}/${p.total}）…`);
    else if (p.phase === 'analyzing') setVideoStatus('正在反推视频…');
  } catch (e) {}
}
function renderVideoSegments(segments) {
  const wrap = $('videoSegmentsWrap'), target = $('videoSegments');
  target.innerHTML = '';
  wrap.hidden = !segments.length;
  $('videoSegmentsSummary').textContent = `分段分析结果（${segments.length} 段）`;
  for (const segment of segments) {
    const row = document.createElement('div'); row.className = 'segment-result';
    const title = document.createElement('div'); title.className = 'segment-title';
    title.textContent = `第 ${segment.index} 段 · ${segment.start}s–${segment.end}s`;
    const text = document.createElement('div'); text.className = 'segment-text';
    text.textContent = segment.prompt;
    row.append(title, text); target.appendChild(row);
  }
}
$('videoCopy').onclick = async () => {
  try {
    await navigator.clipboard.writeText($('videoPrompt').textContent);
    $('videoCopy').textContent = '已复制';
  } catch (e) { setVideoStatus('复制失败，请手动选择结果文本', true); }
  setTimeout(() => $('videoCopy').textContent = '复制提示词', 1200);
};
$('videoDownload').onclick = () => {
  const blob = new Blob([$('videoPrompt').textContent], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = (videoFileObj ? videoFileObj.name.replace(/\.mp4$/i, '') : 'video') + '_prompt.txt';
  a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
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
    try:
        n_ctx = int(data.get("n_ctx", N_CTX))
    except (TypeError, ValueError):
        n_ctx = N_CTX
    kv_cache = str(data.get("kv_cache", "q8_0")).lower()
    if n_ctx not in VIDEO_CTX_OPTIONS:
        return jsonify({"ok": False, "message": "不支持的上下文长度"}), 400
    if kv_cache not in KV_CACHE_OPTIONS:
        return jsonify({"ok": False, "message": "不支持的 KV 缓存类型"}), 400
    if not key or key not in scan_models():
        return jsonify({"ok": False, "message": f"未知模型: {key}"}), 400
    if _state["status"] == "ready" and (
            _state.get("model") != key or _state.get("n_ctx") != n_ctx
            or _state.get("kv_cache") != kv_cache):
        unload_model()  # 切换模型、上下文或 KV 缓存：先释放当前模型再加载新的
    ok, msg = start_load(key, n_ctx, kv_cache)
    return jsonify({"ok": ok, "message": msg})


@app.post("/api/unload")
def api_unload():
    ok, msg = unload_model()
    return jsonify({"ok": ok, "message": msg})


@app.get("/api/prompts")
def api_prompts():
    """列出图片和视频的提示词方案（含当前选中项）。"""
    image_presets = [{"name": n, "prompt": p, "current": n == _prompt_current}
                     for n, p in _prompt_presets.items()]
    video_presets = [{"name": n, "prompt": p, "current": n == _video_prompt_current}
                     for n, p in _video_presets.items()]
    return jsonify({
        "ok": True,
        "current": _prompt_current,
        "presets": image_presets,
        "image": {"current": _prompt_current, "presets": image_presets},
        "video": {"current": _video_prompt_current, "presets": video_presets},
    })


@app.post("/api/prompts")
def api_prompts_set():
    """在图片或视频提示词集中执行方案操作。"""
    global _prompt_current, _prompt_presets, _video_prompt_current, _video_sys_prompt
    d = request.get_json(silent=True) or {}
    mode = "video" if d.get("mode") == "video" else "image"
    if mode == "video":
        presets = _video_presets
        current = _video_prompt_current
        builtin = VIDEO_BUILTIN_PRESETS
        fallback = VIDEO_SYS_PROMPT
    else:
        presets = _prompt_presets
        current = _prompt_current
        builtin = BUILTIN_PRESETS
        fallback = DEFAULT_SYS_PROMPT
    act = d.get("action")
    target = (d.get("name") or current).strip()
    if act == "select":
        if target not in presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        current = target
    elif act == "save":
        p = (d.get("prompt") or "").strip()
        if not p:
            return jsonify({"ok": False, "error": "提示词不能为空"}), 400
        if target not in presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        presets[target] = p
    elif act == "new":
        name = target if d.get("name") else ""
        if not name:
            return jsonify({"ok": False, "error": "方案名不能为空"}), 400
        if name in presets:
            return jsonify({"ok": False, "error": "方案名已存在"}), 400
        presets[name] = (d.get("prompt") or "").strip() or fallback
        target = name
    elif act == "delete":
        if target not in presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        if len(presets) <= 1:
            return jsonify({"ok": False, "error": "至少保留一个预设"}), 400
        del presets[target]
        if target == current:
            current = next(iter(presets))
    elif act == "reset":
        if target not in presets:
            return jsonify({"ok": False, "error": "方案不存在"}), 400
        presets[target] = builtin.get(target, fallback)
    else:
        return jsonify({"ok": False, "error": "未知操作"}), 400

    if mode == "video":
        _video_prompt_current = current
        _video_sys_prompt = presets[current]
        saved = _save_video_presets()
    else:
        _prompt_current = current
        saved = _save_presets()
    if not saved:
        return jsonify({"ok": False, "error": "预设保存失败"}), 500
    return jsonify({"ok": True, "current": current, "target": target,
                    "prompt": presets[current]})


@app.get("/api/video-prompt")
def api_video_prompt():
    return jsonify({"ok": True, "prompt": _video_sys_prompt})


@app.post("/api/video-prompt")
def api_video_prompt_set():
    data = request.get_json(silent=True) or {}
    if data.get("action") == "reset":
        prompt = VIDEO_SYS_PROMPT
    else:
        prompt = data.get("prompt") or ""
    if not _save_video_sys_prompt(prompt):
        return jsonify({"ok": False, "error": "提示词不能为空或保存失败"}), 400
    return jsonify({"ok": True, "prompt": _video_sys_prompt})


@app.get("/api/video-progress/<job_id>")
def api_video_progress(job_id):
    progress = _get_video_job(_video_job_id(job_id))
    if not progress:
        return jsonify({"ok": False, "error": "任务不存在"}), 404
    return jsonify({"ok": True, "progress": progress})


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


@app.post("/api/infer-video")
def infer_video():
    if _state["status"] != "ready":
        if _state["status"] == "unloaded":
            err = "模型已卸载，请先点击「加载模型」"
        elif _state["status"] == "loading":
            err = "模型加载中，请稍候"
        else:
            err = f"模型不可用（{_state['error'] or _state['status']}）"
        return jsonify({"ok": False, "error": err}), 503

    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "请选择 MP4 视频文件"}), 400
    if os.path.splitext(file.filename)[1].lower() != ".mp4":
        return jsonify({"ok": False, "error": "目前只支持 MP4 视频"}), 400

    try:
        sample_fps = int(request.form.get("fps", "1"))
        sample_fps = max(1, min(sample_fps, VIDEO_MAX_FPS))
        max_side = max(256, min(int(request.form.get("res", "768")), 2048))
        max_tokens = max(128, min(int(request.form.get("maxt", "4096")), 8192))
        lang = "en" if request.form.get("lang") == "en" else "zh"
        segment_enabled = request.form.get("segment_enabled") == "1"
        frames_per_segment = (int(request.form.get("frames_per_segment", "8"))
                              if segment_enabled else 8)
        frames_per_segment = max(1, min(frames_per_segment, 60))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "视频参数格式无效"}), 400

    job_id = _video_job_id(request.form.get("job_id"))
    _set_video_job(job_id, status="running", phase="extracting", segment=0,
                   total=0, error=None)
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="img2prompt-video-") as work_dir:
            video_path = os.path.join(work_dir, "input.mp4")
            file.save(video_path)
            frames, duration = extract_video_frames(video_path, sample_fps, max_side)
            extract_sec = round(time.time() - started, 1)
            n_ctx = int(_state.get("n_ctx") or N_CTX)
            segment_count = (math.ceil(len(frames) / frames_per_segment)
                             if segment_enabled else 1)
            frame_groups = ([frames[i:i + frames_per_segment]
                             for i in range(0, len(frames), frames_per_segment)]
                            if segment_enabled else [frames])
            _set_video_job(job_id, phase="segment" if segment_enabled else "analyzing",
                           segment=0, total=segment_count)

            def estimate_text_tokens(text):
                return int(len(text.split()) * 1.5) if lang == "en" else len(text)
            instruction_tokens = estimate_text_tokens(
                _video_sys_prompt + (VIDEO_INSTRUCTION_EN if lang == "en" else VIDEO_INSTRUCTION_ZH)
            ) + 512

            with _infer_lock:
                if _state["status"] != "ready" or _model is None:
                    message = "模型已不可用，请重新加载后重试"
                    _set_video_job(job_id, status="error", phase="error", error=message)
                    return jsonify({"ok": False, "error": message}), 503

                segment_results = []
                model_seconds = 0.0
                total_tokens = 0
                segment_max_tokens = (max(64, min(512, max_tokens // segment_count))
                                      if segment_enabled else max_tokens)
                for index, group in enumerate(frame_groups, 1):
                    if segment_enabled:
                        start_sec = 0.0 if index == 1 else group[0][0]
                        end_sec = (duration if index == segment_count
                                   else frame_groups[index][0][0])
                    else:
                        start_sec, end_sec = 0.0, duration
                    previous_text = (segment_results[-1]["prompt"]
                                     if segment_enabled and segment_results else "")
                    visual_tokens = sum(
                        ((size[0] + 27) // 28) * ((size[1] + 27) // 28)
                        for _timestamp, _image_data, size in group
                    )
                    estimated_tokens = (visual_tokens + segment_max_tokens +
                                        estimate_text_tokens(previous_text) + instruction_tokens)
                    if estimated_tokens > n_ctx:
                        message = (
                            f"第 {index} 段约需 {estimated_tokens:,} 个上下文 token，当前为 {n_ctx:,}。"
                            "请增加上下文长度，或降低采样帧率、分辨率、每段帧数或输出长度。"
                        )
                        _set_video_job(job_id, status="error", phase="error", error=message)
                        return jsonify({"ok": False, "error": message}), 400

                    _set_video_job(
                        job_id, phase="segment" if segment_enabled else "analyzing",
                        segment=index, total=segment_count,
                    )
                    info = (index, segment_count, start_sec, end_sec) if segment_enabled else None
                    segment_prompt, segment_stats = analyze_video_frames(
                        group, lang, segment_max_tokens, info, previous_text or None
                    )
                    model_seconds += segment_stats["sec"]
                    total_tokens += segment_stats["tokens"]
                    if segment_enabled:
                        segment_results.append({
                            "index": index,
                            "start": round(start_sec, 2),
                            "end": round(end_sec, 2),
                            "prompt": segment_prompt,
                        })
                    else:
                        prompt = segment_prompt
                        stats = segment_stats

                if segment_enabled:
                    if segment_count == 1:
                        prompt = segment_results[0]["prompt"]
                    else:
                        synth_instruction = (
                            "Combine chronological video segment descriptions into one coherent prompt."
                            if lang == "en" else "将按时间顺序排列的视频分段描述整合为一条连贯提示词。"
                        )
                        synth_fixed_tokens = estimate_text_tokens(
                            _video_sys_prompt + synth_instruction
                        ) + 512
                        intermediate_max_tokens = max(
                            64, min(512, max_tokens, n_ctx // 16)
                        )
                        chunk_limit = max(
                            256,
                            min(int(n_ctx * 0.55),
                                n_ctx - synth_fixed_tokens - intermediate_max_tokens),
                        )

                        def item_token_cost(item):
                            return estimate_text_tokens(item["prompt"]) + 64

                        summary_items = segment_results
                        while True:
                            summary_tokens = sum(item_token_cost(item)
                                                 for item in summary_items)
                            synthesis_budget = n_ctx - synth_fixed_tokens - summary_tokens
                            if synthesis_budget >= 128:
                                break

                            chunks, current, current_tokens = [], [], 0
                            for item in summary_items:
                                item_tokens = item_token_cost(item)
                                if current and current_tokens + item_tokens > chunk_limit:
                                    chunks.append(current)
                                    current, current_tokens = [], 0
                                current.append(item)
                                current_tokens += item_tokens
                            if current:
                                chunks.append(current)
                            if len(chunks) >= len(summary_items):
                                message = (
                                    "分段摘要仍超出当前上下文，请增加上下文长度，或提高每段帧数以减少分段。"
                                )
                                _set_video_job(job_id, status="error", phase="error", error=message)
                                return jsonify({"ok": False, "error": message}), 400

                            merged_items = []
                            merge_total = sum(len(chunk) > 1 for chunk in chunks)
                            merge_index = 0
                            for chunk in chunks:
                                if len(chunk) == 1:
                                    merged_items.append(chunk[0])
                                    continue
                                merge_index += 1
                                _set_video_job(job_id, phase="synthesizing",
                                               segment=merge_index, total=merge_total)
                                merged_prompt, merge_stats = synthesize_video_prompt(
                                    chunk, lang, intermediate_max_tokens
                                )
                                model_seconds += merge_stats["sec"]
                                total_tokens += merge_stats["tokens"]
                                merged_items.append({
                                    "index": chunk[0]["index"],
                                    "start": chunk[0]["start"],
                                    "end": chunk[-1]["end"],
                                    "prompt": merged_prompt,
                                })
                            summary_items = merged_items

                        _set_video_job(job_id, phase="synthesizing", segment=1, total=1)
                        prompt, final_stats = synthesize_video_prompt(
                            summary_items, lang, min(max_tokens, synthesis_budget)
                        )
                        model_seconds += final_stats["sec"]
                        total_tokens += final_stats["tokens"]
                    stats = {"sec": round(model_seconds, 1), "tokens": total_tokens,
                             "tps": round(total_tokens / model_seconds, 1)
                             if model_seconds > 0 else 0.0}
        stats["extract_sec"] = extract_sec
        stats["duration"] = round(duration, 2)
        stats["frames"] = len(frames)
        stats["fps"] = sample_fps
        stats["n_ctx"] = n_ctx
        stats["segments"] = segment_count
        _set_video_job(job_id, status="complete", phase="complete", segment=segment_count,
                       total=segment_count)
    except Exception as e:
        traceback.print_exc()
        _set_video_job(job_id, status="error", phase="error",
                       error=f"{e.__class__.__name__}: {e}")
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}"}), 500
    return jsonify({"ok": True, "prompt": prompt, "stats": stats,
                    "segments": segment_results if segment_enabled else []})


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
