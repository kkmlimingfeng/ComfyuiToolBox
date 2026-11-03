# -*- coding: utf-8 -*-
"""
数据集打标工具（单文件版）
- 选择文件夹，批量给 jpg/jpeg/png/webp/bmp/gif 图片打标，结果写入同名 .txt
- 检查按钮：校验图片能否正常解码，绿色显示通过数量、红色显示跳过数量
- 标签打标：WD14 系列 ONNX 模型（onnxruntime，优先 GPU，失败自动回退 CPU），
  可分别设置标签阈值 / 角色阈值
- 自然语言打标：复用 model/ 目录下的 GGUF 视觉模型（llama.cpp，自动 GPU offload），
  提示词可编辑
- 附加提示词：前缀追加 / 后缀追加（英文逗号分隔多个）
- 模型配置窗口：增删自定义模型、控制下拉框显示、下载模型、选择下载源（镜像/官方），
  窗口底部显示下载进度。未下载的模型在下拉框中置灰
- 模型配置持久化在 model/tagger/config.json
端口：7869
"""

import base64
import csv
import io
import json
import os
import shutil
import threading
import time
import traceback
import urllib.request
import webbrowser
from tkinter import Tk, filedialog

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))     # 统一启动器根目录
MODEL_DIR = os.path.join(ROOT_DIR, "model")
TAGGER_DIR = os.path.join(MODEL_DIR, "tagger")            # WD14 onnx 模型存放目录
CONFIG_PATH = os.path.join(TAGGER_DIR, "config.json")
PORT = 7869

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
MIN_ONNX_BYTES = 50 * 1024 * 1024   # 小于此值视为下载不完整

# ---------------------------------------------------------------- 模型注册 ---

# 内置标签打标模型：WD14 系列（SmilingWolf），均为 ONNX + selected_tags.csv
WD14_MODELS = [
    {"id": "wd-eva02-large-tagger-v3",
     "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
     "label": "WD EVA02-Large v3（动漫，推荐）", "size": "约 1.2GB"},
    {"id": "wd-v1-4-moat-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2",
     "label": "WD MOAT v2", "size": "约 320MB"},
    {"id": "wd-v1-4-convnextv2-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
     "label": "WD ConvNeXtV2 v2", "size": "约 390MB"},
    {"id": "wd-v1-4-swinv2-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
     "label": "WD SwinV2 v2", "size": "约 350MB"},
    {"id": "wd-v1-4-convnext-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
     "label": "WD ConvNeXt v2", "size": "约 340MB"},
    {"id": "wd-v1-4-vit-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-vit-tagger-v2",
     "label": "WD ViT v2（最小）", "size": "约 360MB"},
]
DEFAULT_FILES = ["model.onnx", "selected_tags.csv"]

# 内置标签打标模型：CL Tagger 系列（cella110n，SigLIP2 架构）。
# v1 用 tag_mapping.json，v2 用 model_vocabulary.json 且为 gated 仓库（下载需 HF Token）。
# 文件路径相对模型目录（v1/v2 的文件在仓库子目录里），推理走 HF AutoProcessor（NaFlex 动态分辨率）。
CL_MODELS = [
    {"id": "cl_tagger_1_02", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_02/model.onnx", "cl_tagger_1_02/tag_mapping.json"],
     "label": "CL Tagger v1.02（SigLIP2，4.2万标签，推荐阈值 0.9）",
     "size": "约 1.4GB", "def_gen": 0.9, "def_char": 0.9},
    {"id": "cl_tagger_1_01", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_01/model.onnx", "cl_tagger_1_01/tag_mapping.json"],
     "label": "CL Tagger v1.01（SigLIP2，4.2万标签）",
     "size": "约 1.4GB", "def_gen": 0.6, "def_char": 0.6},
    {"id": "cl_tagger_1_00", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_00/model.onnx", "cl_tagger_1_00/tag_mapping.json"],
     "label": "CL Tagger v1.00（SigLIP2，4.2万标签）",
     "size": "约 1.4GB", "def_gen": 0.6, "def_char": 0.6},
    {"id": "cl_tagger_v2_01a", "engine": "cl", "gated": True,
     "repo": "cella110n/cl_tagger_v2",
     "files": ["v2_01a/model.onnx", "v2_01a/model.onnx.data",
               "v2_01a/model_vocabulary.json", "v2_01a/model_metadata.json"],
     "label": "CL Tagger v2.01a（SigLIP2 最新，需 HF Token）",
     "size": "约 2.1GB", "def_gen": 0.5, "def_char": 0.5},
    {"id": "cl_tagger_v2_00", "engine": "cl", "gated": True,
     "repo": "cella110n/cl_tagger_v2",
     "files": ["v2_00/model.onnx", "v2_00/model.onnx.data",
               "v2_00/model_vocabulary.json", "v2_00/model_metadata.json"],
     "label": "CL Tagger v2.00（SigLIP2，需 HF Token）",
     "size": "约 2.1GB", "def_gen": 0.5, "def_char": 0.5},
]

CL_DEFAULT_PROCESSOR = "google/siglip2-so400m-patch16-naflex"

MIRRORS = {"mirror": "https://hf-mirror.com", "official": "https://huggingface.co"}


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.setdefault("mirror", "mirror")          # mirror | official
    cfg.setdefault("custom_models", [])         # [{id, repo, label, size, files}]
    cfg.setdefault("hidden", [])                # 下拉框隐藏的模型 id
    cfg.pop("hf_token", None)                   # Token 只存内存，不落盘
    return cfg


# HF Access Token：仅本次运行内有效（gated 仓库下载用），不写入任何文件
_hf_token = ""


def save_config(cfg):
    os.makedirs(TAGGER_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def wd14_dir(model_id):
    return os.path.join(TAGGER_DIR, model_id)


def model_entry(cfg, model_id):
    """按 id 查模型条目（内置 WD14 + CL）。"""
    for m in WD14_MODELS:
        if m["id"] == model_id:
            return dict(m, files=DEFAULT_FILES, engine="wd14")
    for m in CL_MODELS:
        if m["id"] == model_id:
            return dict(m)
    return None


def model_installed(m):
    """条目列出的文件都存在（WD14 的 model.onnx 另有最小体积校验）即视为已安装。"""
    d = wd14_dir(m["id"])
    for f in m.get("files", DEFAULT_FILES):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            return False
        if (m.get("engine", "wd14") == "wd14"
                and os.path.basename(f) == "model.onnx"
                and os.path.getsize(p) < MIN_ONNX_BYTES):
            return False
    return True


def scan_nl_models():
    """扫描 model/ 下的 GGUF 视觉模型（主模型 + mmproj 配对，同 img2prompt）。"""
    models = {}
    if not os.path.isdir(MODEL_DIR):
        return models
    mmprojs, mains = {}, []
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
        best = None
        for prefix in mmprojs:
            if stem.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        if not best:
            continue
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / 1024**3
        models["gguf:" + stem] = {
            "label": f"{stem}（≈{size:.1f}G）",
            "model": f,
            "mmproj": mmprojs[best],
        }
    return models


# ---------------------------------------------------------------- 下载 ---

_dl_lock = threading.Lock()
_dl_state = {"active": False, "id": None, "file": "", "pct": 0,
             "done_files": 0, "total_files": len(DEFAULT_FILES),
             "error": None, "finished": False}


def _url_download(url, dest_path, on_progress, token=""):
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    on_progress(round(done * 100 / total))
    return total


def _download_model(model_id, cfg):
    """下载一个标签模型的全部文件，按配置的下载源优先，另一个做兜底。"""
    global _dl_state
    m = model_entry(cfg, model_id)
    if not m:
        return "未知模型"
    d = wd14_dir(model_id)
    os.makedirs(d, exist_ok=True)
    files = m.get("files", DEFAULT_FILES)
    engine = m.get("engine", "wd14")
    token = _hf_token
    hosts = [MIRRORS.get(cfg.get("mirror", "mirror"), MIRRORS["mirror"])] + \
            [h for k, h in MIRRORS.items() if k != cfg.get("mirror", "mirror")]
    with _dl_lock:
        _dl_state.update(active=True, id=model_id, file="", pct=0, done_files=0,
                         total_files=len(files), error=None, finished=False)
    try:
        for fname in files:
            dest = os.path.join(d, fname)
            os.makedirs(os.path.dirname(dest) or d, exist_ok=True)
            # 已存在且完整则跳过（WD14 的 model.onnx 另做最小体积校验）
            if os.path.isfile(dest) and (
                    engine != "wd14"
                    or os.path.basename(fname) != "model.onnx"
                    or os.path.getsize(dest) >= MIN_ONNX_BYTES):
                continue
            with _dl_lock:
                _dl_state["file"] = fname
                _dl_state["pct"] = 0
            last_err = None
            for host in hosts:
                url = f"{host}/{m['repo']}/resolve/main/{fname}"
                try:
                    _url_download(url, dest + ".part",
                                  lambda p: _set_pct(p), token=token)
                    os.replace(dest + ".part", dest)
                    last_err = None
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (401, 403):
                        last_err = RuntimeError(
                            f"HTTP {e.code}：{m['repo']} 需要登录"
                            + ("（gated 仓库请先在官网同意协议并在配置窗口填 HF Token）"
                               if m.get("gated") else "，或仓库不存在"))
                    elif e.code == 404:
                        last_err = RuntimeError(f"HTTP 404：文件不存在（{m['repo']}/{fname}）")
                    else:
                        last_err = e
                    try:
                        os.remove(dest + ".part")
                    except OSError:
                        pass
                except Exception as e:
                    last_err = e
                    try:
                        os.remove(dest + ".part")
                    except OSError:
                        pass
            if last_err:
                raise last_err
            with _dl_lock:
                _dl_state["done_files"] += 1
        with _dl_lock:
            _dl_state.update(active=False, finished=True, file="", pct=100)
        return None
    except Exception as e:
        traceback.print_exc()
        with _dl_lock:
            _dl_state.update(active=False, finished=True,
                             error=f"{e.__class__.__name__}: {e}")
        return str(e)


def _set_pct(p):
    with _dl_lock:
        _dl_state["pct"] = p


# ---------------------------------------------------------------- WD14 推理 ---

_sessions = {}   # model_id -> (InferenceSession, tags list)
_sess_lock = threading.Lock()


def _load_wd14(model_id, cfg):
    if model_id in _sessions:
        return _sessions[model_id]
    with _sess_lock:
        if model_id in _sessions:
            return _sessions[model_id]
        m = model_entry(cfg, model_id)
        if not m:
            raise FileNotFoundError(f"未知模型：{model_id}")
        d = wd14_dir(model_id)
        onnx_path = os.path.join(d, "model.onnx")
        csv_path = os.path.join(d, "selected_tags.csv")
        if not model_installed(m):
            raise FileNotFoundError(f"模型未下载：{model_id}")

        # 先 import torch，让其捆绑的 CUDA DLL 就绪，onnxruntime 的 CUDA EP 才可用
        try:
            import torch   # noqa: F401
        except ImportError:
            pass
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        avail = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        try:
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception as e:
            print(f"[tagger] GPU 加载失败，回退 CPU: {e}")
            sess = ort.InferenceSession(onnx_path, sess_options=so,
                                        providers=["CPUExecutionProvider"])
        print(f"[tagger] {model_id} providers: {sess.get_providers()}")

        with open(csv_path, "r", encoding="utf-8") as f:
            tags = [(row["name"], int(row["category"]))
                    for row in csv.DictReader(f)]
        _sessions[model_id] = (sess, tags)
        return _sessions[model_id]


def _preprocess(pil_img, size):
    """WD14 预处理：透明底铺白 -> RGB 转 BGR -> 白边补成正方形 -> 缩放到模型输入尺寸。"""
    img = pil_img.convert("RGBA")
    bg = Image.new("RGBA", img.size, "WHITE")
    bg.paste(img, mask=img)
    arr = np.asarray(bg.convert("RGB"))[:, :, ::-1]   # BGR

    h, w = arr.shape[:2]
    desired = max(size, h, w)
    dh, dw = (desired - h) // 2, (desired - w) // 2
    arr = np.pad(arr, ((dh, desired - h - dh), (dw, desired - w - dw), (0, 0)),
                 constant_values=255)

    # 原版 smart_resize：方形图统一缩放到模型输入边长
    sq = Image.fromarray(arr[:, :, ::-1])   # 转回 RGB 供 PIL 缩放
    if desired > size:
        sq = sq.resize((size, size), Image.LANCZOS)
    arr = np.asarray(sq)[:, :, ::-1].astype(np.float32)[None]
    return arr


def _escape_tag(tag):
    return tag.replace("(", r"\(").replace(")", r"\)")


def _wd14_interrogate(model_id, pil_img, gen_t, char_t, cfg):
    """返回 (字符标签列表, 普通标签列表)，各自按置信度降序。"""
    sess, tags = _load_wd14(model_id, cfg)
    _, height, _, _ = sess.get_inputs()[0].shape
    x = _preprocess(pil_img, height)
    confs = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]

    chars, general = [], []
    for (name, cat), conf in zip(tags, confs):
        if cat == 4 and conf >= char_t:
            chars.append((name, float(conf)))
        elif cat == 0 and conf >= gen_t:
            general.append((name, float(conf)))
    chars.sort(key=lambda t: -t[1])
    general.sort(key=lambda t: -t[1])
    return [c[0] for c in chars], [g[0] for g in general]


# ---------------------------------------------------------------- CL Tagger 推理 ---

_cl_sessions = {}   # model_id -> (sess, processor, idx_to_tag, tag_to_category)


def _set_hf_endpoint(cfg):
    """按配置的下载源设置 HF 镜像（须在首次 import transformers/huggingface_hub 之前）。"""
    endpoint = MIRRORS.get(cfg.get("mirror", "mirror"), MIRRORS["mirror"])
    if endpoint == MIRRORS["official"]:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = endpoint


def _load_cl(model_id, cfg):
    """加载 CL Tagger（SigLIP2 架构）：HF AutoProcessor 预处理 + onnxruntime 推理。"""
    if model_id in _cl_sessions:
        return _cl_sessions[model_id]
    with _sess_lock:
        if model_id in _cl_sessions:
            return _cl_sessions[model_id]
        m = model_entry(cfg, model_id)
        if not m:
            raise FileNotFoundError(f"未知模型：{model_id}")
        d = wd14_dir(model_id)
        if not model_installed(m):
            raise FileNotFoundError(f"模型未下载：{model_id}")

        # 词表：v2 用 model_vocabulary.json（idx_to_tag/tag_to_category），
        # v1 用 tag_mapping.json（{索引: {tag, category}}）
        vocab_path = os.path.join(d, "model_vocabulary.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            raw_idx = vocab.get("idx_to_tag")
            if not raw_idx:   # 兼容带版本前缀的键（如 "v2_01a/idx_to_tag"）
                for k, v in vocab.items():
                    if isinstance(k, str) and k.endswith("/idx_to_tag"):
                        raw_idx = v
                        break
            if not raw_idx:
                raise ValueError(f"词表里找不到 idx_to_tag：{vocab_path}")
            idx_to_tag = {int(k): v for k, v in raw_idx.items()}
            tag_to_cat = vocab.get("tag_to_category", {})
        else:
            tm_path = os.path.join(d, "tag_mapping.json")
            with open(tm_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            idx_to_tag, tag_to_cat = {}, {}
            for k, v in mapping.items():
                tag = v.get("tag")
                if tag:
                    idx_to_tag[int(k)] = tag
                    tag_to_cat[tag] = str(v.get("category", "Unknown")).capitalize()

        # 预处理器（第一次使用会联网下载处理器配置，走配置的下载源）
        processor_repo = CL_DEFAULT_PROCESSOR
        meta_path = os.path.join(d, "model_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                processor_repo = json.load(f).get(
                    "vision_encoder_repo", CL_DEFAULT_PROCESSOR)
        _set_hf_endpoint(cfg)
        try:
            import torch   # noqa: F401  先就绪 CUDA DLL
        except ImportError:
            pass
        from transformers import AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(processor_repo,
                                                      local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(processor_repo)

        # ONNX 会话（v2 的 model.onnx 依赖同目录的 model.onnx.data）
        onnx_file = next(f for f in m["files"] if f.endswith(".onnx"))
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        avail = ort.get_available_providers()
        providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail
                     else []) + ["CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(os.path.join(d, onnx_file),
                                        sess_options=so, providers=providers)
        except Exception as e:
            print(f"[tagger] GPU 加载失败，回退 CPU: {e}")
            sess = ort.InferenceSession(os.path.join(d, onnx_file),
                                        sess_options=so,
                                        providers=["CPUExecutionProvider"])
        print(f"[tagger] {model_id} providers: {sess.get_providers()}")

        _cl_sessions[model_id] = (sess, processor, idx_to_tag, tag_to_cat)
        return _cl_sessions[model_id]


def _cl_interrogate(model_id, pil_img, gen_t, char_t, cfg):
    """CL Tagger 推理。返回 (字符标签列表, 普通标签列表)，按置信度降序。
    Character/Copyright 记为字符标签；General 记为普通标签；
    Rating/Quality/Meta/Model 不写入训练标签。"""
    sess, processor, idx_to_tag, tag_to_cat = _load_cl(model_id, cfg)
    pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > 2048:
        pil_img.thumbnail((2048, 2048))
    try:
        inputs = processor(images=pil_img, return_tensors="pt", max_num_patches=256)
    except TypeError:
        inputs = processor(images=pil_img, return_tensors="pt")

    feed = {"pixel_values": inputs["pixel_values"].float().numpy()}
    in_names = {i.name for i in sess.get_inputs()}
    if "pixel_attention_mask" in in_names:
        feed["pixel_attention_mask"] = inputs["pixel_attention_mask"].float().numpy()
    if "spatial_shapes" in in_names:
        feed["spatial_shapes"] = inputs["spatial_shapes"].numpy().astype(np.int64)

    logits = sess.run(["logits"], feed)[0][0].astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))

    chars, general = [], []
    for i, p in enumerate(probs):
        tag = idx_to_tag.get(i)
        if not tag:
            continue
        cat = tag_to_cat.get(tag, "Unknown")
        if cat in ("Character", "Copyright") and p >= char_t:
            chars.append((tag, float(p)))
        elif cat == "General" and p >= gen_t:
            general.append((tag, float(p)))
    chars.sort(key=lambda t: -t[1])
    general.sort(key=lambda t: -t[1])
    return [c[0] for c in chars], [g[0] for g in general]


# ---------------------------------------------------------------- 自然语言打标 ---

_llama_model = None
_llama_key = None
_llama_lock = threading.Lock()

DEFAULT_NL_PROMPT = (
    "Describe this image in one natural English paragraph, to be used as a caption "
    "for image generation training. Describe the subject's appearance, hair, clothing, "
    "pose, expression, any other characters and their positions, background, lighting, "
    "camera angle and art style. Describe only what is actually visible. "
    "No bullet points, no markdown, no quotes, no explanations."
)


def _load_llama(key):
    """加载 GGUF 视觉模型（key 形如 'gguf:模型名'）。"""
    global _llama_model, _llama_key
    with _llama_lock:
        if _llama_model is not None and _llama_key == key:
            return
        if _llama_model is not None:
            _llama_model = None
            import gc
            gc.collect()
        m = scan_nl_models().get(key)
        if not m:
            raise ValueError(f"未知模型: {key}")
        from llama_cpp import Llama
        print(f"[nl] loading {m['model']} ...")
        _llama_model = Llama(
            model_path=os.path.join(MODEL_DIR, m["model"]),
            mmproj_path=os.path.join(MODEL_DIR, m["mmproj"]),
            n_gpu_layers=-1,
            n_ctx=8192,
            flash_attn=True,
            verbose=False,
        )
        _llama_key = key
        print("[nl] ready")


def _nl_interrogate(key, pil_img, prompt):
    _load_llama(key)
    pil_img = pil_img.convert("RGB")
    if max(pil_img.size) > 1536:
        pil_img.thumbnail((1536, 1536))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    resp = _llama_model.create_chat_completion(
        messages=messages, max_tokens=512, temperature=0.3, top_p=0.8, top_k=20)
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text.strip('"').strip()


# ---------------------------------------------------------------- 打标任务 ---

_job = {"running": False, "cancel": False, "finished": False, "total": 0,
        "done": 0, "failed": 0, "current": "", "error": None, "errors": []}
_job_lock = threading.Lock()


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(IMG_EXTS))


def _append_prompts(prefix, suffix, body):
    """前缀/后缀追加（英文逗号分隔），去空项后与主体拼接。"""
    parts = []
    for p in (prefix, body, suffix):
        p = (p or "").strip().strip(",")
        if p:
            parts.append(p)
    return ", ".join(parts)


def _run_job(cfg, model_cfg):
    global _job
    folder = cfg["folder"]
    mode = cfg["mode"]
    model = cfg["model"]
    prefix, suffix = cfg.get("prefix", ""), cfg.get("suffix", "")
    skip = cfg.get("skip_existing", False)

    try:
        imgs = list_images(folder)
        if not imgs:
            raise ValueError("文件夹里没有找到可打标的图片")
        with _job_lock:
            _job.update(running=True, cancel=False, finished=False, total=len(imgs),
                        done=0, failed=0, current="", error=None, errors=[])

        if mode == "tag":
            # 提前加载，失败立刻报错
            if model_entry(model_cfg, model).get("engine") == "cl":
                _load_cl(model, model_cfg)
            else:
                _load_wd14(model, model_cfg)

        for path in imgs:
            if _job["cancel"]:
                break
            name = os.path.basename(path)
            with _job_lock:
                _job["current"] = name
            txt = os.path.splitext(path)[0] + ".txt"
            if skip and os.path.isfile(txt):
                with _job_lock:
                    _job["done"] += 1
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    if mode == "tag":
                        if model_entry(model_cfg, model).get("engine") == "cl":
                            chars, general = _cl_interrogate(
                                model, im, cfg["gen_t"], cfg["char_t"], model_cfg)
                        else:
                            chars, general = _wd14_interrogate(
                                model, im, cfg["gen_t"], cfg["char_t"], model_cfg)
                        body = ", ".join([_escape_tag(c) for c in chars]
                                         + [_escape_tag(g) for g in general])
                    else:
                        body = _nl_interrogate(model, im, cfg.get("nl_prompt", ""))
                text = _append_prompts(prefix, suffix, body)
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(text)
                with _job_lock:
                    _job["done"] += 1
            except Exception as e:
                traceback.print_exc()
                with _job_lock:
                    _job["failed"] += 1
                    _job["errors"].append(f"{name}: {e.__class__.__name__}: {e}")
    except Exception as e:
        traceback.print_exc()
        with _job_lock:
            _job["error"] = f"{e.__class__.__name__}: {e}"
    finally:
        with _job_lock:
            _job["running"] = False
            _job["finished"] = True
            _job["current"] = ""
        # NL 模型很占显存，跑完自动释放
        global _llama_model, _llama_key
        with _llama_lock:
            if _llama_model is not None:
                _llama_model = None
                _llama_key = None
                import gc
                gc.collect()
                print("[nl] model unloaded to free VRAM")


def start_job(cfg):
    global _job, _last_cfg
    model_cfg = load_config()
    with _job_lock:
        if _job["running"]:
            return False, "已有任务在运行中"
    folder = cfg.get("folder", "").strip()
    if not os.path.isdir(folder):
        return False, "文件夹不存在"
    mode = cfg.get("mode", "tag")
    model = cfg.get("model", "")
    if mode == "tag":
        m = model_entry(model_cfg, model)
        if not m:
            return False, "未知标签模型"
        if not model_installed(m):
            return False, "模型未下载，请先在配置窗口下载"
    else:
        if model not in scan_nl_models():
            return False, "未知自然语言模型（GGUF 未找到）"
    try:
        cfg["gen_t"] = float(cfg.get("gen_t", 0.35))
        cfg["char_t"] = float(cfg.get("char_t", 0.85))
    except (TypeError, ValueError):
        return False, "阈值必须是数字"
    _last_cfg = dict(cfg)
    threading.Thread(target=_run_job, args=(cfg, model_cfg), daemon=True).start()
    return True, "开始打标"


# ---------------------------------------------------------------- Flask ---

app = Flask(__name__)


@app.after_request
def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/ping")
def ping():
    return {"ok": True, "app": "dataset-tagger"}


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据集打标</title>
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
  .row { display:flex; gap:10px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
  .row label.lab { width:110px; flex-shrink:0; opacity:.75; font-size:14px; }
  .row input[type=text] { flex:1; min-width:200px; padding:10px 12px; border-radius:8px;
        border:1px solid var(--input-border); background:var(--input-bg);
        color:var(--text); font-size:14px; }
  .row input[type=number] { width:110px; padding:10px 12px; border-radius:8px;
        border:1px solid var(--input-border); background:var(--input-bg);
        color:var(--text); font-size:14px; }
  .row select { padding:10px 12px; border-radius:8px; border:1px solid var(--input-border);
        background:var(--input-bg); color:var(--text); font-size:14px; min-width:280px; }
  textarea { width:100%; box-sizing:border-box; min-height:110px; resize:vertical;
        background:var(--inner); color:var(--text); border:1px solid var(--input-border);
        border-radius:8px; padding:10px; font-size:13px; line-height:1.5; font-family:inherit; }
  button { padding:10px 18px; border:none; border-radius:8px; background:#0984e3;
           color:#fff; font-size:14px; cursor:pointer; }
  button:hover { background:#0b6fc0; }
  button.plain { background:var(--inner); color:var(--text);
                 border:1px solid var(--input-border); }
  button.mini { padding:5px 10px; font-size:12.5px; }
  button.mini.danger { background:#d63031; }
  button.mini.danger:hover { background:#b71540; }
  button.danger { background:#e17055; color:#fff; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  h1 { font-size:20px; margin:0 0 16px; color:var(--head); }
  h2 { font-size:16px; margin:0 0 12px; color:var(--head); }
  .tip { opacity:.6; font-size:12.5px; line-height:1.6; margin-top:6px; }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid #0984e3;
          border-top-color:transparent; border-radius:50%; animation:r .8s linear infinite;
          vertical-align:-2px; margin-right:6px; }
  @keyframes r { to { transform:rotate(360deg); } }
  #jobbox { margin-top:10px; }
  #bar, #cfgbar { height:10px; background:var(--inner); border-radius:999px; overflow:hidden;
         border:1px solid var(--border); }
  #barfill, #cfgbarfill { height:100%; width:0%; background:#27ae60; transition:width .3s; }
  .err { color:#c0392b; font-size:13px; white-space:pre-wrap; }
  body.dark .err { color:#ff7675; }
  .ok { color:#1b7a3d; font-size:13px; }
  body.dark .ok { color:#7ee2a0; }
  /* 配置窗口 */
  .modal { position:fixed; inset:0; background:rgba(0,0,0,.45); display:flex;
           align-items:center; justify-content:center; z-index:1000; }
  .modal-card { background:var(--card); border:1px solid var(--border); border-radius:12px;
           padding:20px 24px; width:min(760px, 94vw); max-height:88vh; overflow:auto;
           box-shadow:0 8px 30px rgba(0,0,0,.25); }
  table { width:100%; border-collapse:collapse; font-size:13px; margin:8px 0; }
  th,td { padding:6px 8px; text-align:left; border-bottom:1px solid var(--border);
          word-break:break-all; }
  th { color:var(--head); font-weight:600; }
  .cfgform { background:var(--inner); border:1px solid var(--border); border-radius:8px;
           padding:12px 14px; margin-top:10px; }
  .cfgform input { padding:8px 10px; border-radius:8px; border:1px solid var(--input-border);
           background:var(--input-bg); color:var(--text); font-size:13px; }
  .radio { display:inline-flex; align-items:center; gap:5px; font-size:13.5px;
           margin-right:16px; cursor:pointer; }
  .theme-toggle { position:fixed; bottom:20px; right:20px; font-size:24px; cursor:pointer;
                  background:transparent; border:none; padding:6px 10px; border-radius:6px;
                  transition:background .2s; z-index:999; }
  body.dark .theme-toggle:hover { background:#38383a; }
  body.light .theme-toggle:hover { background:#e9e9eb; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🏷️ 数据集打标</h1>

  <div class="card">
    <div class="row">
      <label class="lab">文件夹</label>
      <input type="text" id="folder" placeholder="输入或选择目标文件夹路径">
      <button class="plain" onclick="pickFolder()">浏览…</button>
      <button class="plain" onclick="checkFolder()">检查</button>
    </div>
    <div class="row" id="checkRow" style="display:none">
      <label class="lab"></label>
      <span id="checkInfo"></span>
    </div>

    <div class="row">
      <label class="lab">打标方式</label>
      <select id="mode" style="min-width:200px">
        <option value="tag" selected>标签打标（Danbooru 标签）</option>
        <option value="nl">自然语言打标（描述句）</option>
      </select>
    </div>

    <div class="row">
      <label class="lab">模型</label>
      <select id="model"></select>
      <button id="cfgBtn" class="plain" onclick="openConfig()">配置…</button>
    </div>

    <div class="row" id="threshRow">
      <label class="lab">标签阈值</label>
      <input type="number" id="gen_t" value="0.35" min="0" max="1" step="0.05">
      <label class="lab" style="width:90px; text-align:right">角色阈值</label>
      <input type="number" id="char_t" value="0.85" min="0" max="1" step="0.05">
    </div>

    <div id="nlRow" style="display:none">
      <div class="row">
        <label class="lab">提示词</label>
        <div class="tip" style="margin:0">给视觉模型的指令，告诉它怎么描述图片（可自由修改）</div>
      </div>
      <textarea id="nl_prompt" spellcheck="false"></textarea>
    </div>

    <div class="row" style="margin-top:12px">
      <label class="lab">前缀追加</label>
      <input type="text" id="prefix" placeholder="如：masterpiece, best quality（逗号分隔可写多个）">
    </div>
    <div class="row">
      <label class="lab">后缀追加</label>
      <input type="text" id="suffix" placeholder="追加到每个标签文件末尾（逗号分隔可写多个）">
    </div>

    <div class="row">
      <label class="lab"></label>
      <label style="font-size:13px; display:flex; align-items:center; gap:6px;">
        <input type="checkbox" id="skip_existing"> 已有同名 txt 时跳过
      </label>
      <button class="plain" onclick="preview()">扫描</button>
      <button id="runBtn" onclick="run()">开始打标</button>
      <button class="plain" id="cancelBtn" onclick="cancel()" style="display:none">停止</button>
      <span id="scanInfo" class="tip"></span>
    </div>

    <div class="tip">结果写入图片同名的 .txt 文件（UTF-8）。标签打标输出"角色标签在前、普通标签在后"，
    括号会被转义；评分（rating）标签不写入。</div>

    <div id="jobbox" style="display:none">
      <div id="bar"><div id="barfill"></div></div>
      <div id="jobText" class="tip" style="margin-top:8px"></div>
      <div id="jobErr" class="err"></div>
    </div>
  </div>
</div>

<!-- 模型配置窗口 -->
<div class="modal" id="cfgModal" style="display:none">
  <div class="modal-card">
    <h2>⚙️ 模型配置</h2>
    <div class="row">
      <label class="lab">下载源</label>
      <label class="radio"><input type="radio" name="dlsrc" value="mirror" checked> 镜像（hf-mirror.com）</label>
      <label class="radio"><input type="radio" name="dlsrc" value="official"> 官方（huggingface.co）</label>
    </div>
    <table>
      <thead><tr><th style="width:44px">显示</th><th>模型</th><th style="width:60px">类型</th>
        <th style="width:80px">状态</th><th style="width:150px">操作</th></tr></thead>
      <tbody id="cfgBody"></tbody>
    </table>
    <div class="tip">取消勾选"显示"可把模型从主界面下拉框中隐藏。"下载"按上方选择的下载源优先，
    另一个源自动兜底。已下载的标签模型可点"删除"移除文件（需再点一次确认）。
    CL Tagger 系列（SigLIP2 架构）下载后可自动识别；其中 v2 系列为
    gated 仓库，需先在 huggingface.co 模型页登录并同意协议，再在下方填入 HF Token
    （Token 只在本次运行内有效，不写入配置文件）。</div>

    <div class="cfgform">
      <div class="row" style="margin-bottom:8px">
        <label class="lab" style="width:80px">HF Token</label>
        <input type="password" id="hfToken" placeholder="hf_...（仅本次运行有效，不保存）" style="flex:1">
        <button class="mini" onclick="saveToken()">保存</button>
        <span id="tokenMsg" class="tip" style="margin:0"></span>
      </div>
    </div>

    <div id="cfgDl" style="display:none; margin-top:12px">
      <div id="cfgbar"><div id="cfgbarfill"></div></div>
      <div id="cfgDlText" class="tip" style="margin-top:6px"></div>
    </div>

    <div class="row" style="margin:14px 0 0; justify-content:flex-end">
      <button class="plain" onclick="closeConfig()">关闭</button>
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

const $ = id => document.getElementById(id);

// ---- 模型下拉 ----
let models = [];   // {id, label, kind, size, installed, visible, custom}

function renderModels() {
    const mode = $('mode').value;
    const sel = $('model');
    sel.innerHTML = '';
    const list = models.filter(m => m.kind === mode && m.visible);
    for (const m of list) {
        const o = document.createElement('option');
        o.value = m.id;
        o.textContent = m.label + (m.installed ? '' : '（未下载' + (m.size ? ' ' + m.size : '') + '）');
        o.disabled = !m.installed;
        sel.appendChild(o);
    }
    // 默认选中第一个可用的；正在使用的模型仍在列表中则保持选中
    const cur = list.find(m => m.id === sel.value && m.installed);
    if (!cur) {
        const firstOk = list.find(m => m.installed);
        if (firstOk) sel.value = firstOk.id;
    }
}

async function loadModels() {
    try {
        const r = await fetch('/api/models');
        const j = await r.json();
        models = j.models;
        renderModels();
        if (!$('cfgModal').style.display || $('cfgModal').style.display === 'none') return;
        renderCfgTable();
    } catch (e) {}
}

$('mode').onchange = () => {
    const nl = $('mode').value === 'nl';
    $('threshRow').style.display = nl ? 'none' : '';
    $('nlRow').style.display = nl ? '' : 'none';
    renderModels();
};

// 切换模型时自动填入该模型的推荐阈值（WD14: 0.35/0.85，CL: 见各模型定义）
function applyDefThr() {
    const m = models.find(x => x.id === $('model').value);
    if (m && m.kind === 'tag' && m.def_gen != null) {
        $('gen_t').value = m.def_gen;
        $('char_t').value = m.def_char;
    }
}
$('model').onchange = applyDefThr;

// ---- 文件夹：浏览 / 检查 ----
async function pickFolder() {
    const r = await fetch('/api/pick_folder');
    const d = await r.json();
    if (d.path) $('folder').value = d.path;
}

async function checkFolder() {
    const info = $('checkInfo');
    $('checkRow').style.display = '';
    info.innerHTML = '<span class="spin"></span>检查中…';
    try {
        const r = await fetch('/api/check', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder: $('folder').value.trim() }) });
        const d = await r.json();
        if (!d.ok) { info.innerHTML = '<span class="err">❌ ' + d.error + '</span>'; return; }
        let html = '<span class="ok">✔ 检验通过 ' + d.count + ' 张</span>';
        if (d.skipped > 0) html += '　<span class="err">✘ 跳过 ' + d.skipped + ' 张（无法解码）</span>';
        info.innerHTML = html;
    } catch (e) { info.innerHTML = '<span class="err">❌ ' + e + '</span>'; }
}

// ---- 配置窗口 ----
let cfgTimer = null;

async function openConfig() {
    $('cfgModal').style.display = 'flex';
    await loadModels();
    try {
        const r = await fetch('/api/config');
        const c = await r.json();
        document.querySelectorAll('input[name=dlsrc]').forEach(el => {
            el.checked = el.value === c.mirror;
            el.onchange = saveMirror;
        });
        $('hfToken').value = c.hf_token || '';
    } catch (e) {}
    renderCfgTable();
    clearInterval(cfgTimer);
    cfgTimer = setInterval(pollCfgDownload, 800);
    pollCfgDownload();
}

function closeConfig() {
    $('cfgModal').style.display = 'none';
    clearInterval(cfgTimer);
    loadModels();   // 关闭时刷新主界面下拉框
}

async function saveMirror() {
    const v = document.querySelector('input[name=dlsrc]:checked').value;
    await fetch('/api/config', { method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ mirror: v }) });
}

function renderCfgTable() {
    const tb = $('cfgBody');
    tb.innerHTML = '';
    for (const m of models) {
        const tr = document.createElement('tr');

        const tdv = document.createElement('td');
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = m.visible;
        cb.onchange = () => toggleVisible(m.id, cb.checked);
        tdv.appendChild(cb);

        const tdl = document.createElement('td');
        tdl.textContent = m.label;
        const kind = document.createElement('td');
        kind.textContent = m.kind === 'tag' ? '标签' : '自然语言';
        const tds = document.createElement('td');
        if (m.installed) tds.textContent = '已下载';
        else if (m.kind !== 'tag') tds.textContent = '本地';
        else tds.textContent = m.gated ? '未下载·需HF登录' : '未下载';
        tds.style.color = m.installed ? 'var(--ok, #1b7a3d)' : 'inherit';

        const tdo = document.createElement('td');
        if (m.kind === 'tag' && !m.installed) {
            const bd = document.createElement('button');
            bd.className = 'mini'; bd.textContent = '下载';
            bd.onclick = () => downloadCfg(m.id, bd);
            tdo.appendChild(bd);
        }
        if (m.kind === 'tag' && m.installed) {
            const bx = document.createElement('button');
            bx.className = 'mini danger'; bx.textContent = '删除';
            bx.style.marginLeft = '6px';
            bx.onclick = () => delModelArm(m.id, bx);
            tdo.appendChild(bx);
        }

        tr.append(tdv, tdl, kind, tds, tdo);
        tb.appendChild(tr);
    }
}

async function toggleVisible(id, visible) {
    await fetch('/api/config', { method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ hidden_delta: { id, hide: !visible } }) });
    await loadModels();
}

// 删除已下载模型：页面内二次确认（iframe 里 confirm() 会被拦截）
let delArmed = null, delTimer = null;
function delModelArm(id, btn) {
    if (delArmed !== id) {
        delArmed = id;
        btn.textContent = '⚠ 再点一次确认删除';
        clearTimeout(delTimer);
        delTimer = setTimeout(() => {
            delArmed = null;
            btn.textContent = '删除';
        }, 4000);
        return;
    }
    delArmed = null;
    clearTimeout(delTimer);
    btn.disabled = true;
    doDelModel(id, btn);
}

async function doDelModel(id, btn) {
    $('cfgDl').style.display = '';
    $('cfgDlText').innerHTML = '<span class="spin"></span>正在删除模型…';
    try {
        const r = await fetch('/api/delete_model', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ id }) });
        const d = await r.json();
        if (!d.ok) {
            $('cfgDlText').innerHTML = '<span class="err">❌ ' + d.error + '</span>';
            btn.disabled = false; btn.textContent = '删除';
            return;
        }
        $('cfgDlText').innerHTML = '已删除模型文件';
        await loadModels();
        renderCfgTable();
    } catch (e) {
        $('cfgDlText').innerHTML = '<span class="err">❌ ' + e + '</span>';
    }
    btn.disabled = false;
    setTimeout(() => { if (!$('cfgDlText').innerHTML.includes('spin')) $('cfgDl').style.display = 'none'; }, 3000);
}

async function downloadCfg(id, btn) {
    if (btn) { btn.disabled = true; }
    $('cfgDl').style.display = '';
    $('cfgDlText').innerHTML = '<span class="spin"></span>开始下载…';
    try {
        const r = await fetch('/api/download', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ id }) });
        const d = await r.json();
        if (!d.ok) $('cfgDlText').innerHTML = '<span class="err">❌ ' + d.error + '</span>';
    } catch (e) {
        $('cfgDlText').innerHTML = '<span class="err">❌ ' + e + '</span>';
    }
}

async function pollCfgDownload() {
    if ($('cfgModal').style.display === 'none') { clearInterval(cfgTimer); return; }
    try {
        const r = await fetch('/api/download_status');
        const s = await r.json();
        if (!s.active && !s.finished) { $('cfgDl').style.display = 'none'; return; }
        $('cfgDl').style.display = '';
        if (s.error) {
            $('cfgbarfill').style.width = '0%';
            $('cfgDlText').innerHTML = '<span class="err">下载失败：' + s.error + '</span>';
            loadModels();
            ackDownload();
            return;
        }
        if (s.active) {
            $('cfgbarfill').style.width = (s.pct || 0) + '%';
            $('cfgDlText').innerHTML = '<span class="spin"></span>下载中 ' + (s.file || '')
                + '　' + (s.pct || 0) + '%（' + (s.done_files || 0) + '/' + s.total_files
                + ' 个文件）';
            return;
        }
        // finished
        $('cfgbarfill').style.width = '100%';
        $('cfgDlText').innerHTML = '<span class="ok">✔ 下载完成</span>';
        loadModels();
        ackDownload();
    } catch (e) {}
}

async function ackDownload() {
    try { await fetch('/api/download_ack', { method: 'POST' }); } catch (e) {}
}

async function saveToken() {
    const msg = $('tokenMsg');
    try {
        const r = await fetch('/api/config', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ hf_token: $('hfToken').value.trim() }) });
        const d = await r.json();
        msg.innerHTML = d.ok ? '<span class="ok">✔ 已保存</span>'
                             : '<span class="err">❌ ' + d.error + '</span>';
    } catch (e) { msg.innerHTML = '<span class="err">❌ ' + e + '</span>'; }
    setTimeout(() => { msg.textContent = ''; }, 2500);
}

// ---- 扫描 / 执行 ----
function args() {
    return {
        folder: $('folder').value.trim(),
        mode: $('mode').value,
        model: $('model').value,
        gen_t: parseFloat($('gen_t').value),
        char_t: parseFloat($('char_t').value),
        nl_prompt: $('nl_prompt').value,
        prefix: $('prefix').value,
        suffix: $('suffix').value,
        skip_existing: $('skip_existing').checked
    };
}

async function preview() {
    $('scanInfo').textContent = '扫描中…';
    try {
        const r = await fetch('/api/preview', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder: $('folder').value.trim() }) });
        const d = await r.json();
        $('scanInfo').textContent = d.ok ? ('共 ' + d.count + ' 张图片') : ('❌ ' + d.error);
    } catch (e) { $('scanInfo').textContent = '❌ ' + e; }
}

async function run() {
    const btn = $('runBtn');
    btn.disabled = true;
    try {
        const r = await fetch('/api/run', { method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(args()) });
        const d = await r.json();
        if (!d.ok) { alert(d.message || '启动失败'); btn.disabled = false; }
    } catch (e) { alert('启动失败: ' + e); btn.disabled = false; }
}

async function cancel() {
    try { await fetch('/api/cancel', { method: 'POST' }); } catch (e) {}
}

// ---- 任务状态轮询 ----
async function pollJob() {
    try {
        const r = await fetch('/api/job');
        const j = await r.json();
        const show = j.running || j.finished;
        $('jobbox').style.display = show ? '' : 'none';
        $('cancelBtn').style.display = j.running ? '' : 'none';
        $('runBtn').disabled = j.running;
        if (!show) return;
        const pct = j.total ? Math.round((j.done + j.failed) * 100 / j.total) : 0;
        $('barfill').style.width = pct + '%';
        let txt = '进度 ' + (j.done + j.failed) + '/' + j.total + '（成功 ' + j.done
                + '，失败 ' + j.failed + '）';
        if (j.current) txt += ' · 当前：' + j.current;
        if (j.mode) txt += ' · ' + (j.mode === 'tag' ? '标签打标' : '自然语言打标');
        if (j.running && !j.finished) txt = '<span class="spin"></span>' + txt;
        else if (j.cancel) txt += ' · 已停止';
        else if (j.finished && !j.error && j.failed === 0) { txt += ' · 全部完成'; }
        $('jobText').innerHTML = txt;
        $('jobErr').textContent = (j.error ? ('❌ ' + j.error + '\n') : '')
                                + (j.errors && j.errors.length ? j.errors.slice(-8).join('\n') : '');
    } catch (e) {}
}

// ---- 初始化 ----
(async function init() {
    try {
        const r = await fetch('/api/nl_prompt');
        const j = await r.json();
        $('nl_prompt').value = j.prompt;
    } catch (e) {}
    await loadModels();
    pollJob();
    setInterval(pollJob, 1000);
})();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


def _all_models(cfg):
    """主界面下拉框 + 配置窗口共用的模型清单。"""
    models = []
    for m in WD14_MODELS + CL_MODELS:
        models.append({"id": m["id"], "label": m["label"], "kind": "tag",
                       "size": m["size"], "custom": False,
                       "engine": m.get("engine", "wd14"),
                       "gated": m.get("gated", False),
                       "def_gen": m.get("def_gen", 0.35),
                       "def_char": m.get("def_char", 0.85),
                       "visible": m["id"] not in cfg.get("hidden", []),
                       "installed": model_installed(m)})
    for key, m in scan_nl_models().items():
        models.append({"id": key, "label": m["label"], "kind": "nl",
                       "size": "", "custom": False,
                       "visible": key not in cfg.get("hidden", []),
                       "installed": True})
    return models


@app.get("/api/models")
def api_models():
    cfg = load_config()
    return jsonify({"ok": True, "models": _all_models(cfg)})


@app.get("/api/config")
def api_config_get():
    cfg = load_config()
    cfg["hf_token"] = _hf_token    # 只回给当前页面，方便本次会话回显
    return jsonify(cfg)


@app.post("/api/config")
def api_config_post():
    cfg = load_config()
    data = request.get_json(silent=True) or {}

    if "mirror" in data:
        if data["mirror"] not in MIRRORS:
            return jsonify({"ok": False, "error": "无效的下载源"}), 400
        cfg["mirror"] = data["mirror"]
        save_config(cfg)
        return jsonify({"ok": True})

    if "hf_token" in data:
        global _hf_token
        _hf_token = (data.get("hf_token") or "").strip()
        return jsonify({"ok": True})

    if "hidden_delta" in data:
        hid = cfg.setdefault("hidden", [])
        delta = data["hidden_delta"] or {}
        mid, hide = delta.get("id"), bool(delta.get("hide"))
        if mid:
            if hide and mid not in hid:
                hid.append(mid)
            elif not hide and mid in hid:
                hid.remove(mid)
            save_config(cfg)
        return jsonify({"ok": True})

    return jsonify({"ok": False, "error": "无效的配置操作"}), 400


@app.get("/api/nl_prompt")
def api_nl_prompt():
    return jsonify({"ok": True, "prompt": DEFAULT_NL_PROMPT})


@app.post("/api/download")
def api_download():
    cfg = load_config()
    data = request.get_json(silent=True) or {}
    model_id = data.get("id", "")
    m = model_entry(cfg, model_id)
    if not m:
        return jsonify({"ok": False, "error": "该模型不支持在线下载"}), 400
    if model_installed(m):
        return jsonify({"ok": False, "error": "模型已存在"}), 400
    with _dl_lock:
        if _dl_state["active"]:
            return jsonify({"ok": False, "error": "已有下载在进行中"}), 409
    threading.Thread(target=_download_model, args=(model_id, cfg), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/download_status")
def api_download_status():
    with _dl_lock:
        return jsonify(dict(_dl_state))


@app.post("/api/download_ack")
def api_download_ack():
    with _dl_lock:
        if not _dl_state["active"]:
            _dl_state["finished"] = False
    return jsonify({"ok": True})


@app.post("/api/delete_model")
def api_delete_model():
    """删除一个已下载的标签模型目录。"""
    cfg = load_config()
    model_id = (request.get_json(silent=True) or {}).get("id", "")
    m = model_entry(cfg, model_id)
    if not m or m.get("kind") == "nl":
        return jsonify({"ok": False, "error": "该模型不支持删除"}), 400
    with _job_lock:
        if _job["running"]:
            return jsonify({"ok": False, "error": "打标任务进行中，无法删除"}), 409
    d = wd14_dir(model_id)
    if not os.path.isdir(d):
        return jsonify({"ok": False, "error": "模型目录不存在"}), 404
    try:
        shutil.rmtree(d)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}"}), 500
    print(f"[tagger] 已删除模型目录：{d}")
    return jsonify({"ok": True})


@app.get("/api/pick_folder")
def api_pick_folder():
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="选择数据集文件夹")
        root.destroy()
        return jsonify({"ok": True, "path": os.path.normpath(path) if path else None})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.post("/api/preview")
def api_preview():
    folder = (request.get_json(silent=True) or {}).get("folder", "")
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": "文件夹不存在"})
    return jsonify({"ok": True, "count": len(list_images(folder))})


@app.post("/api/check")
def api_check():
    """检查文件夹里的图片能否正常解码（能否用于打标）。"""
    folder = (request.get_json(silent=True) or {}).get("folder", "")
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": "文件夹不存在"})
    passed, skipped = 0, 0
    for path in list_images(folder):
        try:
            with Image.open(path) as im:
                im.load()
                im.convert("RGB")
            passed += 1
        except Exception:
            skipped += 1
    return jsonify({"ok": True, "count": passed, "skipped": skipped})


@app.post("/api/run")
def api_run():
    cfg = request.get_json(silent=True) or {}
    ok, msg = start_job(cfg)
    return jsonify({"ok": ok, "message": msg}), (200 if ok else 400)


@app.get("/api/job")
def api_job():
    with _job_lock:
        j = dict(_job)
    j["mode"] = _last_cfg.get("mode", "")
    return jsonify(j)


@app.post("/api/cancel")
def api_cancel():
    with _job_lock:
        if _job["running"]:
            _job["cancel"] = True
            return jsonify({"ok": True, "message": "将在当前图片完成后停止"})
    return jsonify({"ok": False, "message": "没有正在运行的任务"})


_last_cfg = {}

if __name__ == "__main__":
    os.makedirs(TAGGER_DIR, exist_ok=True)
    cfg = load_config()
    print(f"[app] model dir: {MODEL_DIR}")
    print(f"[app] tagger dir: {TAGGER_DIR}")
    print(f"[app] download source: {MIRRORS.get(cfg.get('mirror'))}")
    for m in _all_models(cfg):
        mark = "√" if m["installed"] else "×"
        vis = "" if m["visible"] else "（已隐藏）"
        print(f"[app]   [{mark}] {m['id']} {vis}")
    print(f"[app] serving at http://127.0.0.1:{PORT}")

    def _open_browser():
        if os.environ.get("TOOL_NO_BROWSER") == "1":
            return
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
