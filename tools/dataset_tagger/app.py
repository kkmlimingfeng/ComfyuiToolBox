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
- 效果对比页：上传单张图片，勾选多个已下载模型（标签 + 自然语言可混选）同时打标，
  共用标签阈值 / 角色阈值 / 上下文长度 / 图片分辨率，结果显示在页面上
端口：7869
"""

import base64
import csv
import io
import json
import os
import shutil
import subprocess
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
# remark 是配置窗口"备注"列的说明文字；size_gb 为预计下载大小（GB，约数）
WD14_MODELS = [
    {"id": "wd-eva02-large-tagger-v3",
     "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
     "label": "WD EVA02-Large v3", "remark": "动漫，推荐", "size_gb": 1.2},
    {"id": "wd-vit-large-tagger-v3",
     "repo": "SmilingWolf/wd-vit-large-tagger-v3",
     "label": "WD ViT-Large v3", "remark": "动漫，大模型精度更高", "size_gb": 1.2},
    {"id": "wd-convnext-tagger-v3",
     "repo": "SmilingWolf/wd-convnext-tagger-v3",
     "label": "WD ConvNeXt v3", "remark": "动漫，轻量", "size_gb": 0.33},
    {"id": "wd-swinv2-tagger-v3",
     "repo": "SmilingWolf/wd-swinv2-tagger-v3",
     "label": "WD SwinV2 v3", "remark": "动漫，速度较快", "size_gb": 0.34},
    {"id": "wd-vit-tagger-v3",
     "repo": "SmilingWolf/wd-vit-tagger-v3",
     "label": "WD ViT v3", "remark": "动漫，体积最小", "size_gb": 0.35},
    # 以下 v2 系列没有出 v3 版本，继续保留
    {"id": "wd-v1-4-moat-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-moat-tagger-v2",
     "label": "WD MOAT v2", "remark": "动漫，无 v3 版", "size_gb": 0.32},
    {"id": "wd-v1-4-convnextv2-tagger-v2",
     "repo": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
     "label": "WD ConvNeXtV2 v2", "remark": "动漫，无 v3 版", "size_gb": 0.38},
]
DEFAULT_FILES = ["model.onnx", "selected_tags.csv"]

# 内置标签打标模型：CL Tagger 系列（cella110n，SigLIP2 架构）。
# v1 用 tag_mapping.json，v2 用 model_vocabulary.json 且为 gated 仓库（下载需 HF Token）。
# 文件路径相对模型目录（v1/v2 的文件在仓库子目录里），推理走 HF AutoProcessor（NaFlex 动态分辨率）。
CL_MODELS = [
    {"id": "cl_tagger_1_02", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_02/model.onnx", "cl_tagger_1_02/tag_mapping.json"],
     "label": "CL Tagger v1.02", "remark": "SigLIP2，4.2万标签，推荐阈值 0.9",
     "size_gb": 1.4, "def_gen": 0.9, "def_char": 0.9},
    {"id": "cl_tagger_1_01", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_01/model.onnx", "cl_tagger_1_01/tag_mapping.json"],
     "label": "CL Tagger v1.01", "remark": "SigLIP2，4.2万标签",
     "size_gb": 1.4, "def_gen": 0.6, "def_char": 0.85},
    {"id": "cl_tagger_1_00", "engine": "cl",
     "repo": "cella110n/cl_tagger",
     "files": ["cl_tagger_1_00/model.onnx", "cl_tagger_1_00/tag_mapping.json"],
     "label": "CL Tagger v1.00", "remark": "SigLIP2，4.2万标签",
     "size_gb": 1.4, "def_gen": 0.6, "def_char": 0.85},
    {"id": "cl_tagger_v2_01a", "engine": "cl", "gated": True,
     "repo": "cella110n/cl_tagger_v2",
     "files": ["v2_01a/model.onnx", "v2_01a/model.onnx.data",
               "v2_01a/model_vocabulary.json", "v2_01a/model_metadata.json"],
     "label": "CL Tagger v2.01a", "remark": "SigLIP2 最新，需 HF Token",
     "size_gb": 2.1, "def_gen": 0.55, "def_char": 0.85},
    {"id": "cl_tagger_v2_00", "engine": "cl", "gated": True,
     "repo": "cella110n/cl_tagger_v2",
     "files": ["v2_00/model.onnx", "v2_00/model.onnx.data",
               "v2_00/model_vocabulary.json", "v2_00/model_metadata.json"],
     "label": "CL Tagger v2.00", "remark": "SigLIP2，需 HF Token",
     "size_gb": 2.1, "def_gen": 0.55, "def_char": 0.85},
]

CL_DEFAULT_PROCESSOR = "google/siglip2-so400m-patch16-naflex"

MIRRORS = {"mirror": "https://hf-mirror.com", "official": "https://huggingface.co"}

# 内置自然语言打标模型：JoyCaption Beta One（Llama 3.1 8B 视觉，擅长详细图像描述）。
# 主模型与 mmproj 分属两个仓库；mmproj 下载后重命名，保证与主模型文件名前缀配对
# （scan_nl_models 按"主模型文件名前缀 + mmproj"自动配对）。
JOYCAPTION_MMPROJ_LOCAL = "llama-joycaption-beta-one-hf-llava.Q4_K_M.mmproj-f16.gguf"
BUILTIN_NL_MODELS = [
    {"id": "joycaption-beta-one", "kind": "nl",
     "repo": "mradermacher/llama-joycaption-beta-one-hf-llava-GGUF",
     "file": "llama-joycaption-beta-one-hf-llava.Q4_K_M.gguf",
     "mmproj_repo": "concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
     "mmproj_file": "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
     "label": "JoyCaption Beta One", "remark": "自然语言，详细图像描述，8B",
     "size_gb": 5.4},
]
# JoyCaption 官方推荐提示词（勾选该模型时自动填入，仍可手动修改）
JOYCAPTION_PROMPT = "Write a long detailed caption for this image."

# JoyCaption GGUF 内嵌的 chat template 不支持多模态 content 列表（content + 字符串
# 直接拼接会抛 TypeError），这里改用等价的 Llama3 模板：content 为列表时把
# image_url 项渲染成 <image> 占位（llama_cpp 的 mtmd 会把它替换为媒体标记），
# 文本项原样拼接。eos_token 由 llama_cpp 渲染时注入，保证与停止符一致。
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


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg.setdefault("mirror", "mirror")          # mirror | official
    cfg.setdefault("custom_models", [])         # [{id, repo, label, size, files}]
    cfg.setdefault("hidden", [])                # 下拉框隐藏的模型 id
    cfg.setdefault("thresholds", {})            # 每个标签模型的默认阈值 {id: {gen, char}}
    cfg.setdefault("tr_color", "#00b894")       # 翻译文字颜色
    cfg.setdefault("tr_size", 13)               # 翻译文字字号(px)
    if not (isinstance(cfg["tr_color"], str) and len(cfg["tr_color"]) == 7
            and cfg["tr_color"].startswith("#")):
        cfg["tr_color"] = "#00b894"
    try:
        cfg["tr_size"] = min(40, max(9, int(cfg["tr_size"])))
    except (TypeError, ValueError):
        cfg["tr_size"] = 13
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


def builtin_nl_entry(model_id):
    """按 id（内置 id 或扫描出的 gguf: 主文件名）查内置自然语言模型条目。"""
    for b in BUILTIN_NL_MODELS:
        if model_id in (b["id"], "gguf:" + os.path.splitext(b["file"])[0]):
            return b
    return None


def builtin_nl_installed(m):
    return (os.path.isfile(os.path.join(MODEL_DIR, m["file"]))
            and os.path.isfile(os.path.join(MODEL_DIR, JOYCAPTION_MMPROJ_LOCAL)))


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
            "label": stem,
            "model": f,
            "mmproj": mmprojs[best],
            "size_gb": round(size, 1),
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


def _download_nl(m, cfg):
    """下载一个内置自然语言模型（主模型 + mmproj，两个仓库），存到 model/ 根目录。"""
    hosts = [MIRRORS.get(cfg.get("mirror", "mirror"), MIRRORS["mirror"])] + \
            [h for k, h in MIRRORS.items() if k != cfg.get("mirror", "mirror")]
    jobs = [(m["repo"], m["file"], m["file"]),
            (m["mmproj_repo"], m["mmproj_file"], JOYCAPTION_MMPROJ_LOCAL)]
    with _dl_lock:
        _dl_state.update(active=True, id=m["id"], file="", pct=0, done_files=0,
                         total_files=len(jobs), error=None, finished=False)
    try:
        for repo, fname, local in jobs:
            dest = os.path.join(MODEL_DIR, local)
            if os.path.isfile(dest):
                with _dl_lock:
                    _dl_state["done_files"] += 1
                continue
            with _dl_lock:
                _dl_state["file"] = fname
                _dl_state["pct"] = 0
            last_err = None
            for host in hosts:
                url = f"{host}/{repo}/resolve/main/{fname}"
                try:
                    _url_download(url, dest + ".part",
                                  lambda p: _set_pct(p), token=_hf_token)
                    os.replace(dest + ".part", dest)
                    last_err = None
                    break
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
        # GPU 优先：CUDA → DirectML（Windows 免装 CUDA/cuDNN）→ CPU
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
                     if p in avail] + ["CPUExecutionProvider"]
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
    """返回 (字符标签列表, 普通标签列表)，元素为 (标签, 置信度)，各自按置信度降序。"""
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
    return chars, general


# ---------------------------------------------------------------- CL Tagger 推理 ---

_cl_sessions = {}   # model_id -> (sess, processor, idx_to_tag, tag_to_category)


def _set_hf_endpoint(cfg):
    """按配置的下载源设置 HF 镜像（须在首次 import transformers/huggingface_hub 之前）。"""
    endpoint = MIRRORS.get(cfg.get("mirror", "mirror"), MIRRORS["mirror"])
    if endpoint == MIRRORS["official"]:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ["HF_ENDPOINT"] = endpoint


def _ensure_processor(repo, cfg):
    """把预处理器的小配置文件下载到本地目录（镜像优先、官方兜底），
    AutoProcessor 直接从本地加载，推理时不再依赖 HF 网络栈。"""
    name = repo.split("/")[-1]
    d = os.path.join(TAGGER_DIR, "_hf_processor", name)
    os.makedirs(d, exist_ok=True)
    files = ["preprocessor_config.json", "tokenizer_config.json",
             "tokenizer.json", "special_tokens_map.json", "config.json"]
    hosts = [MIRRORS.get(cfg.get("mirror", "mirror"), MIRRORS["mirror"])] + \
            [h for k, h in MIRRORS.items() if k != cfg.get("mirror", "mirror")]
    for f in files:
        p = os.path.join(d, f)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            continue
        last = None
        for host in hosts:
            url = f"{host}/{repo}/resolve/main/{f}"
            try:
                _url_download(url, p + ".part", lambda _: None, token=_hf_token)
                os.replace(p + ".part", p)
                last = None
                break
            except Exception as e:
                last = e
                try:
                    os.remove(p + ".part")
                except OSError:
                    pass
        if last is not None:
            raise RuntimeError(f"下载预处理器配置失败（{repo}/{f}）：{last}")
    return d


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
        # 文件可能带仓库子目录（如 cl_tagger_1_02/tag_mapping.json），以 onnx 同目录为准
        onnx_rel = next(f for f in m["files"] if f.endswith(".onnx"))
        sub = os.path.dirname(onnx_rel)
        vocab_path = os.path.join(d, sub, "model_vocabulary.json")
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
            tm_path = os.path.join(d, sub, "tag_mapping.json")
            with open(tm_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            idx_to_tag, tag_to_cat = {}, {}
            for k, v in mapping.items():
                tag = v.get("tag")
                if tag:
                    idx_to_tag[int(k)] = tag
                    tag_to_cat[tag] = str(v.get("category", "Unknown")).capitalize()

        # 预处理器（仅 v2 需要：v1 的 onnx 内嵌固定位置编码，走本地固定尺寸预处理）
        processor = None
        meta_path = os.path.join(d, sub, "model_metadata.json")
        fixed_size = 448
        if os.path.isfile(meta_path):   # v2：动态分辨率（NaFlex），需要 AutoProcessor
            fixed_size = None
            with open(meta_path, "r", encoding="utf-8") as f:
                processor_repo = json.load(f).get(
                    "vision_encoder_repo", CL_DEFAULT_PROCESSOR)
            _set_hf_endpoint(cfg)
            try:
                import torch   # noqa: F401  先就绪 CUDA DLL
            except ImportError:
                pass
            proc_dir = _ensure_processor(processor_repo, cfg)
            from transformers import AutoProcessor
            try:
                processor = AutoProcessor.from_pretrained(proc_dir,
                                                          local_files_only=True)
            except Exception:
                processor = AutoProcessor.from_pretrained(processor_repo)

        # ONNX 会话（v2 的 model.onnx 依赖同目录的 model.onnx.data）
        onnx_file = next(f for f in m["files"] if f.endswith(".onnx"))
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        avail = ort.get_available_providers()
        # GPU 优先：CUDA → DirectML（Windows 免装 CUDA/cuDNN）→ CPU
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider")
                     if p in avail] + ["CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(os.path.join(d, onnx_file),
                                        sess_options=so, providers=providers)
        except Exception as e:
            print(f"[tagger] GPU 加载失败，回退 CPU: {e}")
            sess = ort.InferenceSession(os.path.join(d, onnx_file),
                                        sess_options=so,
                                        providers=["CPUExecutionProvider"])
        print(f"[tagger] {model_id} providers: {sess.get_providers()}")

        _cl_sessions[model_id] = (sess, processor, idx_to_tag, tag_to_cat, fixed_size)
        return _cl_sessions[model_id]


def _cl_interrogate(model_id, pil_img, gen_t, char_t, cfg):
    """CL Tagger 推理。返回 (字符标签列表, 普通标签列表)，按置信度降序。
    Character/Copyright 记为字符标签；General 记为普通标签；
    Rating/Quality/Meta/Model 不写入训练标签。"""
    sess, processor, idx_to_tag, tag_to_cat, fixed_size = _load_cl(model_id, cfg)
    pil_img = pil_img.convert("RGB")
    if fixed_size:
        # v1 系列：onnx 内嵌固定位置编码（1025 tokens），输入必须固定 448×448
        x = pil_img.resize((fixed_size, fixed_size), Image.BICUBIC)
        x = np.asarray(x, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        feed = {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]}
        out_name = sess.get_outputs()[0].name
    else:
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
        out_name = "logits"

    logits = sess.run([out_name], feed)[0][0].astype(np.float64)
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
    return chars, general


# ---------------------------------------------------------------- 自然语言打标 ---

class _CompareStopped(Exception):
    """对比/打标被用户停止（推理中途打断）"""


def _unload_llama_now():
    """立刻卸载当前 llama.cpp 模型并释放显存。"""
    global _llama_model, _llama_key
    with _llama_lock:
        if _llama_model is not None:
            _llama_model = None
            _llama_key = None
            import gc
            gc.collect()
            print("[nl] model unloaded to free VRAM")


def _drop_tagger(model_id):
    """卸载单个标签模型（WD14 / CL）的 onnx session，立刻释放显存/内存。"""
    dropped = False
    with _sess_lock:
        if model_id in _sessions:
            _sessions.pop(model_id, None)
            dropped = True
        if model_id in _cl_sessions:
            _cl_sessions.pop(model_id, None)
            dropped = True
    if dropped:
        import gc
        gc.collect()
        print(f"[tagger] unloaded {model_id} session to free VRAM")


def _unload_taggers():
    """卸载全部标签模型（WD14 / CL 的 onnxruntime session），立刻释放显存/内存。
    只应在批量任务或对比结束后调用（任务开始前必须已结束，无并发使用）。"""
    n = 0
    with _sess_lock:
        n = len(_sessions) + len(_cl_sessions)
        _sessions.clear()
        _cl_sessions.clear()
    if n:
        import gc
        gc.collect()
        print(f"[tagger] unloaded {n} onnx session(s) to free VRAM")


_llama_model = None
_llama_key = None          # (模型key, n_ctx) 二元组，切换上下文长度也会重载
_llama_lock = threading.Lock()

DEFAULT_NL_PROMPT = (
    "Describe this image in one natural English paragraph, to be used as a caption "
    "for image generation training. Describe the subject's appearance, hair, clothing, "
    "pose, expression, any other characters and their positions, background, lighting, "
    "camera angle and art style. Describe only what is actually visible. "
    "No bullet points, no markdown, no quotes, no explanations."
)


def _free_vram_mb():
    """查询 GPU 空闲显存（MB）。查不到（无 nvidia-smi / 无 GPU）返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:
        return None


def _try_free_sibling_vram():
    """显存不够时，通知同套工具里占用显存的服务（反推提示词 7867）卸载模型。
    同一台机器上两个进程无法同时各装一份 27B 模型，先释放再加载。"""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:7867/api/unload", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status == 200:
                print("[nl] 已通知「自然语言提示词反推」卸载模型以释放显存")
                time.sleep(2.0)
                return True
    except Exception:
        pass
    return False


def _load_llama(key, n_ctx=8192):
    """加载 GGUF 视觉模型（key 形如 'gguf:模型名'），缓存键含上下文长度。
    加载前检查空闲显存：不够时先自动释放反推提示词占用的模型，再加载。"""
    global _llama_model, _llama_key
    with _llama_lock:
        if _llama_model is not None and _llama_key == (key, n_ctx):
            return
        if _llama_model is not None:
            _llama_model = None
            import gc
            gc.collect()
        m = scan_nl_models().get(key)
        if not m:
            raise ValueError(f"未知模型: {key}")

        # 加载前检查空闲显存：不够时先自动释放反推提示词占用的模型。
        # 显存仍不够时 NVIDIA 驱动会触发系统内存回退（模型"看似"进了 GPU，
        # 实际一半数据在内存里反复搬运，推理慢 10 倍以上），这里只警告不阻断。
        need_mb = 0
        for f in (m["model"], m["mmproj"]):
            p = os.path.join(MODEL_DIR, f)
            if os.path.isfile(p):
                need_mb += os.path.getsize(p) / 1024 / 1024
        kv_mb = int(n_ctx) * 0.16          # KV 缓存粗估（8k 上下文约 1.3 GB）
        need_mb += kv_mb + 1024            # + 1 GB 安全余量
        free_mb = _free_vram_mb()
        if free_mb is not None and free_mb < need_mb:
            _try_free_sibling_vram()
            free_mb = _free_vram_mb()
        if free_mb is not None and free_mb < need_mb:
            print(f"[nl] 警告：显存不足（约需 {need_mb / 1024:.1f} GB，当前空闲 "
                  f"{free_mb / 1024:.1f} GB），可能触发系统内存回退导致推理明显变慢")

        import ctypes
        from llama_cpp import Llama
        from llama_cpp import llama_cpp as _lc

        lines = []

        @_lc.ggml_log_callback
        def _cb(level, msg, user):
            try:
                lines.append(msg.decode("utf-8", "ignore"))
            except Exception:
                pass

        old_cb = _lc.ggml_log_callback()
        old_ud = ctypes.c_void_p()
        try:
            _lc.llama_log_get(ctypes.byref(old_cb), ctypes.byref(old_ud))
        except Exception:
            pass
        _lc.llama_log_set(_cb, None)
        try:
            print(f"[nl] loading {m['model']} (n_ctx={n_ctx}) ...")
            _llama_model = Llama(
                model_path=os.path.join(MODEL_DIR, m["model"]),
                mmproj_path=os.path.join(MODEL_DIR, m["mmproj"]),
                n_gpu_layers=-1,
                n_ctx=n_ctx,
                flash_attn=True,
                verbose=False,
            )
        finally:
            _lc.llama_log_set(old_cb, old_ud)
        _llama_key = (key, n_ctx)
        # JoyCaption 内嵌模板不兼容多模态 content 列表，替换为等价的 Llama3 模板：
        # 改模板字符串 + 重新编译 jinja（构造时已预编译），并让解析缓存失效
        if "joycaption" in m["model"].lower():
            h = _llama_model.chat_handler
            h.chat_format = JOYCAPTION_CHAT_TEMPLATE
            h._change_chat_template(JOYCAPTION_CHAT_TEMPLATE)
            h._template_initialized = False
        # 解析卸载层数，仅记录/警告，不阻断推理
        off, total = None, None
        for line in lines:
            if "offloaded" in line and "layers" in line:
                try:
                    seg = line.split("offloaded")[1].split("layers")[0]
                    off, total = (int(x.strip()) for x in seg.split("/"))
                except Exception:
                    off, total = None, None
                break
        if total and off is not None and off < total:
            print(f"[nl] 警告：模型仅 {off}/{total} 层加载到 GPU，剩余层用 CPU/内存"
                  f"回退推理，速度会明显变慢")
        if off is not None:
            print(f"[nl] ready · GPU（已卸载 {off}/{total} 层）")
        else:
            print("[nl] ready")


def _fit_image_to_ctx(pil_img, n_ctx, prompt, max_tokens=512):
    """按上下文容量自动缩小图片。
    Qwen3-VL 每 28x28 像素约编码为 1 个视觉 token，图片 token 超过 n_ctx 时
    llama.cpp 会报 Llama.eval(decode) 失败，这里按预算预先缩图。"""
    w, h = pil_img.size
    est = ((h + 27) // 28) * ((w + 27) // 28)
    prompt_est = max(64, int(len(prompt) / 3))
    budget = n_ctx - prompt_est - max_tokens - 128   # 扣掉提示词/生成长度/安全余量
    if budget < 256:
        budget = 256
    if est <= budget:
        return pil_img
    scale = (budget / est) ** 0.5
    nw = max(28, int(w * scale) // 28 * 28)
    nh = max(28, int(h * scale) // 28 * 28)
    print(f"[nl] image {w}x{h} (~{est} visual tokens) exceeds n_ctx={n_ctx}, "
          f"shrunk to {nw}x{nh}")
    return pil_img.resize((nw, nh))


def _nl_interrogate(key, pil_img, prompt, max_side=1536, n_ctx=8192,
                    max_tokens=512, should_cancel=None):
    _load_llama(key, n_ctx)
    pil_img = pil_img.convert("RGB")
    try:
        max_side = int(max_side)
    except (TypeError, ValueError):
        max_side = 1536
    if max_side > 0 and max(pil_img.size) > max_side:
        pil_img.thumbnail((max_side, max_side))
    pil_img = _fit_image_to_ctx(pil_img, n_ctx, prompt, max_tokens)
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
    kw = dict(messages=messages, max_tokens=max_tokens, temperature=0.3,
              top_p=0.8, top_k=20)
    try:
        if should_cancel is None:
            resp = _llama_model.create_chat_completion(**kw)
            text = (resp["choices"][0]["message"].get("content") or "").strip()
        else:
            # 流式逐 token 接收，用户点停止时立刻中断生成（close 内部即终止解码）
            stream = _llama_model.create_chat_completion(**kw, stream=True)
            parts = []
            try:
                for chunk in stream:
                    if should_cancel():
                        raise _CompareStopped()
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
            finally:
                stream.close()
            text = "".join(parts).strip()
    except _CompareStopped:
        raise
    except RuntimeError as e:
        raise RuntimeError(
            f"推理失败（上下文 {n_ctx} 不足或显存溢出，请调大上下文长度、"
            f"降低图片分辨率或换更小模型）：{e}") from e
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
                        body = ", ".join([_escape_tag(t) for t, _ in chars]
                                         + [_escape_tag(t) for t, _ in general])
                    else:
                        body = _nl_interrogate(
                            model, im, cfg.get("nl_prompt", ""),
                            max_side=cfg.get("nl_res", 1536),
                            n_ctx=cfg.get("n_ctx", 8192),
                            max_tokens=cfg.get("maxt", 512),
                            should_cancel=lambda: _job["cancel"])
                text = _append_prompts(prefix, suffix, body)
                with open(txt, "w", encoding="utf-8") as f:
                    f.write(text)
                with _job_lock:
                    _job["done"] += 1
            except _CompareStopped:
                # 用户中途停止：本张作废，跳出循环
                break
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
        # 模型很占显存，跑完立刻全部释放（NL + 标签模型）
        _unload_llama_now()
        _unload_taggers()


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
        cfg["n_ctx"] = int(cfg.get("n_ctx", 8192))
        cfg["nl_res"] = int(cfg.get("nl_res", 1536))
        cfg["maxt"] = min(4096, max(32, int(cfg.get("maxt", 512))))
    except (TypeError, ValueError):
        return False, "阈值或参数必须是数字"
    _last_cfg = dict(cfg)
    threading.Thread(target=_run_job, args=(cfg, model_cfg), daemon=True).start()
    return True, "开始打标"


# ---------------------------------------------------------------- 效果对比 ---

_cmp = {"running": False, "cancel": False, "finished": False, "total": 0,
        "done": 0, "current": "", "error": None, "results": []}
_cmp_lock = threading.Lock()
_cmp_img = None      # 对比用图片在服务端的临时路径


def _run_compare(models, params):
    """依次用每个模型对单张图片打标，结果逐个写入 _cmp['results']。
    标签模型（onnx）先跑、自然语言模型（llama.cpp）后跑，减少显存争用。"""
    global _cmp, _llama_model, _llama_key
    im = None
    model_cfg = load_config()
    try:
        with _cmp_lock:
            _cmp.update(running=True, cancel=False, finished=False,
                        total=len(models), done=0, current="", error=None,
                        results=[])
        im = Image.open(_cmp_img)
        im.load()
        ordered = ([m for m in models if m["kind"] == "tag"]
                   + [m for m in models if m["kind"] != "tag"])
        for m in ordered:
            if _cmp["cancel"]:
                break
            with _cmp_lock:
                _cmp["current"] = m["label"]
            res = {"id": m["id"], "label": m["label"], "kind": m["kind"],
                   "ok": False, "output": "", "sec": 0, "error": None}
            t0 = time.time()
            try:
                if m["kind"] == "tag":
                    if model_entry(model_cfg, m["id"]).get("engine") == "cl":
                        chars, general = _cl_interrogate(
                            m["id"], im, params["gen_t"], params["char_t"],
                            model_cfg)
                    else:
                        chars, general = _wd14_interrogate(
                            m["id"], im, params["gen_t"], params["char_t"],
                            model_cfg)
                    items = chars + general
                    res["output"] = ", ".join(
                        [_escape_tag(t) for t, _ in items])
                    # 附带置信度（3 位小数），对比页在每个标签旁展示
                    res["detail"] = [[t, round(c, 3)] for t, c in items]
                else:
                    res["output"] = _nl_interrogate(
                        m["id"], im, params["nl_prompt"],
                        max_side=params["res"], n_ctx=params["n_ctx"],
                        max_tokens=params["maxt"],
                        should_cancel=lambda: _cmp["cancel"])
                res["ok"] = True
            except _CompareStopped:
                # 用户停止：当前模型作废，整体结束
                res["error"] = "已停止"
                res["sec"] = round(time.time() - t0, 1)
                with _cmp_lock:
                    _cmp["results"].append(res)
                break
            except Exception as e:
                traceback.print_exc()
                res["error"] = f"{e.__class__.__name__}: {e}"
            res["sec"] = round(time.time() - t0, 1)
            with _cmp_lock:
                _cmp["results"].append(res)
                _cmp["done"] += 1
            # 每推理完一个模型立刻卸载它的显存，再跑下一个
            if m["kind"] == "tag":
                _drop_tagger(m["id"])
            else:
                _unload_llama_now()
    except Exception as e:
        traceback.print_exc()
        with _cmp_lock:
            _cmp["error"] = f"{e.__class__.__name__}: {e}"
    finally:
        try:
            if im is not None:
                im.close()
        except Exception:
            pass
        with _cmp_lock:
            _cmp["running"] = False
            _cmp["finished"] = True
            _cmp["current"] = ""
        # 模型很占显存，对比完立刻全部释放（NL + 标签模型）
        _unload_llama_now()
        _unload_taggers()


def start_compare(models, params):
    """校验并启动一次对比。models=[{id, kind}]，params 为共用参数。"""
    cfg = load_config()
    nl_models = scan_nl_models()
    picked = []
    for m in models:
        mid, kind = m.get("id", ""), m.get("kind", "")
        if kind == "tag":
            e = model_entry(cfg, mid)
            if not e or not model_installed(e):
                return False, f"模型未下载：{mid}"
            picked.append({"id": mid, "kind": "tag", "label": e["label"]})
        elif kind == "nl":
            if mid not in nl_models:
                return False, f"未知自然语言模型：{mid}"
            picked.append({"id": mid, "kind": "nl", "label": nl_models[mid]["label"]})
        else:
            return False, "未知模型类型"
    try:
        params = {"gen_t": float(params.get("gen_t", 0.35)),
                  "char_t": float(params.get("char_t", 0.85)),
                  "n_ctx": int(params.get("n_ctx", 8192)),
                  "res": int(params.get("res", 1536)),
                  "maxt": min(4096, max(32, int(params.get("maxt", 512)))),
                  "nl_prompt": (params.get("nl_prompt") or "").strip()
                               or DEFAULT_NL_PROMPT}
    except (TypeError, ValueError):
        return False, "参数必须是数字"
    if params["res"] < 0 or params["n_ctx"] < 512:
        return False, "分辨率/上下文参数不合法"
    threading.Thread(target=_run_compare, args=(picked, params), daemon=True).start()
    return True, "开始对比"


# ---------------------------------------------------------------- Flask ---

app = Flask(__name__)


@app.after_request
def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- 结果翻译 ---

# 复用统一启动器 model/ 目录下的腾讯 Hy-MT2-1.8B 离线翻译模型（与 editor / lora_inspect 共用）
try:
    from llama_cpp import Llama as _MtLlama
    _MT_OK = True
except Exception:
    _MT_OK = False

_MT_MODEL_NAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
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
_mt_lock = threading.Lock()   # llama.cpp 不支持并发推理，必须串行化


def _mt_model_path():
    return os.path.join(MODEL_DIR, _MT_MODEL_NAME)


@app.post("/api/translate")
def api_translate():
    """按需加载翻译模型 → 逐条翻译 → 立刻卸载释放显存。"""
    data = request.get_json(silent=True) or {}
    texts = [t.strip() for t in (data.get("texts") or [])
             if isinstance(t, str) and t.strip()]
    lang = data.get("lang") or "Simplified Chinese"
    if lang not in [v for _, v in _MT_LANGS]:
        lang = "Simplified Chinese"
    if not texts:
        return jsonify({"ok": False, "error": "没有要翻译的内容"}), 400
    if not _MT_OK:
        return jsonify({"ok": False, "error": "未安装 llama-cpp-python，请先安装"}), 400
    mp = _mt_model_path()
    if not os.path.isfile(mp):
        return jsonify({"ok": False,
                        "error": f"缺少翻译模型文件：{_MT_MODEL_NAME}"
                                 "（可先在 editor 或 Lora参数分析 里下载，模型在统一启动器 model/ 目录共用）"}), 400
    out = []
    with _mt_lock:
        llm = None
        try:
            llm = _MtLlama(model_path=mp, n_gpu_layers=-1, n_ctx=2048,
                           verbose=False)
            for t in texts:
                prompt = (f"Translate the following segment into {lang}, "
                          f"without additional explanation.\n\n{t}")
                resp = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512, temperature=0.0)
                out.append(resp["choices"][0]["message"]["content"].strip())
        except Exception as e:
            traceback.print_exc()
            return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}",
                            "translations": out}), 500
        finally:
            if llm is not None:
                del llm   # 翻译完立刻卸载释放显存
                import gc
                gc.collect()
    return jsonify({"ok": True, "translations": out})


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
           padding:20px 24px; width:min(1080px, 96vw); max-height:88vh; overflow:auto;
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
  /* 页签与效果对比 */
  .tabs { display:flex; gap:8px; margin-bottom:14px; }
  .tab { background:var(--inner); color:var(--text); border:1px solid var(--input-border);
         padding:9px 22px; font-size:14px; }
  .tab.active { background:#0984e3; color:#fff; border-color:#0984e3; }
  .tab.active:hover { background:#0b6fc0; }
  .modelgrid { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 12px 120px; }
  .mchip { display:flex; align-items:center; gap:6px; padding:6px 12px;
           border:1px solid var(--input-border); border-radius:999px; font-size:13px;
           cursor:pointer; background:var(--inner); user-select:none; }
  .mchip input { margin:0; cursor:pointer; }
  .mchip.dis { opacity:.45; }
  .badge { font-size:11px; padding:1px 7px; border-radius:4px; color:#fff; flex-shrink:0; }
  .badge.tag { background:#0984e3; }
  .badge.nl { background:#8e44ad; }
  .rcard { border:1px solid var(--border); border-radius:10px; padding:12px 14px;
           margin-top:10px; background:var(--inner); }
  .rcard .rhead { display:flex; align-items:center; gap:8px; margin-bottom:8px;
                  font-size:13.5px; }
  .rcard .rsec { opacity:.6; font-size:12px; margin-left:auto; flex-shrink:0; }
  .rcard .rtag { font-family:Consolas,monospace; font-size:13px; line-height:1.6;
                 word-break:break-all; }
  .rcard .rtagi { display:inline-block; margin:0 10px 2px 0; }
  .rcard .rtagi i { font-style:normal; opacity:.55; font-size:11px; }
  .rcard .rnl { font-size:13.5px; line-height:1.7; white-space:pre-wrap; }
  .rcard .rerr { color:#c0392b; font-size:13px; white-space:pre-wrap; }
  body.dark .rcard .rerr { color:#ff7675; }
  .rcard .rzh { color:var(--tr-color, #00b894); font-size:var(--tr-size, 13px);
                margin-top:8px; line-height:1.6; white-space:pre-wrap; }
  .dropzone { margin:0 0 12px 120px; border:2px dashed var(--input-border);
              border-radius:10px; padding:22px 16px; text-align:center; cursor:pointer;
              color:var(--muted, inherit); font-size:13.5px;
              transition:border-color .2s, background .2s; }
  .dropzone:hover { border-color:#0984e3; }
  .dropzone.drag { border-color:#0984e3; background:rgba(9,132,227,.08); }
  .dropzone img { max-height:140px; max-width:280px; border-radius:8px;
                  border:1px solid var(--border); display:block; margin:0 auto 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🏷️ 数据集打标</h1>

  <div class="tabs">
    <button class="tab active" id="tabBatch" onclick="switchTab('batch')">批量打标</button>
    <button class="tab" id="tabCmp" onclick="switchTab('cmp')">效果对比</button>
    <button id="cfgBtn" class="plain" onclick="openConfig()" style="margin-left:auto">配置…</button>
  </div>

  <div id="batchView">
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
      <div class="row" style="margin-top:10px">
        <label class="lab">输出长度</label>
        <input type="number" id="maxt" value="512" min="32" max="4096" step="32" style="width:90px">
        <label class="lab" style="width:90px; text-align:right">图片分辨率</label>
        <select id="nl_res">
          <option value="0">原始分辨率</option>
          <option value="768">768（快）</option>
          <option value="1024">1024（均衡）</option>
          <option value="1536" selected>1536（高清）</option>
          <option value="2048">2048（超清·慢）</option>
        </select>
      </div>
      <div class="tip" style="margin:-4px 0 8px 120px">输出长度（最多生成的 token 数）与图片分辨率（长边上限）仅对自然语言打标生效</div>
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
  </div><!-- /batchView -->

  <div id="cmpView" style="display:none">
  <div class="card">
    <h2>⚖️ 效果对比（单张图片 · 多模型同时打标）</h2>

    <div class="dropzone" id="dropZone">
      <input type="file" id="cmpFile" accept=".jpg,.jpeg,.png,.webp,.bmp,.gif" style="display:none">
      <img id="cmpThumb" style="display:none" alt="">
      <div id="dzHint">📎 点击选择图片，或把图片拖到这里</div>
    </div>

    <div class="row">
      <label class="lab">对比模型</label>
      <span class="tip" style="margin:0">勾选要对比的模型，可同时选标签模型和自然语言模型（最多 8 个）</span>
    </div>
    <div id="cmpModels" class="modelgrid"></div>

    <div class="row">
      <label class="lab">标签阈值</label>
      <input type="number" id="cmp_gen_t" value="0.35" min="0" max="1" step="0.05">
      <label class="lab" style="width:90px; text-align:right">角色阈值</label>
      <input type="number" id="cmp_char_t" value="0.85" min="0" max="1" step="0.05">
    </div>
    <div class="row">
      <label class="lab">输出长度</label>
      <input type="number" id="cmp_maxt" value="512" min="32" max="4096" step="32" style="width:90px">
      <label class="lab" style="width:90px; text-align:right">图片分辨率</label>
      <select id="cmp_res">
        <option value="0">原始分辨率</option>
        <option value="768">768（快）</option>
        <option value="1024">1024（均衡）</option>
        <option value="1536" selected>1536（高清）</option>
        <option value="2048">2048（超清·慢）</option>
      </select>
    </div>
    <div class="tip" style="margin:-4px 0 10px 120px">标签阈值仅对标签模型生效；输出长度（最多生成的 token 数）与图片分辨率仅对自然语言模型生效</div>

    <div class="row">
      <label class="lab">NL 提示词</label>
      <div class="tip" style="margin:0">自然语言模型使用的指令（可自由修改）</div>
    </div>
    <textarea id="cmp_nl_prompt" spellcheck="false"></textarea>

    <div class="row" style="margin-top:12px">
      <button id="cmpRunBtn" onclick="runCompare()">开始对比</button>
      <button class="plain" id="cmpCancelBtn" onclick="cancelCompare()" style="display:none">停止</button>
      <span id="cmpInfo" class="tip"></span>
    </div>

    <div class="row" id="trRow" style="display:none">
      <label class="lab">翻译结果</label>
      <select id="tr_lang" style="min-width:150px"></select>
      <input type="color" id="tr_color" value="#00b894" title="翻译文字颜色"
             style="width:42px; height:34px; padding:2px; cursor:pointer">
      <input type="number" id="tr_size" value="13" min="9" max="40" title="翻译文字字号"
             style="width:64px">
      <button class="plain" id="trBtn" onclick="translateResults()">翻译</button>
      <span id="trInfo" class="tip"></span>
    </div>

    <div id="cmpResults"></div>
  </div>
  </div><!-- /cmpView -->
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
      <thead><tr><th style="width:44px">显示</th><th>模型</th>
        <th style="width:64px">大小(G)</th><th style="width:88px">状态</th>
        <th style="width:70px">标签阈值</th><th style="width:70px">角色阈值</th>
        <th style="width:170px">备注</th><th style="width:120px">操作</th></tr></thead>
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
        o.textContent = m.label + (m.installed ? '' : '（未下载）');
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

// 切换模型时自动填入该模型的推荐阈值（WD14: 0.35/0.85，CL: 见各模型定义）；
// 切换到带推荐提示词的 NL 模型（如 JoyCaption）且用户没自定义过提示词时一并填入
let _lastAutoPrompt = '';
function maybeAutoPrompt(m, box) {
    if (m && m.kind === 'nl' && m.def_prompt && m.def_prompt !== _lastAutoPrompt
            && (!box.value || box.value === _lastAutoPrompt)) {
        box.value = m.def_prompt;
        _lastAutoPrompt = m.def_prompt;
    }
}
function applyDefThr() {
    const m = models.find(x => x.id === $('model').value);
    if (m && m.kind === 'tag' && m.def_gen != null) {
        $('gen_t').value = m.def_gen;
        $('char_t').value = m.def_char;
    }
    maybeAutoPrompt(m, $('nl_prompt'));
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
    // 标签模型 / 自然语言模型分两组显示，组间插入标题行
    const groups = [
        { kind: 'tag', title: '🏷️ 标签打标模型' },
        { kind: 'nl', title: '💬 自然语言打标模型' },
    ];
    for (const g of groups) {
        const rows = models.filter(m => m.kind === g.kind);
        if (rows.length === 0) continue;
        const gtr = document.createElement('tr');
        const gtd = document.createElement('td');
        gtd.colSpan = 8;
        gtd.textContent = g.title;
        gtd.style.cssText = 'font-weight:600; text-align:left; ' +
            'background:rgba(255,255,255,.06); letter-spacing:1px;';
        gtr.appendChild(gtd);
        tb.appendChild(gtr);
        for (const m of rows) {
            const tr = document.createElement('tr');

            const tdv = document.createElement('td');
            tdv.style.textAlign = 'center';
            const cb = document.createElement('input');
            cb.type = 'checkbox'; cb.checked = m.visible;
            cb.onchange = () => toggleVisible(m.id, cb.checked);
            tdv.appendChild(cb);

            const tdl = document.createElement('td');
            tdl.textContent = m.label;
            const tdsz = document.createElement('td');
            tdsz.textContent = m.size_gb ? (m.kind === 'nl' ? '≈' + m.size_gb : m.size_gb) : '-';
            const tds = document.createElement('td');
        if (m.installed) tds.textContent = '已下载';
        else if (m.kind !== 'tag') tds.textContent = '本地';
        else tds.textContent = m.gated ? '未下载·需HF登录' : '未下载';
        tds.style.color = m.installed ? 'var(--ok, #1b7a3d)' : 'inherit';
        const tdr = document.createElement('td');
        tdr.textContent = m.remark || '';

        // 标签模型可编辑默认阈值（change 即保存），自然语言模型显示 —
        const thrs = [];
        if (m.kind === 'tag') {
            for (const [key, val] of [['gen', m.def_gen], ['char', m.def_char]]) {
                const td = document.createElement('td');
                td.style.textAlign = 'center';
                const inp = document.createElement('input');
                inp.type = 'number'; inp.min = '0.01'; inp.max = '1'; inp.step = '0.01';
                inp.value = val;
                inp.style.cssText = 'width:58px; padding:3px 5px; border-radius:6px;'
                    + 'border:1px solid var(--input-border); background:var(--input-bg);'
                    + 'color:var(--text); font-size:12.5px;';
                inp.onchange = () => {
                    const v = parseFloat(inp.value);
                    if (isNaN(v) || v <= 0 || v > 1) {
                        inp.value = val;
                        return;
                    }
                    saveThr(m.id, key, v);
                };
                td.appendChild(inp);
                thrs.push(td);
            }
        } else {
            thrs.push(Object.assign(document.createElement('td'),
                { textContent: '—', style: 'text-align:center; opacity:.5' }));
            thrs.push(Object.assign(document.createElement('td'),
                { textContent: '—', style: 'text-align:center; opacity:.5' }));
        }

        const tdo = document.createElement('td');
        tdo.style.textAlign = 'center';
        tdo.style.whiteSpace = 'nowrap';
        if (!m.installed) {
            const bd = document.createElement('button');
            bd.className = 'mini'; bd.textContent = '下载';
            bd.onclick = () => downloadCfg(m.id, bd);
            tdo.appendChild(bd);
        }
        if (m.installed && (m.kind === 'tag' || m.builtin_nl)) {
            const bx = document.createElement('button');
            bx.className = 'mini danger'; bx.textContent = '删除';
            bx.onclick = () => delModelArm(m.id, bx);
            tdo.appendChild(bx);
        }

            tr.append(tdv, tdl, tdsz, tds, ...thrs, tdr, tdo);
            tb.appendChild(tr);
        }
    }
}

async function toggleVisible(id, visible) {
    await fetch('/api/config', { method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ hidden_delta: { id, hide: !visible } }) });
    await loadModels();
}

// 保存单个模型的默认阈值（标签或角色），并刷新主界面下拉数据
async function saveThr(id, key, val) {
    const d = { id };
    d[key] = val;
    await fetch('/api/config', { method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ thresholds: d }) });
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
        nl_res: parseInt($('nl_res').value),
        maxt: parseInt($('maxt').value) || 512,
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

// ---- 页签切换：批量打标 / 效果对比 ----
function switchTab(t) {
    const batch = t !== 'cmp';
    $('tabBatch').classList.toggle('active', batch);
    $('tabCmp').classList.toggle('active', !batch);
    $('batchView').style.display = batch ? '' : 'none';
    $('cmpView').style.display = batch ? 'none' : '';
    if (!batch) renderCmpModels();
}

// ---- 效果对比 ----
function renderCmpModels() {
    const grid = $('cmpModels');
    grid.innerHTML = '';
    const list = models.filter(m => m.installed);
    if (!list.length) {
        grid.innerHTML = '<span class="tip" style="margin-left:-120px">没有已下载的模型，请先在右上角"配置…"里下载</span>';
        return;
    }
    for (const m of list) {
        const lab = document.createElement('label');
        lab.className = 'mchip';
        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.value = m.id; cb.dataset.kind = m.kind;
        cb.onchange = () => {
            // 勾选标签模型时自动填入该模型的默认阈值（同主界面逻辑）
            if (cb.checked && m.kind === 'tag' && m.def_gen != null) {
                $('cmp_gen_t').value = m.def_gen;
                $('cmp_char_t').value = m.def_char;
            }
            // 勾选 JoyCaption 等带推荐提示词的 NL 模型时自动填入
            if (cb.checked) maybeAutoPrompt(m, $('cmp_nl_prompt'));
        };
        const badge = document.createElement('span');
        badge.className = 'badge ' + m.kind;
        badge.textContent = m.kind === 'tag' ? '标签' : 'NL';
        lab.append(cb, badge, document.createTextNode(m.label));
        grid.appendChild(lab);
    }
}

// 图片：点击选择 + 拖拽放入
const dz = $('dropZone');
dz.onclick = () => $('cmpFile').click();
dz.ondragover = e => { e.preventDefault(); dz.classList.add('drag'); };
dz.ondragleave = () => dz.classList.remove('drag');
dz.ondrop = e => {
    e.preventDefault(); dz.classList.remove('drag');
    if (e.dataTransfer.files.length) {
        $('cmpFile').files = e.dataTransfer.files;
        showCmpThumb();
    }
};
$('cmpFile').onchange = showCmpThumb;

function showCmpThumb() {
    const f = $('cmpFile').files[0];
    const th = $('cmpThumb');
    if (!f) { th.style.display = 'none'; $('dzHint').textContent = '📎 点击选择图片，或把图片拖到这里'; return; }
    const rd = new FileReader();
    rd.onload = e => {
        th.src = e.target.result; th.style.display = '';
        $('dzHint').textContent = '✅ ' + f.name + '（点击可更换）';
    };
    rd.readAsDataURL(f);
}

async function runCompare() {
    const chips = [...document.querySelectorAll('#cmpModels input:checked')];
    if (!chips.length) { alert('请先勾选至少一个模型'); return; }
    if (!$('cmpFile').files.length) { alert('请先选择一张图片'); return; }
    const fd = new FormData();
    fd.append('file', $('cmpFile').files[0]);
    fd.append('models', JSON.stringify(
        chips.map(c => ({ id: c.value, kind: c.dataset.kind }))));
    fd.append('gen_t', $('cmp_gen_t').value);
    fd.append('char_t', $('cmp_char_t').value);
    fd.append('maxt', $('cmp_maxt').value);
    fd.append('res', $('cmp_res').value);
    fd.append('nl_prompt', $('cmp_nl_prompt').value);
    $('cmpRunBtn').disabled = true;
    try {
        const r = await fetch('/api/compare', { method: 'POST', body: fd });
        const d = await r.json();
        if (!d.ok) { alert(d.error || d.message || '启动失败'); $('cmpRunBtn').disabled = false; }
    } catch (e) { alert('启动失败: ' + e); $('cmpRunBtn').disabled = false; }
}

async function cancelCompare() {
    try { await fetch('/api/compare_cancel', { method: 'POST' }); } catch (e) {}
}

function escapeHtml(s) {
    return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---- 对比结果翻译（复用 Hy-MT2 离线翻译模型）----
const TR_LANGS = [["中文（简体）","Simplified Chinese"],["中文（繁体）","Traditional Chinese"],
    ["English","English"],["日本語","Japanese"],["한국어","Korean"],["粵語","Cantonese"],
    ["Français","French"],["Deutsch","German"],["Español","Spanish"],["Русский","Russian"],
    ["Português","Portuguese"],["Italiano","Italian"]];
for (const [lab, val] of TR_LANGS) {
    const o = document.createElement('option');
    o.value = val; o.textContent = lab;
    $('tr_lang').appendChild(o);
}

// 翻译文字样式（颜色/字号）：即时生效并永久保存
function applyTrStyle() {
    document.documentElement.style.setProperty('--tr-color', $('tr_color').value);
    document.documentElement.style.setProperty('--tr-size', $('tr_size').value + 'px');
}
function saveTrStyle() {
    applyTrStyle();
    fetch('/api/config', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tr_style: { color: $('tr_color').value,
                                           size: parseInt($('tr_size').value) } })
    }).catch(() => {});
}
$('tr_color').oninput = saveTrStyle;
$('tr_size').onchange = saveTrStyle;

async function translateResults() {
    const cards = [...document.querySelectorAll('#cmpResults .rcard')]
        .filter(c => c.dataset.ok === '1' && c.querySelector('.rtag,.rnl'));
    if (!cards.length) return;
    const btn = $('trBtn');
    btn.disabled = true;
    $('trInfo').innerHTML = '<span class="spin"></span>翻译中…（首次需加载翻译模型）';
    try {
        const r = await fetch('/api/translate', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: cards.map(c => {
                const el = c.querySelector('.rtag,.rnl');
                // 标签卡片里混有置信度，只取纯标签文本
                if (el.classList.contains('rtag')) {
                    return [...el.querySelectorAll('.rtagi')]
                        .map(s => s.dataset.t).join(', ');
                }
                return el.textContent;
            }), lang: $('tr_lang').value })
        });
        const d = await r.json();
        if (!d.ok) {
            $('trInfo').innerHTML = '<span class="err">❌ ' + escapeHtml(d.error || '翻译失败') + '</span>';
        } else {
            cards.forEach((c, i) => {
                let zh = c.querySelector('.rzh');
                if (!zh) { zh = document.createElement('div'); zh.className = 'rzh'; c.appendChild(zh); }
                zh.textContent = (d.translations || [])[i] || '';
            });
            $('trInfo').textContent = '✔ 翻译完成';
        }
    } catch (e) {
        $('trInfo').innerHTML = '<span class="err">❌ ' + escapeHtml(String(e)) + '</span>';
    }
    btn.disabled = false;
}

let _cmpRenderN = -1;   // 已渲染的结果卡片数，结果没变化就不重绘（保留翻译内容）

async function pollCmp() {
    try {
        const r = await fetch('/api/compare_status');
        const s = await r.json();
        $('cmpCancelBtn').style.display = s.running ? '' : 'none';
        $('cmpRunBtn').disabled = s.running;
        let info = '';
        if (s.running) {
            info = '<span class="spin"></span>进行中 ' + s.done + '/' + s.total;
            if (s.current) info += ' · 当前：' + escapeHtml(s.current);
        } else if (s.finished && !s.error && s.done === s.total && s.total > 0) {
            info = '✔ 对比完成';
        } else if (s.finished && s.total > 0) {
            info = s.cancel ? '已停止' : '完成（部分失败）';
        }
        if (s.error) info += '　<span class="err">❌ ' + escapeHtml(s.error) + '</span>';
        $('cmpInfo').innerHTML = info;
        const n = (s.results || []).length;
        $('trRow').style.display = n ? '' : 'none';
        if (!n) {
            if (_cmpRenderN !== 0) { $('cmpResults').innerHTML = ''; _cmpRenderN = 0; }
            return;
        }
        if (!s.running && n === _cmpRenderN) return;   // 已完成且结果未变，跳过重绘
        _cmpRenderN = n;
        let html = '';
        for (const res of s.results) {
            const badge = '<span class="badge ' + res.kind + '">'
                        + (res.kind === 'tag' ? '标签' : 'NL') + '</span>';
            let body;
            if (res.ok) {
                if (res.kind === 'tag' && Array.isArray(res.detail)) {
                    // 每个标签旁展示置信度（3 位小数）
                    body = '<div class="rtag">' + res.detail.map(d =>
                        '<span class="rtagi" data-t="' + escapeHtml(d[0]) + '">'
                        + escapeHtml(d[0]) + ' <i>' + Number(d[1]).toFixed(3) + '</i></span>'
                    ).join('') + '</div>';
                } else {
                    body = '<div class="' + (res.kind === 'tag' ? 'rtag' : 'rnl') + '">'
                         + escapeHtml(res.output) + '</div>';
                }
            } else {
                body = '<div class="rerr">❌ ' + escapeHtml(res.error) + '</div>';
            }
            html += '<div class="rcard" data-ok="' + (res.ok ? 1 : 0) + '">'
                  + '<div class="rhead">' + badge
                  + '<b>' + escapeHtml(res.label) + '</b>'
                  + '<span class="rsec">' + (res.sec ? res.sec + 's' : '') + '</span></div>'
                  + body + '</div>';
        }
        if (s.running && s.current && s.done < s.total) {
            html += '<div class="rcard"><div class="rhead"><span class="spin"></span>'
                  + escapeHtml(s.current) + ' 推理中…</div></div>';
        }
        $('cmpResults').innerHTML = html;
    } catch (e) {}
}

// ---- 初始化 ----
(async function init() {
    try {
        const c = await (await fetch('/api/config')).json();
        if (c.tr_color) $('tr_color').value = c.tr_color;
        if (c.tr_size) $('tr_size').value = c.tr_size;
        applyTrStyle();
    } catch (e) {}
    try {
        const r = await fetch('/api/nl_prompt');
        const j = await r.json();
        $('nl_prompt').value = j.prompt;
        $('cmp_nl_prompt').value = j.prompt;
        _lastAutoPrompt = j.prompt;
    } catch (e) {}
    await loadModels();
    pollJob();
    setInterval(pollJob, 1000);
    pollCmp();
    setInterval(pollCmp, 1200);
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
    thr_map = cfg.get("thresholds") or {}
    for m in WD14_MODELS + CL_MODELS:
        engine = m.get("engine", "wd14")
        # 默认阈值：配置页设置优先，否则用注册表内置值（WD 无内置 → 0.35 / 0.85）
        t = thr_map.get(m["id"]) or {}
        def_gen = t.get("gen", m.get("def_gen", 0.35))
        def_char = t.get("char", m.get("def_char", 0.85))
        models.append({"id": m["id"], "label": m["label"], "kind": "tag",
                       "remark": m.get("remark", ""), "size_gb": m.get("size_gb", 0),
                       "custom": False,
                       "engine": engine,
                       "gated": m.get("gated", False),
                       "def_gen": def_gen,
                       "def_char": def_char,
                       "visible": m["id"] not in cfg.get("hidden", []),
                       "installed": model_installed(m)})
    for key, m in scan_nl_models().items():
        is_joy = "joycaption" in m["label"].lower()
        models.append({"id": key, "label": m["label"], "kind": "nl",
                       "remark": ("自然语言，详细图像描述（JoyCaption）" if is_joy
                                  else "本地视觉模型，可调提示词"),
                       "size_gb": m.get("size_gb", 0),
                       "custom": False,
                       "def_prompt": JOYCAPTION_PROMPT if is_joy else DEFAULT_NL_PROMPT,
                       "builtin_nl": is_joy,
                       "visible": key not in cfg.get("hidden", []),
                       "installed": True})
    for m in BUILTIN_NL_MODELS:
        if builtin_nl_installed(m):
            continue   # 已安装的由上面扫描条目呈现
        models.append({"id": m["id"], "label": m["label"], "kind": "nl",
                       "remark": m.get("remark", ""), "size_gb": m.get("size_gb", 0),
                       "custom": False,
                       "def_prompt": JOYCAPTION_PROMPT,
                       "builtin_nl": True,
                       "visible": m["id"] not in cfg.get("hidden", []),
                       "installed": False})
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

    if "thresholds" in data:   # 单个标签模型的默认阈值 {id, gen, char}
        d = data.get("thresholds") or {}
        mid = d.get("id")
        if not mid:
            return jsonify({"ok": False, "error": "缺少模型 id"}), 400
        tmap = cfg.setdefault("thresholds", {})
        entry = tmap.setdefault(mid, {})
        try:
            entry["gen"] = round(min(1.0, max(0.01, float(d.get("gen")))), 2)
        except (TypeError, ValueError):
            pass
        try:
            entry["char"] = round(min(1.0, max(0.01, float(d.get("char")))), 2)
        except (TypeError, ValueError):
            pass
        save_config(cfg)
        return jsonify({"ok": True})

    if "tr_style" in data:   # 翻译文字颜色/字号，永久保存
        d = data.get("tr_style") or {}
        c = d.get("color")
        if (isinstance(c, str) and len(c) == 7 and c.startswith("#")
                and all(ch in "0123456789abcdefABCDEF" for ch in c[1:])):
            cfg["tr_color"] = c
        try:
            cfg["tr_size"] = min(40, max(9, int(d.get("size"))))
        except (TypeError, ValueError):
            pass
        save_config(cfg)
        return jsonify({"ok": True, "tr_color": cfg["tr_color"],
                        "tr_size": cfg["tr_size"]})

    return jsonify({"ok": False, "error": "无效的配置操作"}), 400


@app.get("/api/nl_prompt")
def api_nl_prompt():
    return jsonify({"ok": True, "prompt": DEFAULT_NL_PROMPT})


@app.post("/api/download")
def api_download():
    cfg = load_config()
    data = request.get_json(silent=True) or {}
    model_id = data.get("id", "")
    bnl = builtin_nl_entry(model_id)
    if bnl and bnl["kind"] == "nl":
        if builtin_nl_installed(bnl):
            return jsonify({"ok": False, "error": "模型已存在"}), 400
        with _dl_lock:
            if _dl_state["active"]:
                return jsonify({"ok": False, "error": "已有下载在进行中"}), 409
        threading.Thread(target=_download_nl, args=(bnl, cfg), daemon=True).start()
        return jsonify({"ok": True})
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
    """删除一个已下载的模型（标签模型目录 / 内置自然语言模型文件）。"""
    cfg = load_config()
    model_id = (request.get_json(silent=True) or {}).get("id", "")
    bnl = builtin_nl_entry(model_id)
    if bnl:
        if not builtin_nl_installed(bnl):
            return jsonify({"ok": False, "error": "模型未下载"}), 400
        with _job_lock:
            if _job["running"]:
                return jsonify({"ok": False, "error": "打标任务进行中，无法删除"}), 409
        with _cmp_lock:
            if _cmp["running"]:
                return jsonify({"ok": False, "error": "效果对比进行中，无法删除"}), 409
        try:
            for f in (bnl["file"], JOYCAPTION_MMPROJ_LOCAL):
                p = os.path.join(MODEL_DIR, f)
                if os.path.isfile(p):
                    os.remove(p)
                    print(f"[tagger] 已删除模型文件：{p}")
        except Exception as e:
            return jsonify({"ok": False, "error": f"{e.__class__.__name__}: {e}"}), 500
        return jsonify({"ok": True})
    m = model_entry(cfg, model_id)
    if not m or m.get("kind") == "nl":
        return jsonify({"ok": False, "error": "该模型不支持删除"}), 400
    with _job_lock:
        if _job["running"]:
            return jsonify({"ok": False, "error": "打标任务进行中，无法删除"}), 409
    with _cmp_lock:
        if _cmp["running"]:
            return jsonify({"ok": False, "error": "效果对比进行中，无法删除"}), 409
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


@app.post("/api/compare")
def api_compare():
    """上传单张图片 + 模型列表 + 共用参数，启动效果对比。"""
    global _cmp_img
    with _cmp_lock:
        if _cmp["running"]:
            return jsonify({"ok": False, "error": "已有对比任务在运行中"}), 409
    with _job_lock:
        if _job["running"]:
            return jsonify({"ok": False, "error": "批量打标任务进行中，请先等待完成"}), 409
    file = request.files.get("file")
    try:
        models = json.loads(request.form.get("models", "[]"))
    except ValueError:
        return jsonify({"ok": False, "error": "参数错误"}), 400
    if not file or not isinstance(models, list) or not models:
        return jsonify({"ok": False, "error": "请选择图片并至少勾选一个模型"}), 400
    if len(models) > 8:
        models = models[:8]
    if not file.filename.lower().endswith(IMG_EXTS):
        return jsonify({"ok": False, "error": "不支持的图片格式"}), 400
    os.makedirs(TAGGER_DIR, exist_ok=True)
    _cmp_img = os.path.join(TAGGER_DIR, "_compare_input"
                            + os.path.splitext(file.filename)[1].lower())
    file.save(_cmp_img)
    try:
        with Image.open(_cmp_img) as im:
            im.verify()
    except Exception:
        _cmp_img = None
        return jsonify({"ok": False, "error": "图片无法解码"}), 400
    ok, msg = start_compare(models, request.form)
    return jsonify({"ok": ok, "error": msg}), (200 if ok else 400)


@app.get("/api/compare_status")
def api_compare_status():
    with _cmp_lock:
        return jsonify(dict(_cmp))


@app.post("/api/compare_cancel")
def api_compare_cancel():
    with _cmp_lock:
        if _cmp["running"]:
            _cmp["cancel"] = True
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "没有正在运行的对比"})


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
