# -*- coding: utf-8 -*-
"""
inspect_lora.py
================
扫描当前目录（或指定目录）下的 .safetensors 文件，识别其中 LoRA /
LyCORIS (LoCon / LoHa / LoKr / GLoRA / OFT / IA³) 网络的训练参数：

  · 算法类型 + 网络 rank / dim / alpha
  · 作用在 UNet / TE 的哪些子层（按 input_blocks / middle_block /
    output_blocks / encoder_layers 等聚合）
  · 训练步数、epoch、最终 epoch
  · 基础模型、分辨率、batch size、优化器、学习率、调度器等

使用：
    python inspect_lora.py                 # 扫描当前目录
    python inspect_lora.py <dir>           # 扫描指定目录
    python inspect_lora.py file1.safetensors file2.safetensors
    python inspect_lora.py --json out.json # 同时把结果写成 JSON

依赖：
    pip install safetensors
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone

try:
    from safetensors import safe_open
except ImportError:
    sys.stderr.write("缺少依赖 safetensors，请先执行：pip install safetensors\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. 算法 / key 识别
# ---------------------------------------------------------------------------

# 训练器常用的 LoRA / LyCORIS key 后缀。注意 LyCORIS 没有 ".weight"。
LORA_STD_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")
LYCORIS_SUFFIXES = (
    # LoCon
    ".lora_down", ".lora_up", ".lora_mid",
    # LoHa
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".hada_t1", ".hada_t2",
    # LoKr
    ".lokr_w1", ".lokr_w2", ".lokr_w1_b", ".lokr_w1_a",
    ".lokr_t1", ".lokr_t2", ".lokr_w2_b", ".lokr_w2_a",
    # OFT
    ".oft_blocks", ".oft_diag",
    # IA³
    ".on_input", ".on_output",
    # GLoRA / (IA)³V
    ".w1", ".w2", ".w1_a", ".w1_b", ".w2_a", ".w2_b", ".b1", ".b2",
)

# 已知的网络类型在 metadata 中的标识
KNOWN_NET_MODULES = {
    "networks.lora": "LoRA",
    "networks.dylora": "DyLoRA",
    "networks.loha": "LyCORIS/LoHa",
    "networks.lokr": "LyCORIS/LoKr",
    "lycoris.kohya": "LyCORIS",
    "lycoris.loha": "LyCORIS/LoHa",
    "lycoris.lokr": "LyCORIS/LoKr",
    "lycoris.locon": "LyCORIS/LoCon",
    "lycoris.glocora": "LyCORIS/GLoRA",
    "lycoris.oft": "LyCORIS/OFT",
    "lycoris.ia3": "LyCORIS/IA³",
    "lycoris.diag_oft": "LyCORIS/Diag-OFT",
}

# key 命名前缀（决定作用在哪个子模型上）
TE_PREFIXES = (
    "lora_te_text_model_",   # SD 1.5
    "lora_te1_",             # SDXL TE1
    "lora_te2_",             # SDXL TE2
    "lora_te_",              # 通用
)
UNET_PREFIX = "lora_unet_"


def detect_algorithm(keys):
    """根据 key 后缀推断算法类型。"""
    has_std = any(k.endswith(LORA_STD_SUFFIXES) for k in keys)
    has_lyc = any(k.endswith(LYCORIS_SUFFIXES) for k in keys)
    if not has_std and not has_lyc:
        return ("unknown", set())

    suffix_counter = Counter()
    for k in keys:
        for s in LORA_STD_SUFFIXES:
            if k.endswith(s):
                suffix_counter["std_" + s] += 1
                break
        else:
            for s in LYCORIS_SUFFIXES:
                if k.endswith(s):
                    suffix_counter["lyc_" + s] += 1
                    break

    # 决定算法
    if has_lyc:
        if any("hada_" in s for s in suffix_counter):
            return ("LyCORIS/LoHa", suffix_counter)
        if any("lokr_" in s for s in suffix_counter):
            return ("LyCORIS/LoKr", suffix_counter)
        if any("oft_" in s for s in suffix_counter):
            return ("LyCORIS/OFT", suffix_counter)
        if any(s in ("lyc_.on_input", "lyc_.on_output") for s in suffix_counter):
            return ("LyCORIS/IA³", suffix_counter)
        if any("w1" in s or "b1" in s for s in suffix_counter):
            return ("LyCORIS/GLoRA", suffix_counter)
        if "lyc_.lora_mid" in suffix_counter:
            return ("LyCORIS/LoCon", suffix_counter)
        if "lyc_.lora_down" in suffix_counter or "lyc_.lora_up" in suffix_counter:
            return ("LyCORIS/LoCon", suffix_counter)
        return ("LyCORIS", suffix_counter)
    return ("LoRA", suffix_counter)


def infer_rank(open_file, keys, algorithm):
    """通过一两个 tensor 的形状推断网络 rank。"""
    for k in keys:
        try:
            shape = open_file.get_slice(k).get_shape()
        except Exception:
            continue
        s = list(shape)
        if algorithm == "LoRA" and k.endswith(".lora_down.weight"):
            # 形状 (rank, in_features)
            return s[0] if s else None
        if algorithm == "LyCORIS/LoCon":
            if k.endswith(".lora_down"):
                return s[0] if s else None
            if k.endswith(".lora_mid"):
                # LoCon mid 形状 (rank, rank, kernel_h, kernel_w)，rank 是第 0 维
                return s[0] if s else None
        if algorithm == "LyCORIS/LoHa":
            if k.endswith(".hada_w1_a") or k.endswith(".hada_w2_a"):
                return s[0] if s else None
        if algorithm == "LyCORIS/LoKr":
            if k.endswith(".lokr_w1") or k.endswith(".lokr_w1_b"):
                return s[0] if s else None
        if algorithm == "LyCORIS/GLoRA":
            if k.endswith(".w1") or k.endswith(".w1_a"):
                return s[0] if s else None
        if algorithm == "LyCORIS/OFT":
            if k.endswith(".oft_blocks"):
                # 形状 (rank, num_blocks)
                return s[0] if s else None
        if algorithm == "LyCORIS/IA³":
            if k.endswith(".on_input") or k.endswith(".on_output"):
                return s[0] if s else None
    return None


# ---------------------------------------------------------------------------
# 2. 层级解析
# ---------------------------------------------------------------------------

def strip_suffix(key):
    """剥掉算法后缀，留下“层路径”部分。"""
    for s in LORA_STD_SUFFIXES + LYCORIS_SUFFIXES:
        if key.endswith(s):
            return key[:-len(s)]
    return key


def parse_submodel(layer_path):
    """把 key 归到 unet / te1 / te2 / other。"""
    if layer_path.startswith(UNET_PREFIX):
        return "unet", layer_path[len(UNET_PREFIX):]
    if layer_path.startswith("lora_te_text_model_"):
        return "te1", layer_path[len("lora_te_text_model_"):]
    if layer_path.startswith("lora_te1_"):
        return "te1", layer_path[len("lora_te1_"):]
    if layer_path.startswith("lora_te2_"):
        return "te2", layer_path[len("lora_te2_"):]
    if layer_path.startswith("lora_te_"):
        return "te", layer_path[len("lora_te_"):]
    return "other", layer_path


def unet_region(rest):
    """把 unet 层路径归到 input_blocks / middle_block / output_blocks / other。"""
    parts = rest.split(".")
    if not parts:
        return "other"
    head = parts[0]
    if head in ("input_blocks", "down_blocks"):
        # SDXL: down_blocks.X.attentions.Y
        if len(parts) >= 2:
            return f"{head}.{parts[1]}"
        return head
    if head in ("output_blocks", "up_blocks"):
        if len(parts) >= 2:
            return f"{head}.{parts[1]}"
        return head
    if head == "middle_block":
        if len(parts) >= 2:
            return f"middle_block.{parts[1]}"
        return "middle_block"
    return head


def te_region(rest):
    """把 te 层路径归到 encoder_layers / 其他。"""
    parts = rest.split(".")
    if not parts:
        return "other"
    head = parts[0]
    if head == "encoder":
        if len(parts) >= 3 and parts[1] == "layers":
            return f"encoder_layers.{parts[2]}"
        return "encoder"
    if head == "embeddings":
        return "embeddings"
    if head == "final":
        return "final"
    return head


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def safe_get(meta, key, default=None):
    """metadata 字段可能以 JSON 字符串保存，做个安全读取。"""
    v = meta.get(key, default)
    if v == "None":
        return None
    return v


def to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return int(float(v))
        except Exception:
            return None


def to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def human_size(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def inspect_one(path):
    """读取单个 safetensors 文件，返回结构化信息 dict。"""
    info = OrderedDict()
    info["file"] = os.path.basename(path)
    info["path"] = os.path.abspath(path)
    info["file_size"] = human_size(os.path.getsize(path))

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}

    info["num_tensors"] = len(keys)

    algorithm, suffix_counter = detect_algorithm(keys)
    info["algorithm"] = algorithm
    info["suffix_counter"] = dict(suffix_counter) if suffix_counter else {}

    # ---- 网络类型 / dim / alpha ----
    net_module = safe_get(meta, "ss_network_module") or safe_get(meta, "network_module")
    info["network_module"] = net_module
    info["network_module_pretty"] = KNOWN_NET_MODULES.get(net_module, net_module or "")

    network_args = safe_get(meta, "ss_network_args")
    if isinstance(network_args, str) and network_args.startswith("{"):
        try:
            network_args = json.loads(network_args)
        except json.JSONDecodeError:
            pass
    info["network_args"] = network_args

    rank_meta = to_int(safe_get(meta, "ss_network_dim")) or to_int(safe_get(meta, "network_dim"))
    alpha_meta = to_int(safe_get(meta, "ss_network_alpha")) or to_int(safe_get(meta, "network_alpha"))
    conv_rank_meta = to_int(safe_get(meta, "ss_network_conv_dim"))
    conv_alpha_meta = to_int(safe_get(meta, "ss_network_conv_alpha"))

    with safe_open(path, framework="pt") as f:
        rank_infer = infer_rank(f, keys, algorithm)

    info["rank_meta"] = rank_meta
    info["alpha_meta"] = alpha_meta
    info["conv_rank_meta"] = conv_rank_meta
    info["conv_alpha_meta"] = conv_alpha_meta
    info["rank_inferred"] = rank_infer
    info["alpha_effective"] = (
        alpha_meta if alpha_meta is not None
        else (rank_meta if rank_meta is not None else rank_infer)
    )

    # ---- 层级统计 ----
    region_counter = Counter()
    submodel_counter = Counter()
    per_region_layers = defaultdict(set)

    for k in keys:
        layer_path = strip_suffix(k)
        sub, rest = parse_submodel(layer_path)
        submodel_counter[sub] += 1
        if sub == "unet":
            r = unet_region(rest)
        elif sub in ("te", "te1", "te2"):
            r = te_region(rest)
        else:
            r = "other"
        region_counter[r] += 1
        # 取更精细的层名
        short = rest
        if len(short) > 80:
            short = short[:77] + "..."
        per_region_layers[r].add(short)

    info["submodel_summary"] = dict(submodel_counter)
    info["region_summary"] = OrderedDict(sorted(region_counter.items()))
    info["region_unique_layers"] = {k: len(v) for k, v in per_region_layers.items()}
    info["region_samples"] = {
        k: sorted(v)[:3] for k, v in per_region_layers.items()
    }

    # ---- 训练参数 ----
    info["base_model_name"] = safe_get(meta, "ss_sd_model_name")
    info["base_model_version"] = safe_get(meta, "ss_base_model_version")
    info["architecture"] = safe_get(meta, "modelspec.architecture") or safe_get(meta, "ss_v2")
    info["is_v2"] = safe_get(meta, "ss_v2")
    info["is_sdxl"] = (
        "xl" in str(info["architecture"] or "").lower()
        or "xl" in str(info["base_model_version"] or "").lower()
        or "sdxl" in str(info["base_model_name"] or "").lower()
    )

    info["resolution"] = safe_get(meta, "ss_resolution")
    info["clip_skip"] = safe_get(meta, "ss_clip_skip")
    info["batch_size_per_device"] = to_int(safe_get(meta, "ss_batch_size_per_device"))
    info["total_batch_size"] = to_int(safe_get(meta, "ss_total_batch_size"))
    info["grad_accum"] = to_int(safe_get(meta, "ss_gradient_accumulation_steps"))

    info["learning_rate"] = to_float(safe_get(meta, "ss_learning_rate"))
    info["unet_lr"] = to_float(safe_get(meta, "ss_unet_lr"))
    info["text_encoder_lr"] = to_float(safe_get(meta, "ss_text_encoder_lr"))
    info["lr_scheduler"] = safe_get(meta, "ss_lr_scheduler")
    info["lr_warmup_steps"] = to_int(safe_get(meta, "ss_lr_warmup_steps"))
    info["optimizer"] = safe_get(meta, "ss_optimizer")

    info["num_train_images"] = to_int(safe_get(meta, "ss_num_train_images"))
    info["num_reg_images"] = to_int(safe_get(meta, "ss_num_reg_images"))
    info["num_batches_per_epoch"] = to_int(safe_get(meta, "ss_num_batches_per_epoch"))
    info["num_epochs"] = to_int(safe_get(meta, "ss_num_epochs"))
    info["max_train_steps"] = to_int(safe_get(meta, "ss_max_train_steps"))
    info["final_step"] = to_int(safe_get(meta, "ss_steps"))
    info["final_epoch"] = to_int(safe_get(meta, "ss_epoch"))

    started = to_float(safe_get(meta, "ss_training_started_at"))
    finished = to_float(safe_get(meta, "ss_training_finished_at"))
    if started:
        info["started_at"] = datetime.fromtimestamp(started, tz=timezone.utc).isoformat()
    if finished:
        info["finished_at"] = datetime.fromtimestamp(finished, tz=timezone.utc).isoformat()
    if started and finished:
        info["training_duration"] = f"{finished - started:.1f} 秒"
    info["seed"] = to_int(safe_get(meta, "ss_seed"))
    info["mixed_precision"] = safe_get(meta, "ss_mixed_precision")
    info["gradient_checkpointing"] = safe_get(meta, "ss_gradient_checkpointing")
    info["min_snr_gamma"] = to_float(safe_get(meta, "ss_min_snr_gamma"))
    info["noise_offset"] = to_float(safe_get(meta, "ss_noise_offset"))
    info["ip_noise_gamma"] = to_float(safe_get(meta, "ss_ip_noise_gamma"))
    info["zero_terminal_snr"] = safe_get(meta, "ss_zero_terminal_snr")
    info["bucket_info"] = safe_get(meta, "ss_bucket_info")
    info["shuffle_caption"] = safe_get(meta, "ss_shuffle_caption")
    info["keep_tokens"] = to_int(safe_get(meta, "ss_keep_tokens"))
    info["caption_dropout_rate"] = to_float(safe_get(meta, "ss_caption_dropout_rate"))
    info["caption_tag_dropout_rate"] = to_float(safe_get(meta, "ss_caption_tag_dropout_rate"))
    info["loss_type"] = safe_get(meta, "ss_loss_type")

    # ---- 模型 hash / 训练器信息 ----
    info["sd_model_hash"] = safe_get(meta, "ss_sd_model_hash") or safe_get(meta, "sshs_model_hash")
    info["new_sd_model_hash"] = safe_get(meta, "ss_new_sd_model_hash")
    info["session_id"] = safe_get(meta, "ss_session_id")
    info["sd_scripts_commit"] = safe_get(meta, "ss_sd_scripts_commit_hash")
    info["output_name"] = safe_get(meta, "ss_output_name") or safe_get(meta, "modelspec.title")

    return info


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------

def _bool_str(v):
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if v is None:
        return "-"
    return str(v)


def print_report(info):
    print("=" * 78)
    print(f"文件:       {info['file']}")
    print(f"路径:       {info['path']}")
    print(f"大小:       {info['file_size']}    tensor 数: {info['num_tensors']}")
    print("-" * 78)
    print(f"算法:       {info['algorithm']}"
          + (f"  ({info['network_module_pretty']})" if info.get('network_module_pretty') else ""))

    rank = info["rank_meta"] or info["rank_inferred"]
    alpha = info["alpha_meta"] if info["alpha_meta"] is not None else rank
    if rank is not None:
        line = f"网络 dim/rank: {rank}    alpha: {alpha}"
        if info["conv_rank_meta"]:
            line += f"    conv dim/alpha: {info['conv_rank_meta']}/{info['conv_alpha_meta']}"
        if info["network_args"]:
            line += f"    args: {info['network_args']}"
        print(line)
    else:
        print("网络 dim/rank: (metadata 未记录，已无法从权重推断)")

    print("-" * 78)
    arch = info.get("architecture") or (
        "SDXL" if info.get("is_sdxl") else ("SDv2" if str(info.get("is_v2")).lower() == "true" else "SD 1.5(?)")
    )
    print(f"基础架构:   {arch}")
    if info.get("base_model_name"):
        print(f"底模:       {info['base_model_name']}"
              + (f"  ({info['base_model_version']})" if info.get("base_model_version") else ""))
    if info.get("sd_model_hash"):
        print(f"底模 hash:  {info['sd_model_hash']}")

    print("-" * 78)
    print("训练过程:")
    if info.get("final_step") is not None or info.get("num_epochs") is not None:
        print(f"  步数 / epoch:  step={info.get('final_step')}  "
              f"max_step={info.get('max_train_steps')}  "
              f"epoch={info.get('final_epoch')} / {info.get('num_epochs')}")
    if info.get("num_train_images") is not None or info.get("num_batches_per_epoch") is not None:
        print(f"  训练图片:      {info.get('num_train_images')}  "
              f"正则图片: {info.get('num_reg_images')}  "
              f"每 epoch batch: {info.get('num_batches_per_epoch')}")
    if info.get("batch_size_per_device") is not None or info.get("total_batch_size") is not None:
        print(f"  batch size:    每卡={info.get('batch_size_per_device')}  "
              f"总={info.get('total_batch_size')}  "
              f"梯度累加={info.get('grad_accum')}")
    if info.get("resolution"):
        print(f"  分辨率:        {info['resolution']}    clip_skip={info.get('clip_skip')}")
    if info.get("learning_rate") is not None or info.get("unet_lr") is not None:
        print(f"  学习率:        lr={info.get('learning_rate')}  "
              f"unet={info.get('unet_lr')}  te={info.get('text_encoder_lr')}  "
              f"scheduler={info.get('lr_scheduler')}")
    if info.get("lr_warmup_steps") is not None:
        print(f"  warmup:        {info.get('lr_warmup_steps')} 步")
    if info.get("optimizer"):
        print(f"  优化器:        {info.get('optimizer')}")
    extras = []
    if info.get("min_snr_gamma") is not None:
        extras.append(f"min_snr_gamma={info['min_snr_gamma']}")
    if info.get("noise_offset") is not None:
        extras.append(f"noise_offset={info['noise_offset']}")
    if info.get("ip_noise_gamma") is not None:
        extras.append(f"ip_noise_gamma={info['ip_noise_gamma']}")
    if info.get("zero_terminal_snr"):
        extras.append("zero_terminal_snr")
    if info.get("loss_type"):
        extras.append(f"loss={info['loss_type']}")
    if extras:
        print(f"  其他:          {', '.join(extras)}")
    if info.get("mixed_precision"):
        print(f"  精度:          {info['mixed_precision']}  "
              f"gradient_checkpointing={_bool_str(info.get('gradient_checkpointing'))}")
    if info.get("started_at"):
        dur = f"   耗时: {info.get('training_duration')}" if info.get("training_duration") else ""
        print(f"  起止时间:      {info['started_at']}  →  {info.get('finished_at')}{dur}")
    if info.get("seed") is not None:
        print(f"  随机种子:      {info['seed']}")

    if info.get("bucket_info"):
        try:
            bi = info["bucket_info"]
            if isinstance(bi, str):
                bi = json.loads(bi)
            buckets = bi.get("buckets", {}) if isinstance(bi, dict) else {}
            if buckets:
                print(f"  分桶:          共 {len(buckets)} 个")
        except Exception:
            pass

    print("-" * 78)
    print("作用层级统计:")
    sub = info["submodel_summary"]
    if sub:
        for name, cnt in sub.items():
            print(f"  {name:<6}: {cnt} 个 tensor")
    print("  详细分布:")
    for region, cnt in info["region_summary"].items():
        unique = info["region_unique_layers"].get(region, 0)
        sample = info["region_samples"].get(region, [])
        sample_s = ", ".join(sample)
        if len(sample_s) > 80:
            sample_s = sample_s[:77] + "..."
        print(f"    {region:<32}  tensor={cnt:<5}  不同层={unique:<3}  例: {sample_s}")

    print("=" * 78)
    print()


def main():
    ap = argparse.ArgumentParser(
        description="识别当前目录里的 .safetensors LoRA 文件，提取训练参数。",
    )
    ap.add_argument("paths", nargs="*", help="要扫描的文件/目录；省略则扫描当前目录")
    ap.add_argument("--json", help="把结果额外写入此 JSON 文件", default=None)
    ap.add_argument("--quiet", action="store_true", help="只打印 JSON，不打印文本报告")
    args = ap.parse_args()

    targets = []
    if not args.paths:
        cwd = os.getcwd()
        for f in sorted(os.listdir(cwd)):
            if f.lower().endswith(".safetensors"):
                targets.append(os.path.join(cwd, f))
    else:
        for p in args.paths:
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(".safetensors"):
                        targets.append(os.path.join(p, f))
            elif os.path.isfile(p):
                targets.append(p)
            else:
                print(f"[跳过] 找不到: {p}", file=sys.stderr)

    if not targets:
        print("未在指定位置找到 .safetensors 文件。")
        return 1

    results = []
    for t in targets:
        try:
            info = inspect_one(t)
        except Exception as e:
            print(f"[失败] {t}: {e}", file=sys.stderr)
            continue
        results.append(info)
        if not args.quiet:
            print_report(info)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
        print(f"JSON 已写入: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
