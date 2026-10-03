# -*- coding: utf-8 -*-
"""
构建 tags_merged.csv（label_shop 离线数据包）
1. 复用 app.py 的三来源合并/分类管线（danbooru_all + WD14 v2/v3 并集）
2. 把整理好的数据写成本工具目录下的 tags_merged.csv（files 文件夹外面）
   列：name, count, cat, sub, zh, zh_mt, aliases, sources, note
3. 用启动器的 HY-MT 翻译模型（model/Hy-MT2-1.8B-Q4_K_M.gguf）把所有标签
   批量机翻（含已有人工中文的，与 zh 并列保存），结果写进 zh_mt 列，
   随 CSV 离线保存，label_shop 每张卡片都会显示机排行

断点续跑：每次翻译先继承 tags_merged.csv 里已有的 zh_mt；运行中每 15 秒
原子重写一次 CSV，中断（Ctrl+C / 崩溃）不丢进度。_mt_progress.json 只做进度展示。

并行加速（多进程）：Windows 下同进程多线程跑多个 llama 实例会因 OpenMP
死锁，因此并行用多进程：--shard "i/N" 让本进程只翻第 i 份分片（结果写
_mt_shard_i.json，不直接写 CSV），N 个进程并行跑完后再用 --merge-only
把所有分片合并进 tags_merged.csv。每份模型约占 2GB 显存，按显存决定 N。

用法（用启动器自带的 python）：
  python build_data.py --limit 40            # 小批量试跑
  python build_data.py --workers 5           # 单进程多线程（Windows 不建议，>1 会死锁）
  多进程并行（PowerShell，5 进程示例）：
    0..4 | % { Start-Process $py -ArgumentList "`"$bd`" --shard `"$_/5`"" -WindowStyle Hidden }
    # 全部跑完后：
    python build_data.py --merge-only
"""
import argparse
import csv
import gc
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app as shop  # noqa: E402  复用 load_tags() 的并集 + 分类管线

MERGED = shop.MERGED_CSV
PROG = os.path.join(HERE, "_mt_progress.json")
ROOT = os.path.dirname(os.path.dirname(HERE))  # 统一启动器/
MODEL = os.path.join(ROOT, "model", "Hy-MT2-1.8B-Q4_K_M.gguf")

PROMPT = "Translate the following segment into Chinese, without additional explanation.\n\n"
FIELDS = ["name", "count", "cat", "sub", "zh", "zh_mt", "aliases", "sources", "note"]


def write_csv(rows):
    """原子写：先写 .tmp 再替换，避免 label_shop 读到半截文件；tmp 带 pid 防多进程冲突"""
    tmp = f"{MERGED}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, MERGED)


def translate_one(llm, name):
    text = name.replace("_", " ")
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT + text}],
        max_tokens=128, temperature=0)
    txt = (out["choices"][0]["message"]["content"] or "").strip()
    return txt.replace("\n", " ")


def load_shard_results():
    """读取所有分片结果文件，合并成 {name: zh_mt}"""
    import glob
    out = {}
    for p in glob.glob(os.path.join(HERE, "_mt_shard_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                out.update(json.load(f))
        except Exception as e:
            print(f"[build] 读取分片 {p} 失败: {e}")
    return out


def merge_only():
    """把所有 _mt_shard_*.json 的机翻结果合并进 tags_merged.csv，然后清理分片文件"""
    results = load_shard_results()
    if not results:
        print("[build] 没有可合并的分片结果（_mt_shard_*.json）")
        return
    with open(MERGED, encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))
    new = 0
    for r in rows:
        mt = results.get((r.get("name") or "").strip())
        if mt and not (r.get("zh_mt") or "").strip():
            r["zh_mt"] = mt
            new += 1
    write_csv(rows)
    import glob
    removed = 0
    for p in glob.glob(os.path.join(HERE, "_mt_shard_*.json")):
        try:
            os.remove(p)
            removed += 1
        except Exception:
            pass
    total_mt = sum(1 for r in rows if (r.get("zh_mt") or "").strip())
    with open(PROG, "w", encoding="utf-8") as f:
        json.dump({
            "translated": total_mt,
            "total_rows": len(rows),
            "merged_this_run": new,
            "shards_removed": removed,
            "ts": time.time(),
        }, f, ensure_ascii=False, indent=2)
    print(f"[build] 合并完成：新增机翻 {new} 条，累计带机翻 {total_mt} 条"
          f"（清理 {removed} 个分片文件）")


def main():
    ap = argparse.ArgumentParser(description="构建 tags_merged.csv 并离线机翻全部标签")
    ap.add_argument("--limit", type=int, default=0, help="本次最多翻译多少条（0=不限）")
    ap.add_argument("--min-count", type=int, default=0, help="只翻译出现次数 >= 该值的标签")
    ap.add_argument("--workers", type=int, default=1,
                    help="单进程内多线程并行实例数（Windows 下 >1 易因 OpenMP 死锁，建议用 --shard 多进程）")
    ap.add_argument("--shard", default="",
                    help='多进程分片，格式 "i/N"：本进程只翻第 i 份（0 起），结果写 _mt_shard_i.json')
    ap.add_argument("--merge-only", action="store_true",
                    help="不翻译，只把 _mt_shard_*.json 合并进 tags_merged.csv 后清理")
    args = ap.parse_args()

    if args.merge_only:
        merge_only()
        return

    # 分片解析提前：分片进程只写自己的 shard json，绝不碰 tags_merged.csv
    shard_i = shard_n = 0
    if args.shard:
        try:
            a, b = args.shard.split("/")
            shard_i, shard_n = int(a), int(b)
            if not (0 <= shard_i < shard_n):
                raise ValueError
        except ValueError:
            print(f"[build] --shard 参数无效：{args.shard}（应为 \"i/N\"，0 <= i < N）")
            return

    print("[build] 合并三来源数据…")
    shop.load_tags()
    tags = sorted(shop._TAGS, key=lambda t: -t["c"])

    # 断点续跑：继承已有 tags_merged.csv 里的机翻结果
    old_mt = {}
    if os.path.isfile(MERGED):
        with open(MERGED, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                mtv = (r.get("zh_mt") or "").strip()
                if mtv:
                    old_mt[(r.get("name") or "").strip()] = mtv

    rows = []
    for t in tags:
        rows.append({
            "name": t["n"],
            "count": t["c"],
            "cat": t["cat"],
            "sub": t["sub"],
            "zh": t["zh"],
            "zh_mt": old_mt.get(t["n"], ""),
            "aliases": "|".join(t["a"]),
            "sources": ",".join(x for x, k in (("db", t["db"]), ("v2", t["v2"]), ("v3", t["v3"])) if k),
            "note": t.get("note", ""),
        })
    if shard_n:
        print(f"[build] 分片 {shard_i}/{shard_n} 模式：不写 tags_merged.csv（由 --merge-only 统一合并）")
    else:
        write_csv(rows)
        print(f"[build] tags_merged.csv 已写出 {len(rows)} 行（{HERE}）")

    # 翻译目标：所有还没机翻过的标签（有人工中文的也翻，与 zh 并列展示）
    targets = [r for r in rows
               if not r["zh_mt"]
               and r["count"] >= args.min_count
               and re.search(r"[A-Za-z]{2,}", r["name"])]
    if args.limit:
        targets = targets[:args.limit]

    if shard_n:
        targets = [r for k, r in enumerate(targets) if k % shard_n == shard_i]
        print(f"[build] 分片 {shard_i}/{shard_n}：本进程待机翻 {len(targets)} 条")
    else:
        print(f"[build] 待机翻 {len(targets)} 条（全部未翻标签，按热度降序）")
    if not targets:
        return

    if not os.path.isfile(MODEL):
        print(f"[build] 找不到翻译模型：{MODEL}")
        return
    from llama_cpp import Llama

    shard_file = os.path.join(HERE, f"_mt_shard_{shard_i}.json")

    def save_shard():
        """shard 模式落盘：只把本分片新翻的结果写进独立 json，不碰 CSV"""
        tmp = shard_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({r["name"]: r["zh_mt"] for r in targets if r["zh_mt"]},
                      f, ensure_ascii=False)
        os.replace(tmp, shard_file)

    workers = max(1, args.workers)
    if workers > 1:
        print(f"[build] 警告：--workers {workers} 单进程多线程（Windows 下易死锁，出问题请改用 --shard 多进程）")
    print(f"[build] 加载 HY-MT 翻译模型…")

    jobs = list(targets)
    ji = [0]                 # 共享任务游标：worker 各自领活，天然不重复
    ji_lock = threading.Lock()
    stat = {"done": 0, "fail": 0}
    stat_lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()

    def worker():
        # 每线程一份独立模型实例：各自拥有独立上下文
        llm = Llama(model_path=MODEL, n_gpu_layers=-1, n_ctx=512, verbose=False, n_threads=4)
        try:
            while not stop.is_set():
                with ji_lock:
                    idx = ji[0]
                    if idx >= len(jobs):
                        return
                    ji[0] = idx + 1
                r = jobs[idx]
                try:
                    r["zh_mt"] = translate_one(llm, r["name"])
                except Exception as e:
                    with stat_lock:
                        stat["fail"] += 1
                    print(f"\n[build] 翻译失败 {r['name']}: {e}")
                with stat_lock:
                    stat["done"] += 1
        finally:
            del llm

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    try:
        for t in threads:
            t.start()
        last_save, last_print = time.time(), 0
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            d = stat["done"]
            if d >= last_print + 10:
                rate = d / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - d) / max(rate, 1e-6)
                print(f"\r[build] {d}/{len(jobs)}  {rate:.1f} 条/秒  剩余约 {eta / 60:.0f} 分钟   ",
                      end="", flush=True)
                last_print = d
            if d and time.time() - last_save > 15:   # 定期落盘，中断不丢进度
                if shard_n:
                    save_shard()
                else:
                    write_csv(rows)
                last_save = time.time()
    except KeyboardInterrupt:
        print("\n[build] 手动中断，停止任务并落盘…")
        stop.set()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
        if shard_n:
            save_shard()
            done_mt = sum(1 for r in targets if r["zh_mt"])
            print(f"\n[build] 分片 {shard_i} 落盘：{done_mt}/{len(targets)} 条 → {os.path.basename(shard_file)}"
                  f"（全部进程结束后请运行 --merge-only 合并）")
        else:
            write_csv(rows)
            with open(PROG, "w", encoding="utf-8") as f:
                json.dump({
                    "translated": sum(1 for r in rows if r["zh_mt"]),
                    "total_rows": len(rows),
                    "last_run": stat["done"],
                    "failed": stat["fail"],
                    "workers": workers,
                    "ts": time.time(),
                }, f, ensure_ascii=False, indent=2)
        gc.collect()
    if not shard_n:
        total_mt = sum(1 for r in rows if r["zh_mt"])
        print(f"\n[build] 完成：本次翻 {stat['done'] - stat['fail']} 条（失败 {stat['fail']}），"
              f"累计带机翻 {total_mt} 条")


if __name__ == "__main__":
    main()
