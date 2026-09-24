# -*- coding: utf-8 -*-
"""多连接分片下载 ModelScope 模型文件（单连接被限速时的加速方案）。

用法：
    # 下载整个仓库的全部文件
    python tools/fast_download.py Qwen/Qwen3-1.7B --dest data/models/Qwen3-1.7B --conns 8

    # 只下指定文件
    python tools/fast_download.py Qwen/Qwen3-1.7B --dest data/models/Qwen3-1.7B \
        --include model-00001-of-00002.safetensors,model-00002-of-00002.safetensors
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_LIST = "https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Root="
API_FILE = "https://www.modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) fast-downloader/1.0"}


def list_files(model: str) -> list[dict]:
    r = requests.get(API_LIST.format(model=model), headers=UA, timeout=60)
    js = r.json()
    files = (js.get("Data") or {}).get("Files") or []
    return [{"path": f["Path"], "size": f["Size"]} for f in files]


def download_file(model: str, fmeta: dict, dest_dir: Path, conns: int = 8) -> None:
    path, size = fmeta["path"], int(fmeta["size"])
    dest = dest_dir / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  [skip] {path} 已存在")
        return
    url = API_FILE.format(model=model, path=path.replace("/", "%2F"))
    tmp_dir = dest_dir / f".parts_{dest.name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk = max(4 << 20, size // (conns * 2))  # 4MB 起，约 2 倍连接数的分片
    ranges = [(a, min(a + chunk - 1, size - 1)) for a in range(0, size, chunk)]
    print(f"  [dl] {path}: {size/1e6:.1f}MB，{len(ranges)} 个分片，{conns} 连接")

    lock = threading.Lock()
    progress = {"done": 0, "t0": time.time()}
    for p in tmp_dir.glob("*.part"):
        progress["done"] += p.stat().st_size

    def fetch(idx: int, start: int, end: int) -> None:
        part = tmp_dir / f"{idx:04d}.part"
        want = end - start + 1
        if part.exists() and part.stat().st_size == want:
            return
        last_err = None
        for attempt in range(4):
            try:
                headers = {**UA, "Range": f"bytes={start}-{end}"}
                with requests.get(url, headers=headers, stream=True, timeout=90) as r:
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    got = 0
                    with open(part, "wb") as f:
                        for ch in r.iter_content(1 << 16):
                            if not ch:
                                continue
                            f.write(ch)
                            got += len(ch)
                            with lock:
                                progress["done"] += len(ch)
                if got != want:
                    # 服务器忽略 Range 时返回整个文件，取前 want 字节即可
                    if got > want:
                        with open(part, "r+b") as f:
                            f.truncate(want)
                        with lock:
                            progress["done"] -= got - want
                        return
                    raise RuntimeError(f"part size {got} != {want}")
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"分片 {idx} 失败: {last_err}")

    with ThreadPoolExecutor(max_workers=conns) as ex:
        futs = [ex.submit(fetch, i, a, b) for i, (a, b) in enumerate(ranges)]
        last_report = time.time()
        for fut in as_completed(futs):
            fut.result()
            if time.time() - last_report > 15:
                sp = progress["done"] / max(1e-9, time.time() - progress["t0"]) / 1e6
                print(f"    {progress['done']/size*100:5.1f}%  {sp:.2f}MB/s")
                last_report = time.time()

    # 合并
    with open(dest, "wb") as out:
        for i in range(len(ranges)):
            part = tmp_dir / f"{i:04d}.part"
            with open(part, "rb") as pf:
                out.write(pf.read())
    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"合并后大小不符: {got} != {size}")
    for p in tmp_dir.glob("*.part"):
        p.unlink()
    tmp_dir.rmdir()
    print(f"  [ok] {path} ({got/1e6:.1f}MB)，用时 {(time.time()-progress['t0'])/60:.1f} 分钟")


def main() -> None:
    ap = argparse.ArgumentParser(description="ModelScope 多连接分片下载")
    ap.add_argument("model", help="如 Qwen/Qwen3-1.7B")
    ap.add_argument("--dest", required=True, help="本地目录")
    ap.add_argument("--include", default="", help="仅下载指定文件（逗号分隔，文件路径）")
    ap.add_argument("--conns", type=int, default=8)
    args = ap.parse_args()

    dest_dir = Path(args.dest)
    files = list_files(args.model)
    if args.include:
        keep = {x.strip() for x in args.include.split(",") if x.strip()}
        files = [f for f in files if f["path"] in keep]
    total = sum(f["size"] for f in files)
    print(f"模型 {args.model}: {len(files)} 个文件，合计 {total/1e6:.1f}MB")
    for f in files:
        download_file(args.model, f, dest_dir, conns=args.conns)


if __name__ == "__main__":
    main()
