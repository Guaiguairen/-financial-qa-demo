# -*- coding: utf-8 -*-
"""审计工具：在 chunks.jsonl 中按关键词检索，输出块元数据与原文片段。

用法（从项目根目录运行）：python tools/dev/grep_chunks.py <关键词> [limit=8] [公司名] [报告标签]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import common  # noqa: E402


def main() -> None:
    kw = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    comp = sys.argv[3] if len(sys.argv) > 3 else ""
    label = sys.argv[4] if len(sys.argv) > 4 else ""
    chunks = common.read_jsonl(common.CHUNKS_PATH)
    hits = 0
    for c in chunks:
        if comp and comp not in c["company"]:
            continue
        if label and label not in c["report_label"]:
            continue
        text = c["text"]
        if kw in text:
            hits += 1
            if hits > limit:
                break
            idx = text.find(kw)
            lo = max(0, idx - 120)
            hi = min(len(text), idx + 220)
            print("-" * 96)
            print(f"[{c['chunk_id']}] {c['company']}《{c['report_label']}》 | {c['section']} | "
                  f"第{c['page']}页 | {c['type']}")
            print("   ...", text[lo:hi].replace("\n", " ⏎ "), "...")
    print(f"\n共命中 {hits} 块（显示 {min(hits, limit)}）")


if __name__ == "__main__":
    main()
