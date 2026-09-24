# -*- coding: utf-8 -*-
"""审计工具：汇总语料与切块统计（报告用）。

用法（从项目根目录运行）：python tools/dev/corpus_stats.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import common  # noqa: E402


def main() -> None:
    stats = []
    for p in sorted(common.EXTRACT_DIR.glob("*/doc_stats.json")):
        stats.append(json.loads(p.read_text(encoding="utf-8")))
    print(f"=== 抽取统计（{len(stats)} 份）===")
    print("页面总数:", sum(s["pages"] for s in stats))
    print("内容块总数:", sum(s["blocks"] for s in stats))
    print("表格总数:", sum(s["tables"] for s in stats))
    print("档位分布:", dict(Counter(s.get("tier", "?") for s in stats)))
    print(f"解析耗时合计: {sum(s.get('mineru_cost_s', 0) for s in stats)/60:.1f} 分钟")

    chunks = common.read_jsonl(common.CHUNKS_PATH)
    print(f"\n=== 切块统计（{len(chunks)} 块）===")
    print("类型分布:", dict(Counter(c["type"] for c in chunks)))
    total_chars = sum(c["n_chars"] for c in chunks)
    print(f"总字数: {total_chars/10000:.1f} 万，平均 {total_chars//len(chunks)} 字/块")
    print("按公司:", dict(Counter(c["company"] for c in chunks)))
    no_sec = sum(1 for c in chunks if not c["section"] or c["section"] == "（正文前部）")
    print(f"无章节归属的块: {no_sec}（{no_sec/len(chunks)*100:.1f}%）")


if __name__ == "__main__":
    main()
