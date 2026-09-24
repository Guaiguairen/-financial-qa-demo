# -*- coding: utf-8 -*-
"""审计工具：查看某问题的 Top-N 召回构成（含各路排名与加权后得分）。

用法（从项目根目录运行）：python tools/dev/retrieval_probe.py "问题" [k=8]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.step5_qa import HybridRetriever  # noqa: E402


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "寒武纪2025年营业收入是多少？"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    r = HybridRetriever(device="cuda", emb_device="cpu")
    for i, e in enumerate(r.search(q, k=k), 1):
        sec = e["section"][:40]
        print(f"[{i}] {e['company']}《{e['report_label']}》| {sec} | 第{e['page']}页 | {e['type']} | "
              f"score={e['score']:.5f} v#{e['vec_rank']} b#{e['bm25_rank']}")
        print("     ", (e["text"] or "")[:90].replace("\n", " / "))


if __name__ == "__main__":
    main()
