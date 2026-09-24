# -*- coding: utf-8 -*-
"""评测脚本：对 10 道测试题逐题检索 + 生成回答，记录召回块与回答。

输出：
    eval/results/eval_raw.json     完整原始记录（含召回块元数据）
    eval/results/eval_raw.md       可读记录（人工评估用）
    eval/results/eval_results.csv  评测骨架（correct / error_reason 列为空，人工填写）

用法：
    python eval/run_eval.py [--device cuda] [--only q01,q02] [--retrieve-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from src import common  # noqa: E402
from src.step5_qa import AnswerEngine, HybridRetriever, answer_question  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--k-override", type=int, default=0)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    qs = yaml.safe_load((Path(__file__).parent / "test_questions.yml").read_text(encoding="utf-8"))["questions"]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        qs = [q for q in qs if q["id"] in keep]

    retriever = HybridRetriever(device=args.device)
    gen = None if args.retrieve_only else AnswerEngine()

    records = []
    for q in qs:
        k = args.k_override or q.get("k", 8)
        t0 = time.time()
        res = answer_question(q["question"], retriever, gen, k=k, retrieve_only=args.retrieve_only)
        res["elapsed_s"] = round(time.time() - t0, 1)
        rec = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "k": k, "expected_doc": q.get("expected_doc", ""),
            "ground_truth": q.get("ground_truth", ""),
            "answer": res["answer"], "citations": res["citations"],
            "elapsed_s": res["elapsed_s"],
            "evidence": [
                {kk: e[kk] for kk in ("chunk_id", "company", "doc_id", "report_label", "section",
                                       "page", "type", "score", "vec_rank", "bm25_rank")}
                for e in res["evidence"]
            ],
        }
        records.append(rec)
        print(f"[{q['id']}] {res['elapsed_s']}s  答案长度 {len(res['answer'] or '')}" +
              (f"  usage={gen.last_stats}" if gen is not None and gen.last_stats else ""))
        if res["answer"]:
            print("   ", res["answer"][:160].replace("\n", " "))

    (RESULTS_DIR / "eval_raw.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可读 Markdown
    lines = ["# 评测原始记录\n"]
    for r in records:
        lines.append(f"\n## {r['id']}（{r['type']}）\n")
        lines.append(f"**问题**：{r['question']}\n")
        lines.append(f"**参考答案要点**：{r['ground_truth']}\n")
        lines.append(f"**模型回答**：\n\n{r['answer'] or '（未生成）'}\n")
        if r["citations"]:
            lines.append("**引用出处**：\n")
            for c in r["citations"]:
                lines.append(f"- [{c['idx']}] {c['company']}《{c['report_label']}》 {c['section']} 第{c['page']}页")
        lines.append(f"\n**召回块（k={r['k']}）**：\n")
        for i, e in enumerate(r["evidence"], 1):
            lines.append(f"{i}. `{e['chunk_id']}` {e['company']} 第{e['page']}页 {e['type']} "
                         f"score={e['score']}（v#{e['vec_rank']} b#{e['bm25_rank']}）")
        lines.append("\n---")
    (RESULTS_DIR / "eval_raw.md").write_text("\n".join(lines), encoding="utf-8")

    # CSV 骨架（人工填写 correct / error_reason）
    with open(RESULTS_DIR / "eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "k", "top1_chunk", "top1_company", "top1_page",
                    "answer_correct", "error_reason", "notes"])
        for r in records:
            ev = r["evidence"][0] if r["evidence"] else {}
            w.writerow([r["id"], r["type"], r["question"], r["k"],
                        ev.get("chunk_id", ""), ev.get("company", ""), ev.get("page", ""),
                        "", "", ""])
    print(f"\n完成，共 {len(records)} 题 → {RESULTS_DIR}")


if __name__ == "__main__":
    main()
