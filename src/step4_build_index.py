# -*- coding: utf-8 -*-
"""Step 4：构建混合检索索引（Qwen3-Embedding 向量索引 + BM25 倒排索引）。

输入
----
data/chunks/chunks.jsonl（step3 产出）

输出（data/index/）
----
- embeddings.npy      float16 N×D，已 L2 归一化
- chunk_ids.json      向量行号 → chunk_id 映射（与 embeddings 行序一致）
- bm25.pkl            BM25 倒排索引（jieba 分词；含词表、postings、idf、文档长度）
- index_meta.json     模型信息、维度、数量、构建时间

用法
----
    python src/step4_build_index.py [--device cuda|cpu] [--batch-size 32] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402

EMB_MODEL_DIR = common.DATA_DIR / "models" / "Qwen3-Embedding-0.6B"
IDX_DIR = common.INDEX_DIR

# 查询侧指令前缀（Qwen3-Embedding 官方用法：查询加任务指令，文档不加）
QUERY_TASK = (
    "Given a user's question about Chinese A-share computing-chip companies' "
    "annual and semi-annual reports, retrieve passages that can answer the question"
)


def build_doc_text(c: dict, max_chars: int = 1600) -> str:
    """构造用于向量化的文档文本：轻量上下文头 + 正文。"""
    header = f"{c['company']}《{c['report_label']}》{c['section']}"
    body = c["text"]
    return f"{header}\n{body}"[:max_chars]


# ---------------------------------------------------------------- 向量
def build_embeddings(chunks: list[dict], device: str, batch_size: int, logger,
                     max_length: int = 1024) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    logger.info("加载嵌入模型：%s（device=%s）", EMB_MODEL_DIR, device)
    tok = AutoTokenizer.from_pretrained(str(EMB_MODEL_DIR), padding_side="left")
    model = AutoModel.from_pretrained(str(EMB_MODEL_DIR), dtype=torch.float16)
    model = model.to(device).eval()

    texts = [build_doc_text(c) for c in chunks]
    dim = None
    all_vecs: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"]
            # Qwen3-Embedding：取每条序列最后一个有效 token 的隐状态
            seq_len = mask.sum(dim=1) - 1
            vec = last[torch.arange(last.size(0), device=device), seq_len].float()
            vec = F.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.cpu().numpy().astype(np.float16))
            if dim is None:
                dim = vec.shape[1]
            done = min(i + batch_size, len(texts))
            if done % (batch_size * 10) == 0 or done == len(texts):
                speed = done / max(1e-9, time.time() - t0)
                logger.info("  向量化 %d/%d（%.1f 条/秒）", done, len(texts), speed)
    emb = np.vstack(all_vecs)
    logger.info("向量矩阵：%s，耗时 %.1fs", emb.shape, time.time() - t0)
    return emb


# ---------------------------------------------------------------- BM25
PUNCT = set("，。；：！？、（）《》“”‘’【】…—·,.;:!?()<>\"'`~@#$%^&*+=|\\/[]{}＿ \t\n\r")


def tokenize(text: str) -> list[str]:
    import jieba

    toks = []
    for t in jieba.lcut(text, cut_all=False):
        t = t.strip().lower()
        if not t or t in PUNCT:
            continue
        if all(ch in PUNCT for ch in t):
            continue
        toks.append(t)
    return toks


def build_bm25(chunks: list[dict], logger) -> dict:
    t0 = time.time()
    vocab: dict[str, int] = {}
    postings: dict[int, list[tuple[int, int]]] = defaultdict(list)  # term_id -> [(doc_id, tf)]
    doc_len = np.zeros(len(chunks), dtype=np.float32)

    for di, c in enumerate(chunks):
        # 公司/章节/报告期重复加入，增强字段权重
        text = (
            c["text"]
            + "\n" + c["company"] * 2 + "\n" + c["section"] * 2 + "\n" + c["report_label"]
        )
        toks = tokenize(text)
        doc_len[di] = len(toks)
        tf: dict[int, int] = defaultdict(int)
        for t in toks:
            tid = vocab.get(t)
            if tid is None:
                tid = len(vocab)
                vocab[t] = tid
            tf[tid] += 1
        for tid, f in tf.items():
            postings[tid].append((di, f))

    n = len(chunks)
    df = np.zeros(len(vocab), dtype=np.float32)
    for tid, plist in postings.items():
        df[tid] = len(plist)
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    avg_len = float(doc_len.mean()) if n else 0.0

    index = {
        "vocab": vocab,
        "postings": {int(k): v for k, v in postings.items()},
        "idf": idf,
        "doc_len": doc_len,
        "avg_len": avg_len,
        "n_docs": n,
        "k1": 1.5,
        "b": 0.75,
    }
    logger.info("BM25 索引：%d 文档，词表 %d，均长 %.1f，耗时 %.1fs",
                n, len(vocab), avg_len, time.time() - t0)
    return index


def bm25_search(index: dict, query: str, top_k: int = 60) -> list[tuple[int, float]]:
    """BM25 检索，返回 [(doc_idx, score)] 降序。"""
    k1, b = index["k1"], index["b"]
    doc_len, avg_len = index["doc_len"], index["avg_len"] or 1.0
    scores: dict[int, float] = defaultdict(float)
    for t in set(tokenize(query)):
        tid = index["vocab"].get(t)
        if tid is None:
            continue
        idf = index["idf"][tid]
        for di, tf in index["postings"].get(tid, []):
            denom = tf + k1 * (1 - b + b * doc_len[di] / avg_len)
            scores[di] += idf * tf * (k1 + 1) / denom
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="构建混合检索索引")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024, help="嵌入截断长度（token）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试用）")
    args = ap.parse_args()

    common.ensure_dirs()
    logger = common.setup_logger("step4", "step4_index.log")
    chunks = common.read_jsonl(common.CHUNKS_PATH)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        logger.error("chunks.jsonl 为空，先运行 step3")
        sys.exit(1)
    logger.info("载入 %d 个 chunk", len(chunks))

    emb = build_embeddings(chunks, args.device, args.batch_size, logger, max_length=args.max_length)
    np.save(IDX_DIR / "embeddings.npy", emb)
    with open(IDX_DIR / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump([c["chunk_id"] for c in chunks], f, ensure_ascii=False)

    bm25 = build_bm25(chunks, logger)
    with open(IDX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    meta = {
        "embedding_model": str(EMB_MODEL_DIR),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "n_chunks": len(chunks),
        "query_task": QUERY_TASK,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(IDX_DIR / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("索引构建完成：%s", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
