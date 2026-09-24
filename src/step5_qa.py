# -*- coding: utf-8 -*-
"""Step 5：混合检索问答引擎。

- HybridRetriever：Qwen3-Embedding-0.6B 向量检索 + BM25 倒排检索，RRF 融合，
  支持公司名提及加权；返回带完整元数据（公司/章节/页码）的召回块。
- AnswerEngine：本地 Qwen3-1.7B，严格依据召回证据生成答案，输出 [n] 引用标记。
- answer_question()：一站式接口，供问答页面（app.py）与评测（eval）复用。

CLI 调试：
    python src/step5_qa.py "寒武纪2025年营业收入是多少？" --k 8
    python src/step5_qa.py "对比海光信息与寒武纪的研发投入" --k 16 --retrieve-only
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402
from src.step4_build_index import EMB_MODEL_DIR, QUERY_TASK, bm25_search  # noqa: E402

LLM_DIR = common.DATA_DIR / "models" / "Qwen3-1.7B"

RRF_K = 60          # RRF 平滑参数
K_EACH = 50         # 每路召回的候选数


# ================================================================ 检索
class HybridRetriever:
    """向量 + BM25 混合检索器。"""

    def __init__(self, device: str = "cuda", emb_device: str | None = None):
        self.chunks: list[dict] = common.read_jsonl(common.CHUNKS_PATH)
        if not self.chunks:
            raise RuntimeError("chunks.jsonl 为空，请先运行 step3 / step4")
        self.emb = np.load(common.INDEX_DIR / "embeddings.npy")  # float16, L2 已归一化
        if self.emb.shape[0] != len(self.chunks):
            raise RuntimeError(f"索引行数 {self.emb.shape[0]} 与 chunks {len(self.chunks)} 不一致")
        with open(common.INDEX_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.emb_device = emb_device or device
        self._tok = None
        self._model = None
        cfg = common.load_config()
        self.company_names = [c["name"] for c in cfg["companies"]]

    # ---------- 查询向量化 ----------
    def _ensure_embedder(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(EMB_MODEL_DIR), padding_side="left")
        self._model = AutoModel.from_pretrained(str(EMB_MODEL_DIR), dtype=torch.float16)
        self._model = self._model.to(self.emb_device).eval()

    def embed_query(self, query: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._ensure_embedder()
        text = f"Instruct: {QUERY_TASK}\nQuery: {query}"
        with torch.no_grad():
            enc = self._tok([text], padding=True, truncation=True, max_length=1024,
                            return_tensors="pt").to(self.emb_device)
            out = self._model(**enc)
            mask = enc["attention_mask"]
            seq_len = mask.sum(dim=1) - 1
            vec = out.last_hidden_state[torch.arange(1, device=self.emb_device), seq_len].float()
            vec = F.normalize(vec, p=2, dim=1)
        return vec[0].cpu().numpy().astype(np.float32)

    # ---------- 混合检索 ----------
    def search(self, query: str, k: int = 8, k_each: int = K_EACH,
               rrf_k: int = RRF_K, company_boost: float = 0.30) -> list[dict]:
        # 1) 向量召回
        qv = self.embed_query(query)
        sims = self.emb.astype(np.float32) @ qv
        vec_top = np.argsort(-sims)[:k_each]

        # 2) BM25 召回
        bm = bm25_search(self.bm25, query, top_k=k_each)

        # 3) RRF 融合
        fused: dict[int, float] = {}
        vec_rank: dict[int, int] = {}
        bm_rank: dict[int, int] = {}
        for rank, idx in enumerate(vec_top):
            i = int(idx)
            vec_rank[i] = rank
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (i, _s) in enumerate(bm):
            bm_rank[i] = rank
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)

        # 4) 公司名提及加权（问题中出现的公司，其块获得小幅加成）
        mentioned = [name for name in self.company_names if name in query]
        if mentioned:
            for i in fused:
                if self.chunks[i]["company"] in mentioned:
                    fused[i] += company_boost / (rrf_k + 1)

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        out = []
        for i, score in ranked:
            c = self.chunks[i]
            out.append({
                "chunk_id": c["chunk_id"], "score": round(float(score), 6),
                "vec_rank": vec_rank.get(i), "bm25_rank": bm_rank.get(i),
                "vec_sim": round(float(sims[i]), 4) if i in set(vec_rank) else None,
                "company": c["company"], "code": c["code"], "doc_id": c["doc_id"],
                "report_label": c["report_label"], "section": c["section"],
                "page": c["page"], "type": c["type"], "text": c["text"],
                "html": c.get("html", ""), "n_chars": c["n_chars"],
                "mentioned_company": c["company"] in mentioned,
            })
        return out


# ================================================================ 生成
SYSTEM_PROMPT = (
    "你是一名严谨的金融财报问答助手。请严格依据用户提供的【资料】回答问题：\n"
    "1. 只使用【资料】中明确出现的信息，禁止编造数字、名称或引入资料之外的知识；\n"
    "2. 引用数字时保持与资料一致（含单位与口径），不要换算或推测；\n"
    "3. 在每处引用的事实后标注来源编号，如 [1]、[2]（编号对应【资料】中各条目）；\n"
    "4. 若【资料】不足以回答，请直接说明“提供的资料中未找到相关信息”，不要强行作答；\n"
    "5. 回答简洁：先给结论，再补充必要细节；如涉及多家公司对比，分条列出。"
)


class AnswerEngine:
    """本地 Qwen3-1.7B 答案生成器。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._tok = None
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(LLM_DIR))
        self._model = AutoModelForCausalLM.from_pretrained(str(LLM_DIR), dtype=torch.float16)
        self._model = self._model.to(self.device).eval()

    def generate(self, question: str, evidence: list[dict], max_new_tokens: int = 640,
                 max_ctx_chars: int = 1800) -> str:
        import torch

        self._ensure()
        parts = []
        for i, e in enumerate(evidence):
            text = (e["text"] or "")[:max_ctx_chars]
            parts.append(f"[{i + 1}] 《{e['company']}》{e['report_label']}｜{e['section']}｜第{e['page']}页\n{text}")
        user = f"【问题】{question}\n\n【资料】\n" + "\n\n".join(parts)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        try:
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            inputs = self._tok([prompt], return_tensors="pt").to(self.device)
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, eos_token_id=self._tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


# ================================================================ 一站式
_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def parse_citations(answer: str, evidence: list[dict]) -> list[dict]:
    """从答案文本解析 [n] 引用，返回被引用的证据列表（含来源元数据）。"""
    used = sorted({int(m) for m in _CITE_RE.findall(answer)})
    cites = []
    for n in used:
        if 1 <= n <= len(evidence):
            e = evidence[n - 1]
            cites.append({
                "idx": n, "chunk_id": e["chunk_id"], "company": e["company"],
                "report_label": e["report_label"], "section": e["section"],
                "page": e["page"], "type": e["type"],
                "snippet": (e["text"] or "")[:180],
            })
    return cites


def answer_question(question: str, retriever: HybridRetriever, generator: AnswerEngine | None,
                    k: int = 8, retrieve_only: bool = False) -> dict:
    evidence = retriever.search(question, k=k)
    result = {"question": question, "k": k, "evidence": evidence, "answer": None, "citations": []}
    if retrieve_only or generator is None:
        return result
    answer = generator.generate(question, evidence)
    result["answer"] = answer
    result["citations"] = parse_citations(answer, evidence)
    return result


# ================================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="财报知识库问答（命令行）")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    retriever = HybridRetriever(device=args.device)
    gen = None if args.retrieve_only else AnswerEngine(device=args.device)
    res = answer_question(args.question, retriever, gen, k=args.k, retrieve_only=args.retrieve_only)

    print("=" * 80)
    print("问题:", res["question"])
    print("-" * 80)
    if res["answer"]:
        print(res["answer"])
        print("-" * 80)
        print("引用出处:")
        for c in res["citations"]:
            print(f"  [{c['idx']}] {c['company']}《{c['report_label']}》 {c['section']} 第{c['page']}页")
    print("-" * 80)
    print("召回块:")
    for i, e in enumerate(res["evidence"], 1):
        print(f"  [{i}] ({e['score']:.5f} 向量#{e['vec_rank']} BM25#{e['bm25_rank']}) "
              f"{e['company']} 第{e['page']}页 {e['type']} :: {(e['text'] or '')[:70]}")


if __name__ == "__main__":
    main()
