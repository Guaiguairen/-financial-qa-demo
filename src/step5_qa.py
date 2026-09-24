# -*- coding: utf-8 -*-
"""Step 5：混合检索问答引擎。

- HybridRetriever：Qwen3-Embedding-0.6B 向量检索 + BM25 倒排检索，RRF 融合，
  支持公司名提及加权；返回带完整元数据（公司/章节/页码）的召回块。
- AnswerEngine：基于 DeepSeek API（OpenAI 兼容接口）生成答案，严格依据召回证据，
  输出 [n] 引用标记。
- answer_question()：一站式接口，供问答页面（app.py）与评测（eval）复用。

API Key 配置（二选一）：
    $env:DEEPSEEK_API_KEY = "sk-..."            # 环境变量（推荐）
    data/config/deepseek.json  {"api_key": "sk-..."}}   # 本地配置文件
可选环境变量：DEEPSEEK_MODEL（默认 deepseek-chat）、DEEPSEEK_BASE_URL。

CLI 调试：
    python src/step5_qa.py "寒武纪2025年营业收入是多少？" --k 8
    python src/step5_qa.py "对比海光信息与寒武纪的研发投入" --k 16 --retrieve-only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402
from src.step4_build_index import EMB_MODEL_DIR, QUERY_TASK, bm25_search  # noqa: E402

RRF_K = 60          # RRF 平滑参数
K_EACH = 50         # 每路召回的候选数

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_KEY_FILE = common.DATA_DIR / "config" / "deepseek.json"


def load_deepseek_key() -> str:
    """读取 DeepSeek API Key：环境变量优先，其次 data/config/deepseek.json。"""
    key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key and DEEPSEEK_KEY_FILE.exists():
        try:
            key = (json.loads(DEEPSEEK_KEY_FILE.read_text(encoding="utf-8")).get("api_key") or "").strip()
        except Exception:  # noqa: BLE001
            key = ""
    return key


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

        dtype = torch.float16 if self.emb_device.startswith("cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(str(EMB_MODEL_DIR), padding_side="left")
        self._model = AutoModel.from_pretrained(str(EMB_MODEL_DIR), dtype=dtype)
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

        # 5) 报告期间消歧：问题未指明“半年/中期”时默认偏好年度报告，反之亦然
        semi_words = ("半年", "中期", "中报", "上半", "H1", "半期")
        prefer = "semi" if any(w in query for w in semi_words) else "annual"
        for i in fused:
            if self.chunks[i]["report_type"] == prefer:
                fused[i] += 0.5 / (rrf_k + 1)

        # 6) 叙述型对比数据加权：含"营业收入/净利润 + 同比/较上年"的正文块
        #    （管理层讨论中的表述口径通常比报表附表更直接，避免单表数字歧义）
        for i in fused:
            t = self.chunks[i]["text"] or ""
            if ("营业收入" in t or "净利润" in t) and ("同比" in t or "较上年" in t):
                fused[i] += 0.2 / (rrf_k + 1)

        # 7) "归母"类问题的口径定向：问题点名归母/上市公司股东时，
        #    加权含"归属于上市公司股东的净利润"字样的块，避开母公司/净利润总额混淆
        if any(w in query for w in ("归母", "归属于上市公司", "归属于母公司")):
            for i in fused:
                if "归属于上市公司股东的净利润" in (self.chunks[i]["text"] or ""):
                    fused[i] += 0.35 / (rrf_k + 1)

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


# ================================================================ 生成（DeepSeek API）
SYSTEM_PROMPT = (
    "你是一名严谨的金融财报问答助手。请严格依据用户提供的【资料】回答问题：\n"
    "1. 只使用【资料】中明确出现的信息，禁止编造数字、名称或引入资料之外的知识；\n"
    "2. 引用数字时保持与资料一致（含单位与口径），不要换算或推测；\n"
    "3. 在每处引用的事实后标注来源编号，如 [1]、[2]（编号对应【资料】中各条目）；\n"
    "4. 若【资料】不足以回答，请直接说明“提供的资料中未找到相关信息”，不要强行作答；\n"
    "5. 涉及公司整体层面的财务数据（营收、净利润等）时，优先采用管理层讨论与分析中的表述口径（通常为合并口径）并注明；若仅有母公司口径数据，必须明确说明“（母公司口径）”；\n"
    "6. 回答简洁：先给结论，再补充必要细节；如涉及多家公司对比，分条列出。"
)


class AnswerEngine:
    """基于 DeepSeek API 的答案生成器（OpenAI 兼容接口）。

    证据预算：默认最多 16 块、合计 ≤12000 字——API 上下文充裕，
    该预算主要用于控制成本与信噪比，可随 API 参数调整。
    """

    def __init__(self, model: str | None = None, timeout: int = 180):
        self.api_key = load_deepseek_key()
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE).rstrip("/")
        self.timeout = timeout
        self.last_stats: dict | None = None
        self._session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, question: str, evidence: list[dict], max_new_tokens: int = 1200,
                 total_ctx_chars: int = 12000, max_chunks: int = 16) -> str:
        if not self.api_key:
            raise RuntimeError(
                "未找到 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，"
                f"或写入 {DEEPSEEK_KEY_FILE}（内容：{{\"api_key\": \"sk-...\"}}）")
        evidence = evidence[:max_chunks]
        per = max(400, total_ctx_chars // max(1, len(evidence)))
        parts = []
        for i, e in enumerate(evidence):
            text = (e["text"] or "")[:per]
            parts.append(f"[{i + 1}] 《{e['company']}》{e['report_label']}｜{e['section']}｜第{e['page']}页\n{text}")
        user = f"【问题】{question}\n\n【资料】\n" + "\n\n".join(parts)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(3):
            t0 = time.time()
            try:
                r = self._session.post(f"{self.base_url}/chat/completions",
                                       json=payload, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 401:
                raise RuntimeError("DeepSeek API 鉴权失败（401）：请检查 API Key 是否正确或已过期")
            if r.status_code == 402:
                raise RuntimeError("DeepSeek API 余额不足（402）：请前往 DeepSeek 平台充值")
            if r.status_code in (429, 500, 502, 503):
                last_err = RuntimeError(f"DeepSeek API 暂时不可用（HTTP {r.status_code}）")
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            usage = data.get("usage") or {}
            self.last_stats = {
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "seconds": round(time.time() - t0, 2),
            }
            return text
        raise RuntimeError(f"DeepSeek API 调用失败（已重试）：{last_err}")


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
    gen = None if args.retrieve_only else AnswerEngine()
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
        if gen is not None and gen.last_stats:
            print("-" * 80)
            print("生成统计:", gen.last_stats)
    print("-" * 80)
    print("召回块:")
    for i, e in enumerate(res["evidence"], 1):
        print(f"  [{i}] ({e['score']:.5f} 向量#{e['vec_rank']} BM25#{e['bm25_rank']}) "
              f"{e['company']} 第{e['page']}页 {e['type']} :: {(e['text'] or '')[:70]}")


if __name__ == "__main__":
    main()
