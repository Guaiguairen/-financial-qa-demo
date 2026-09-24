# -*- coding: utf-8 -*-
"""问答页面（Flask）：混合检索 + DeepSeek API 生成，答案标注引用出处。

启动：
    python src/app.py --port 8000
页面：
    http://127.0.0.1:8000/
前置：设置环境变量 DEEPSEEK_API_KEY（或写入 data/config/deepseek.json）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402
from src.step5_qa import AnswerEngine, HybridRetriever, answer_question  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"

app = Flask(__name__, static_folder=None)
_state: dict = {}
_lock = threading.Lock()  # 检索/生成串行化，避免并发争用


def get_state(device: str = "cuda") -> dict:
    if not _state:
        t0 = time.time()
        _state["retriever"] = HybridRetriever(device=device)  # 嵌入模型用 GPU
        _state["generator"] = AnswerEngine()                  # DeepSeek API
        _state["load_s"] = round(time.time() - t0, 1)
        print(f"[app] 检索索引加载完成，用时 {_state['load_s']}s；"
              f"DeepSeek Key {'已配置' if _state['generator'].configured else '未配置'}")
    return _state


@app.route("/")
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.route("/api/status")
def api_status():
    st = get_state()
    meta_path = common.INDEX_DIR / "index_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    manifest = common.read_jsonl(common.MANIFEST_PATH)
    gen = st["generator"]
    return jsonify({
        "n_chunks": meta.get("n_chunks", len(st["retriever"].chunks)),
        "n_docs": len(manifest),
        "n_pages": sum(m.get("pages") or 0 for m in manifest),
        "n_companies": len({m["code"] for m in manifest}),
        "embedding_model": "Qwen3-Embedding-0.6B",
        "llm": f"DeepSeek（{gen.model}）" if gen.configured else "DeepSeek（未配置 Key）",
        "built_at": meta.get("built_at", ""),
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    k = int(payload.get("k") or 8)
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    st = get_state()
    t0 = time.time()
    try:
        with _lock:
            res = answer_question(question, st["retriever"], st["generator"], k=k)
    except RuntimeError as e:  # API 配置/调用错误，返回可读信息
        return jsonify({"error": str(e)}), 502
    res["elapsed_ms"] = int((time.time() - t0) * 1000)
    # 控制返回体量：证据截断
    for e in res["evidence"]:
        e["html"] = ""
        e["text"] = (e["text"] or "")[:500]
    return jsonify(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    if not args.no_warmup:
        get_state(args.device)  # 启动时预加载检索索引
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
