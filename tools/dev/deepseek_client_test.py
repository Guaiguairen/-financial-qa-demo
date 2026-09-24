# -*- coding: utf-8 -*-
"""开发工具：DeepSeek 客户端健壮性测试（本地 mock 服务器 + 真实 401 路径）。"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["DEEPSEEK_API_KEY"] = "sk-mock-test-key"
os.environ["DEEPSEEK_BASE_URL"] = "http://127.0.0.1:8123"
os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"

from src.step5_qa import AnswerEngine  # noqa: E402

RESP = {
    "id": "mock",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {
        "role": "assistant",
        "content": "根据资料[1]，寒武纪2025年营业收入为 649,719.62 万元。 [1]"},
        "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290},
}


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        ln = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(ln))
        # 校验到达服务器的请求格式
        assert req["model"] == "deepseek-chat"
        assert req["messages"][0]["role"] == "system"
        assert "[1]" in req["messages"][1]["content"]
        assert req["temperature"] == 0.0
        assert self.headers.get("Authorization", "").startswith("Bearer ")
        body = json.dumps(RESP).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静默
        pass


def main() -> None:
    print("== 1) mock 服务器路径 ==")
    srv = HTTPServer(("127.0.0.1", 8123), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ev = [{"company": "寒武纪", "report_label": "2025年年度报告", "section": "第三节",
           "page": 27, "type": "text", "text": "报告期内实现营业收入 649,719.62 万元。", "chunk_id": "t1"}]
    eng = AnswerEngine()
    print("configured:", eng.configured, "| model:", eng.model, "| base:", eng.base_url)
    out = eng.generate("寒武纪2025年营业收入是多少？", ev)
    print("reply:", out)
    print("stats:", eng.last_stats)
    srv.shutdown()

    print("== 2) 真实 API 401 路径（无效 Key） ==")
    os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
    os.environ["DEEPSEEK_API_KEY"] = "sk-invalid-key-for-test"
    eng2 = AnswerEngine()
    try:
        eng2.generate("test", ev)
        print("401 path: UNEXPECTED SUCCESS")
    except RuntimeError as e:
        print("401 path OK ->", e)

    print("== 3) 无 Key 路径 ==")
    eng3 = AnswerEngine()
    eng3.api_key = ""
    try:
        eng3.generate("test", ev)
        print("no-key path: UNEXPECTED SUCCESS")
    except RuntimeError as e:
        print("no-key path OK ->", str(e)[:60], "...")


if __name__ == "__main__":
    main()
