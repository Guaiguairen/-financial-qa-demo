# -*- coding: utf-8 -*-
"""公共工具：路径约定、配置加载、日志、JSONL 读写、HTTP 会话。

本模块被 step1~step5 及 app 共用，保证各环节路径与元数据口径一致。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "companies.yml"

DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
EXTRACT_DIR = DATA_DIR / "extracted"
CHUNK_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = DATA_DIR / "logs"

EVAL_DIR = ROOT / "eval"
DOCS_DIR = ROOT / "docs"

CHUNKS_PATH = CHUNK_DIR / "chunks.jsonl"
MANIFEST_PATH = RAW_PDF_DIR / "manifest.jsonl"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def ensure_dirs() -> None:
    for d in (RAW_PDF_DIR, EXTRACT_DIR, CHUNK_DIR, INDEX_DIR, LOG_DIR, EVAL_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 日志
def setup_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复挂 handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(LOG_DIR / logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------- 配置
def load_config(path: str | Path | None = None) -> dict:
    """读取 companies.yml。"""
    import yaml  # 局部导入，便于错误定位

    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_targets(cfg: dict):
    """展开为 (company, report_type, year) 三元组列表。

    report_type: "annual" | "semi"
    """
    targets = []
    for comp in cfg["companies"]:
        for year in cfg["periods"].get("annual", []):
            targets.append((comp, "annual", str(year)))
        for year in cfg["periods"].get("semi", []):
            targets.append((comp, "semi", str(year)))
    return targets


def period_label(report_type: str, year: str) -> str:
    return f"{year}年年度报告" if report_type == "annual" else f"{year}年半年度报告"


def doc_id(code: str, report_type: str, year: str) -> str:
    return f"{code}_{year}_{report_type}"


# ---------------------------------------------------------------- JSONL
def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- HTTP
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def get_with_retry(sess: requests.Session, url: str, *, tries: int = 4, timeout: int = 60,
                   stream: bool = False, **kw):
    last_err = None
    for i in range(tries):
        try:
            r = sess.get(url, timeout=timeout, stream=stream, **kw)
            if r.status_code == 200:
                return r
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"请求失败 {url}: {last_err}")


# ---------------------------------------------------------------- 杂项
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def strip_em(s: str) -> str:
    return (s or "").replace("<em>", "").replace("</em>", "").strip()
