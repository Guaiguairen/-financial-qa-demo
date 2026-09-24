# -*- coding: utf-8 -*-
"""Step 2：调用 MinerU 4.x 把 PDF 解析为结构化内容，并转换为规范化内容块文件。

对每份文档：
  1. 调用 mineru-kit parse（本地模型，支持 flash / basic / standard 档位）；
     - 子进程日志写入 data/logs/mineru_cli/{doc_id}.log，便于排障；
     - 失败自动重试（默认 1 次）；
     - 若已存在可用输出 zip 而缺少 blocks.jsonl，直接复用 zip，不重新解析。
  2. 解包输出（markdown.md / middle_json.json / images/）；
  3. 把 middle_json.json 转换为规范化 blocks.jsonl：
     - 页码 1-based；顺序保持阅读序；
     - 跳过页眉、页码、目录等噪声块；
     - 表格保留 HTML 行列结构（table_body），并保留标题层级（paragraph_title / doc_title）。

用法
----
    python src/step2_extract_mineru.py --tier basic                 # 全量抽取（跳过已完成）
    python src/step2_extract_mineru.py --tier basic --workers 2     # 并发（谨慎：注意内存）
    python src/step2_extract_mineru.py --tier basic --docs 688256_2025_annual
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402

MINERU_KIT = _ROOT / ".venv-mineru" / "Scripts" / "mineru-kit.exe"
CLI_LOG_DIR = common.LOG_DIR / "mineru_cli"

# 噪声块类型（不进入知识库）
SKIP_TYPES = {"header", "page_number", "footer", "index", "footnote", "aside_text"}
TEXT_TYPES = {"text", "paragraph_title", "doc_title", "caption", "list", "code", "reference"}

_print_lock = threading.Lock()


def collect_text(block: dict) -> str:
    """递归收集块内的文本片段。"""
    leaves: list[str] = []

    def walk(obj):
        if isinstance(obj, str):
            leaves.append(obj)
            return
        if isinstance(obj, dict):
            c = obj.get("content")
            if isinstance(c, str):
                leaves.append(c)
            elif isinstance(c, (list, dict)):
                walk(c)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(block.get("content"))
    if not leaves:
        return ""
    return "\n".join(leaves) if len(leaves) > 1 else leaves[0]


def middle_to_blocks(mj: dict, doc_meta: dict, logger=None) -> list[dict]:
    """把 MinerU middle_json 转成规范化内容块列表。"""
    blocks: list[dict] = []
    seq = 0
    unknown_types: dict[str, int] = {}
    for page in mj.get("pages") or []:
        pno = int(page.get("page_idx", 0)) + 1
        for b in page.get("blocks") or []:
            btype = b.get("type") or "unknown"
            if btype in SKIP_TYPES:
                continue
            if btype == "image":
                cap = collect_text({"content": [p for p in (b.get("content") or [])
                                                if isinstance(p, dict) and p.get("type") == "caption"]})
                if cap.strip():
                    blocks.append({"page": pno, "seq": seq, "type": "text", "src_type": "image_caption",
                                   "text": cap.strip(), "html": "", "level": None, "caption": ""})
                    seq += 1
                continue
            if btype == "table":
                html = ""
                cap_parts: list[str] = []
                for part in b.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "table_body":
                        html = part.get("content") or ""
                    elif part.get("type") == "caption":
                        cap_parts.append(collect_text({"content": [part]}))
                blocks.append({"page": pno, "seq": seq, "type": "table", "src_type": "table",
                               "text": "", "html": html, "level": None,
                               "caption": " ".join(x for x in cap_parts if x).strip()})
                seq += 1
                continue
            if btype in TEXT_TYPES or btype == "unknown":
                text = collect_text(b).strip()
                if not text:
                    continue
                lvl = b.get("level")
                if not isinstance(lvl, int) or lvl <= 0:
                    lvl = None
                blocks.append({"page": pno, "seq": seq, "type": "text", "src_type": btype,
                               "text": text, "html": "", "level": lvl, "caption": ""})
                seq += 1
                continue
            unknown_types[btype] = unknown_types.get(btype, 0) + 1
    if unknown_types and logger:
        logger.warning("未识别的块类型：%s", unknown_types)
    return blocks


def run_mineru(pdf: Path, out_dir: Path, tier: str, pages: str, doc_id: str,
               timeout: int = 7200) -> tuple[Path, float]:
    """调用 mineru-kit 解析单个 PDF（输出写独立日志文件），返回 (zip 路径, 耗时)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CLI_LOG_DIR / f"{doc_id}.log"
    cmd = [str(MINERU_KIT), "parse", str(pdf), "-o", str(out_dir), "-f", "zip", "--tier", tier]
    if pages and pages != "all":
        cmd += ["-p", pages]
    t0 = time.time()
    with open(log_path, "ab") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode("utf-8"))
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    cost = time.time() - t0
    if r.returncode != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"mineru-kit 退出码 {r.returncode}；日志尾部：{tail}")
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise RuntimeError(f"未找到输出 zip：{out_dir}")
    return zips[0], cost


def zip_to_blocks(zip_path: Path, doc_dir: Path, doc_meta: dict, logger) -> dict:
    """解包 zip 并转换为 blocks.jsonl，返回统计信息。"""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names:
            z.extract(n, doc_dir)
    mj_path = None
    for cand in doc_dir.rglob("middle_json.json"):
        mj_path = cand
        break
    if mj_path is None:
        raise RuntimeError(f"zip 中没有 middle_json.json（内容：{names[:10]}）")
    mj = json.loads(mj_path.read_text(encoding="utf-8"))
    blocks = middle_to_blocks(mj, doc_meta, logger)
    common.write_jsonl(doc_dir / "blocks.jsonl", blocks)
    return {"pages": len(mj.get("pages") or []), "blocks": len(blocks),
            "tables": sum(1 for b in blocks if b["type"] == "table")}


def extract_doc(doc_id: str, m: dict, tier: str, pages: str, logger,
                retries: int = 1) -> dict | None:
    """解析单份文档（可重试）。返回统计 dict 或 None（已存在/无 PDF）。"""
    doc_dir = common.EXTRACT_DIR / doc_id
    blocks_path = doc_dir / "blocks.jsonl"
    if blocks_path.exists():
        logger.info("%s: 已有 blocks.jsonl，跳过", doc_id)
        return None

    pdf = _ROOT / m["file"]
    if not pdf.exists():
        logger.warning("%s: PDF 不存在 %s", doc_id, pdf)
        return None

    doc_meta = {
        "doc_id": doc_id, "company": m["name"], "code": m["code"],
        "report_type": m["report_type"], "report_label": m["period_label"],
    }
    raw_dir = doc_dir / "raw"
    last_err = None
    for attempt in range(retries + 1):
        try:
            # 已有 zip（上次解析完成但未转换）→ 直接复用
            existing = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True) \
                if raw_dir.exists() else []
            if existing and pages == "all":
                zip_path, cost = existing[0], 0.0
                logger.info("%s: 复用已有 zip（%s）", doc_id, zip_path.name)
            else:
                zip_path, cost = run_mineru(pdf, raw_dir, tier, pages, doc_id)
                logger.info("%s: MinerU 解析完成（%.1fs，tier=%s）", doc_id, cost, tier)
            st = zip_to_blocks(zip_path, doc_dir, doc_meta, logger)
            stats = {"doc_id": doc_id, "tier": tier, "mineru_cost_s": round(cost, 1),
                     "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"), **st}
            (doc_dir / "doc_stats.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("%s: %d 页 → %d 块（表格 %d）", doc_id, st["pages"], st["blocks"], st["tables"])
            return stats
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                logger.warning("%s: 第 %d 次尝试失败（%s），10 秒后重试", doc_id, attempt + 1, e)
                time.sleep(10)
    raise RuntimeError(f"{doc_id}: 重试后仍失败：{last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MinerU 结构化抽取")
    ap.add_argument("--tier", default="basic", choices=["flash", "basic", "standard", "advanced"])
    ap.add_argument("--docs", default="", help="指定 doc_id，逗号分隔")
    ap.add_argument("--pages", default="all", help="页码范围（调试用），如 1-20")
    ap.add_argument("--workers", type=int, default=1, help="并发数（默认 1；注意内存）")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    common.ensure_dirs()
    logger = common.setup_logger("step2", "step2_extract.log")
    manifest = {m["doc_id"]: m for m in common.read_jsonl(common.MANIFEST_PATH)}
    if not manifest:
        logger.error("manifest.jsonl 为空，先运行 step1")
        sys.exit(1)
    docs = list(sorted(manifest.items()))
    if args.docs:
        keep = {d.strip() for d in args.docs.split(",") if d.strip()}
        docs = [(k, v) for k, v in docs if k in keep]
    if args.limit:
        docs = docs[: args.limit]

    logger.info("共 %d 份文档待处理（tier=%s, workers=%d）", len(docs), args.tier, args.workers)
    t0 = time.time()
    ok, skip, fail = 0, 0, []
    total_pages = 0
    done_count = 0

    def _run(idx: int, did: str, m: dict):
        logger.info("[%d/%d] %s", idx, len(docs), did)
        return extract_doc(did, m, args.tier, args.pages, logger, retries=args.retries)

    results = []
    if args.workers <= 1:
        for i, (did, m) in enumerate(docs, 1):
            try:
                results.append((did, _run(i, did, m), None))
            except Exception as e:  # noqa: BLE001
                results.append((did, None, e))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, i, did, m): did for i, (did, m) in enumerate(docs, 1)}
            for fut in as_completed(futs):
                did = futs[fut]
                try:
                    results.append((did, fut.result(), None))
                except Exception as e:  # noqa: BLE001
                    results.append((did, None, e))

    for did, st, err in results:
        if err is not None:
            logger.error("%s: 失败 %s", did, err)
            fail.append(f"{did}: {err}")
        elif st is None:
            skip += 1
        else:
            ok += 1
            total_pages += st.get("pages", 0)

    elapsed = (time.time() - t0) / 60
    logger.info("=" * 60)
    logger.info("完成：新解析 %d，跳过 %d，失败 %d，共 %d 页，总耗时 %.1f 分钟",
                ok, skip, len(fail), total_pages, elapsed)
    for f in fail:
        logger.warning("  ! %s", f)


if __name__ == "__main__":
    main()
