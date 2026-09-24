# -*- coding: utf-8 -*-
"""Step 1：自动检索并下载样本公司近两年年报与半年报全文（PDF）。

数据源说明
----------
巨潮资讯网（www.cninfo.com.cn）由深圳证券交易所下属深圳证券信息有限公司运营，
是沪深两市上市公司公告的官方指定披露平台。本脚本调用其公告检索接口，按
"公司 × 披露时间窗口"查询公告，定位年报 / 半年报全文 PDF 的官方文件地址
（static.cninfo.com.cn），全程自动下载，无需人工干预。

处理流程
--------
1) 按公司 + 报告期窗口检索公告列表（自动分页）；
2) 标题精确匹配（"20XX年年度报告" / "20XX年半年度报告"，允许"（修订版）"等后缀），
   排除摘要、英文版、取消 / 延期公告等干扰项；
3) 对同一报告期选择披露时间最新的匹配版本，下载 PDF 至
   data/raw_pdfs/{code}_{name}/{year}_{年度报告|半年度报告}.pdf；
4) 逐条写入 data/raw_pdfs/manifest.jsonl（标题、公告日期、原文链接、SHA256、页数）。

用法
----
    python src/step1_download_reports.py --dry-run            # 只检索与展示，不下载
    python src/step1_download_reports.py                      # 正式下载（幂等：已存在则跳过）
    python src/step1_download_reports.py --codes 688256,688041
    python src/step1_download_reports.py --force              # 强制重新下载
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "https://static.cninfo.com.cn/"

# 标题中不允许出现的干扰词（年报摘要、英文版、取消/延期公告等）
TITLE_BAD_WORDS = ["摘要", "英文", "取消", "延期", "暂停", "摘要版"]


# ------------------------------------------------------------------ 检索
def period_window(year: str, report_type: str, wide: bool = False) -> str:
    """返回公告检索的时间窗口（披露时间）。

    年报：次年 2 月 ~ 7 月披露（多数为 3~4 月）；半年报：当年 7 月 ~ 次年 1 月披露。
    wide=True 时放宽窗口，用于兜底重试。
    """
    y = int(year)
    if report_type == "annual":
        return f"{y + 1}-01-01~{y + 1}-12-31" if wide else f"{y + 1}-02-01~{y + 1}-07-31"
    return f"{y}-01-01~{y + 1}-03-31" if wide else f"{y}-07-01~{y + 1}-01-31"


def title_pattern(year: str, report_type: str) -> re.Pattern:
    """标题精确匹配：以'20XX年年度报告'或'20XX年半年度报告'结尾。

    允许的尾部变体：空、（修订版）/（更新后）等括号后缀、'全文'字样，
    及二者的组合（如'2025年半年度报告全文'）。
    """
    core = f"{year}年年度报告" if report_type == "annual" else f"{year}年半年度报告"
    return re.compile(re.escape(core) + r"(?:（[^）]*）|全文|\s)*$")


def query_announcements(sess, comp: dict, se_date: str, *, max_pages: int = 10, logger=None) -> list[dict]:
    """按公司代码 + 时间窗口检索公告（自动分页）。

    注意：cninfo 单页上限 30 条（即使 pageSize 请求更大），因此按
    totalAnnouncement 逐页翻取，避免遗漏埋在列表深处的定期报告。
    """
    column = "sse" if comp["exchange"] == "sse" else "szse"
    page_size = 30  # cninfo 服务端单页硬上限
    out: list[dict] = []
    fetched = 0
    total = None
    for page in range(1, max_pages + 1):
        data = {
            "pageNum": str(page), "pageSize": str(page_size), "column": column, "tabName": "fulltext",
            "plate": "", "stock": "", "searchkey": comp["code"], "secid": "",
            "category": "", "trade": "", "seDate": se_date,
            "sortName": "", "sortType": "", "isHLtitle": "true",
        }
        js = None
        for attempt in range(3):
            try:
                resp = sess.post(
                    CNINFO_QUERY, data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": "http://www.cninfo.com.cn/"},
                    timeout=30,
                )
                js = resp.json()
                break
            except Exception as e:  # noqa: BLE001
                if logger:
                    logger.warning("检索失败 %s 第%d页（第%d次重试）: %s", comp["code"], page, attempt + 1, e)
                time.sleep(2 + attempt * 2)
        if js is None:
            break

        if total is None:
            total = int(js.get("totalAnnouncement") or 0)
        anns = js.get("announcements") or []
        # 仅保留属于目标公司的公告
        anns = [a for a in anns if comp["code"] in str(a.get("secCode") or "")]
        if not anns:
            break
        out.extend(anns)
        fetched += len(anns)
        if fetched >= total:
            break
        time.sleep(0.6 + random.random() * 0.4)
    return out


def pick_report(anns: list[dict], year: str, report_type: str) -> dict | None:
    """在公告列表中挑选目标报告全文，返回公告 dict（含 _match 标题）。"""
    pat = title_pattern(year, report_type)
    cands = []
    for a in anns:
        title = common.strip_em(a.get("announcementTitle") or "")
        if not pat.search(title):
            continue
        if any(b in title for b in TITLE_BAD_WORDS):
            continue
        cands.append(a)
    if not cands:
        return None
    # 披露时间最新的版本（若同日有多条，优先非修订版之外的顺序不关键）
    cands.sort(key=lambda a: int(a.get("announcementTime") or 0), reverse=True)
    best = dict(cands[0])
    best["_match_title"] = common.strip_em(best.get("announcementTitle") or "")
    best["_candidates"] = [
        {
            "title": common.strip_em(c.get("announcementTitle") or ""),
            "date": ts_to_date(c.get("announcementTime")),
            "url": CNINFO_STATIC + str(c.get("adjunctUrl") or "").lstrip("/"),
        }
        for c in cands
    ]
    return best


def ts_to_date(ms) -> str:
    try:
        return dt.datetime.fromtimestamp(int(ms) / 1000).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------ 下载
def download_pdf(sess, url: str, dest: Path, *, logger, force: bool = False) -> str:
    """下载 PDF，返回状态：downloaded / skipped。"""
    if dest.exists() and dest.stat().st_size > 1024 and not force:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = common.get_with_retry(sess, url, timeout=180, stream=True,
                              headers={"Referer": "http://www.cninfo.com.cn/"})
    tmp = dest.with_suffix(".part")
    total = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    # 校验 PDF 魔数
    with open(tmp, "rb") as f:
        magic = f.read(5)
    if magic[:4] != b"%PDF":
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"下载内容不是 PDF（magic={magic!r}）: {url}")
    tmp.replace(dest)
    logger.info("  已下载 %s（%s）", dest.name, common.human_size(total))
    return "downloaded"


def pdf_pages(path: Path) -> int | None:
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as d:
            return d.page_count
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="下载样本公司年报/半年报全文 PDF")
    ap.add_argument("--dry-run", action="store_true", help="只检索与展示，不下载")
    ap.add_argument("--codes", type=str, default="", help="只处理指定代码，逗号分隔")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    common.ensure_dirs()
    logger = common.setup_logger("step1", "step1_download.log")
    cfg = common.load_config()
    targets = common.iter_targets(cfg)
    if args.codes:
        keep = {c.strip() for c in args.codes.split(",") if c.strip()}
        targets = [t for t in targets if t[0]["code"] in keep]

    logger.info("目标：%d 家公司 × %d 个报告期 = %d 份文档（%s）",
                len({t[0]['code'] for t in targets}),
                len(targets) // max(1, len({t[0]['code'] for t in targets})),
                len(targets), "DRY-RUN" if args.dry_run else "下载模式")

    sess = common.make_session()
    manifest = {m["doc_id"]: m for m in common.read_jsonl(common.MANIFEST_PATH)}
    errors: list[str] = []
    results: list[tuple] = []

    for idx, (comp, report_type, year) in enumerate(targets, 1):
        label = common.period_label(report_type, year)
        did = common.doc_id(comp["code"], report_type, year)
        prefix = f"[{idx}/{len(targets)}] {comp['code']} {comp['name']} {label}"
        try:
            pick = None
            for wide in (False, True):
                se_date = period_window(year, report_type, wide=wide)
                anns = query_announcements(sess, comp, se_date, logger=logger)
                pick = pick_report(anns, year, report_type)
                if pick:
                    break
                time.sleep(0.5)

            if not pick:
                msg = f"未检索到 {label} 全文公告"
                logger.warning("%s -> %s", prefix, msg)
                errors.append(f"{did}: {msg}")
                results.append((did, "missing", ""))
                continue

            url = CNINFO_STATIC + str(pick.get("adjunctUrl") or "").lstrip("/")
            ann_date = ts_to_date(pick.get("announcementTime"))

            if args.dry_run:
                logger.info("%s -> 命中《%s》（披露 %s）", prefix, pick["_match_title"], ann_date)
                if len(pick["_candidates"]) > 1:
                    logger.info("    其他候选：%s", [c["title"] for c in pick["_candidates"][1:]])
                results.append((did, "dry-run", pick["_match_title"]))
                time.sleep(0.4)
                continue

            fname = f"{year}_{'年度报告' if report_type == 'annual' else '半年度报告'}.pdf"
            dest = common.RAW_PDF_DIR / f"{comp['code']}_{comp['name']}" / fname
            status = download_pdf(sess, url, dest, logger=logger, force=args.force)
            pages = pdf_pages(dest)
            manifest[did] = {
                "doc_id": did,
                "code": comp["code"],
                "name": comp["name"],
                "full_name": comp.get("full_name", ""),
                "exchange": comp["exchange"],
                "sub_theme": comp.get("sub_theme", ""),
                "report_type": report_type,
                "report_year": year,
                "period_label": label,
                "title": pick["_match_title"],
                "announcement_date": ann_date,
                "url": url,
                "file": str(dest.relative_to(_ROOT)).replace("\\", "/"),
                "size": dest.stat().st_size,
                "sha256": common.sha256_file(dest),
                "pages": pages,
                "candidates": pick["_candidates"],
                "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            results.append((did, status, pick["_match_title"]))
            time.sleep(0.8 + random.random() * 0.6)
        except Exception as e:  # noqa: BLE001
            logger.error("%s -> 失败: %s", prefix, e)
            errors.append(f"{did}: {e}")

    if not args.dry_run:
        common.write_jsonl(common.MANIFEST_PATH,
                           [manifest[k] for k in sorted(manifest.keys())])

    # ---------------- 汇总 ----------------
    logger.info("=" * 60)
    ok = sum(1 for _, s, _ in results if s in ("downloaded", "skipped", "dry-run"))
    logger.info("完成：成功 %d / 共 %d；失败 %d", ok, len(targets), len(errors))
    total_pages = sum((m.get("pages") or 0) for m in manifest.values()) if not args.dry_run else 0
    if total_pages:
        logger.info("manifest 累计页数：%d 页 / %d 份文档", total_pages, len(manifest))
    for e in errors:
        logger.warning("  ! %s", e)


if __name__ == "__main__":
    main()
