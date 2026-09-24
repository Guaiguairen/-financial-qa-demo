# -*- coding: utf-8 -*-
"""Step 3：把 MinerU 抽取出的规范化内容块切分为知识库文本块（chunk）。

输入
----
data/extracted/{doc_id}/blocks.jsonl（由 step2 产出，每行一个内容块）：
    {"doc_id","company","code","report_label","page"(1-based),"seq","type",
     "text","html","level","caption"}

切块规则
--------
1. 顺序遍历块，维护"章节路径"（如：第三节 管理层讨论与分析 / 二、报告期内主要经营情况）：
   - level 字段由 MinerU 提供时直接使用；否则用中文报告标题模式启发式识别
     （第X节 / 一、 / （一） / 1、 等）；
2. 正文块按页内合并：同一页、同一章节的连续文本合并为一个 chunk，
   长度达到目标值（默认 800 字）即冲刷；跨页不合并（保证页码引用唯一）；
3. 表格块独立成 chunk，保留行列结构（html），同时提供行文本化 text 供检索；
4. 每个 chunk 携带三类必备元数据：公司名称、章节标题、页码。

输出
----
data/chunks/chunks.jsonl  每行一个 chunk：
    {"chunk_id","doc_id","company","code","report_type","report_label",
     "section","page","type","text","html","n_chars"}

用法
----
    python src/step3_chunk.py                 # 全量切块
    python src/step3_chunk.py --docs 688256_2025_annual
    python src/step3_chunk.py --sample 3      # 打印抽样检查
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import common  # noqa: E402

# ---------------------------------------------------------------- 标题目录
RE_PART = re.compile(r"^第[一二三四五六七八九十百]+[节章部分]")
RE_CN_NUM = re.compile(r"^[一二三四五六七八九十]+、")
RE_BRACKET_NUM = re.compile(r"^[（(][一二三四五六七八九十]+[)）]")
RE_DIGIT = re.compile(r"^\d{1,2}\s*[、.．]")

MAX_HEADING_LEN = 60  # 超过则大概率是正文而非标题


def detect_level(text: str) -> int | None:
    """启发式判断标题层级（与 MinerU 标注层级统一）：
    1 文档标题（doc_title）；2 第X节；3 一、；4 （一）；5 1、。返回 None 表示非标题。"""
    t = text.strip()
    if not t or len(t) > MAX_HEADING_LEN:
        return None
    # 标题通常不以句末标点结尾
    if t[-1] in "。；，：,;":
        return None
    if RE_PART.match(t):
        return 2
    if RE_CN_NUM.match(t):
        return 3
    if RE_BRACKET_NUM.match(t) and len(t) <= 40:
        return 4
    if RE_DIGIT.match(t) and len(t) <= 30 and "。" not in t and "，" not in t:
        return 5
    return None


def clean_text(s: str) -> str:
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def heading_sane(t: str) -> bool:
    """标题合理性安检：排除被误标注的断句残片 / 正文片段 / 勾选噪声。

    规则：含句号，或以标点开头，或带逗号且较长（>16字），或为勾选框噪声
    （如 "√适用 □不适用"）——都不算标题。
    """
    if not t:
        return False
    if "。" in t:
        return False
    if t[0] in "，。、；：）」』】,;)":
        return False
    if "，" in t and len(t) > 16:
        return False
    if set(t) <= set("√□适用不 \u3000"):  # 勾选定式噪声
        return False
    return True


def remap_title_level(t: str) -> int | None:
    """对 MinerU 标注的标题重新分配层级（统一到 2~5 的编号体系）。

    MinerU（basic）会把所有标题都标成同一层级，这里用中文报告标题模式
    按语义归类；无法归类的标题交由调用方按上下文处理。
    """
    if RE_PART.match(t):
        return 2
    if RE_CN_NUM.match(t):
        return 3
    if RE_BRACKET_NUM.match(t) or RE_DIGIT.match(t):
        return 4
    return None


# ---------------------------------------------------------------- 章节栈
class SectionTracker:
    """维护标题栈，输出 '第三节 xxx / 二、yyy' 形式的章节路径。"""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def update(self, level: int, title: str) -> None:
        title = clean_text(title)
        self.stack = [(lv, t) for lv, t in self.stack if lv < level]
        self.stack.append((level, title))

    @property
    def path(self) -> str:
        if not self.stack:
            return "（正文前部）"
        return " / ".join(t for _, t in self.stack[-3:])


# ---------------------------------------------------------------- 切块
def chunk_document(blocks: list[dict], doc_meta: dict, target_len: int = 800,
                   min_chunk: int = 80, min_emit: int = 30) -> list[dict]:
    """把规范化内容块切分为 chunk 列表。

    - 同一页、同一章节的正文合并；长度达到 target_len 冲刷；
    - 跨页边界不携带残留（保证每个 chunk 的页码唯一且准确）；
    - 边界处的小残片（<min_chunk 字）若 >= min_emit 则独立成小块，否则丢弃；
    - 表格独立成块，保留 html 行列结构。
    """
    out: list[dict] = []
    tracker = SectionTracker()
    buf: list[dict] = []
    cur_page: int | None = None
    prev_level: int | None = None

    def emit(text: str, page: int | None, ctype: str, *, html: str = "", caption: str = "") -> None:
        prefix = "t" if ctype == "table" else "c"
        chunk = {
            "chunk_id": f"{doc_meta['doc_id']}_p{(page or 0):04d}_{prefix}{len(out):04d}",
            **{k: doc_meta[k] for k in ("doc_id", "company", "code", "report_type", "report_label")},
            "section": tracker.path,
            "page": page,
            "type": ctype,
            "text": text,
            "html": html,
            "caption": caption,
            "n_chars": len(text),
        }
        out.append(chunk)

    def flush(reason: str) -> None:
        """冲刷缓冲区。reason='length' 时保留残片继续吞并（同页）；
        其他边界（换页/标题/表格/结尾）不跨边界携带。"""
        nonlocal buf
        if not buf:
            return
        text = clean_text("\n".join(b["text"] for b in buf))
        page = buf[0]["page"]
        if len(text) < min_chunk:
            if reason == "length":  # 同页内继续积累
                buf = [{**buf[0], "text": text}]
                return
            if len(text) >= min_emit:  # 边界残片：独立成小块
                emit(text, page, "text")
            buf = []
            return
        emit(text, page, "text")
        buf = []

    for b in blocks:
        text = clean_text(b.get("text") or "")
        btype = b.get("type", "text")

        # 换页 → 先冲刷上一页（页码引用唯一性）
        if cur_page is not None and b["page"] != cur_page:
            flush("boundary")
        cur_page = b["page"]

        if btype == "table":
            flush("boundary")
            ttext = text or table_html_to_text(b.get("html") or "")
            if not ttext and not b.get("html"):
                continue
            emit(ttext, b["page"], "table", html=b.get("html") or "", caption=b.get("caption") or "")
            continue

        if btype in ("image", "equation", "chart"):
            # 图片/公式暂不建块（图表信息多在周边文本中）
            continue

        # 文本块：先判断是否标题
        raw_level = b.get("level")
        tagged = isinstance(raw_level, int) and raw_level > 0
        if tagged:
            level = remap_title_level(text)
            if level is None:  # 无模式的标注标题：按上下文嵌套
                level = min(5, (prev_level or 3) + 1)
        else:
            level = detect_level(text)
        if level is not None and not heading_sane(text):
            level = None
        if level is not None:
            flush("boundary")
            if level >= 2:  # 文档级标题（doc_title）不进章节栈
                tracker.update(level, text)
                prev_level = level
            buf.append({**b, "text": text})  # 标题文本并入后续块，增强上下文
            continue

        buf.append({**b, "text": text})
        total = sum(len(x["text"]) for x in buf)
        if total >= target_len:
            flush("length")

    flush("boundary")
    return out


def table_html_to_text(html: str) -> str:
    """把表格 HTML 转成行列文本（用于检索），保留 '列1 | 列2' 的行结构。"""
    if not html:
        return ""
    import re as _re

    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=_re.S | _re.I)
    lines = []
    for row in rows:
        cells = _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=_re.S | _re.I)
        cells = [_re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
        cells = [c for c in cells]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def load_blocks(path: Path) -> list[dict]:
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(json.loads(line))
    blocks.sort(key=lambda b: (b["page"], b.get("seq", 0)))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description="切块并附加元数据")
    ap.add_argument("--docs", type=str, default="", help="只处理指定 doc_id，逗号分隔")
    ap.add_argument("--sample", type=int, default=0, help="打印 N 条抽样 chunk 检查")
    args = ap.parse_args()

    common.ensure_dirs()
    logger = common.setup_logger("step3", "step3_chunk.log")
    manifest = {m["doc_id"]: m for m in common.read_jsonl(common.MANIFEST_PATH)}
    if args.docs:
        keep = {d.strip() for d in args.docs.split(",") if d.strip()}
        manifest = {k: v for k, v in manifest.items() if k in keep}

    all_chunks: list[dict] = []
    stats = []
    for did, m in sorted(manifest.items()):
        blocks_path = common.EXTRACT_DIR / did / "blocks.jsonl"
        if not blocks_path.exists():
            logger.warning("%s: 缺少 blocks.jsonl（跳过，请先运行 step2）", did)
            continue
        blocks = load_blocks(blocks_path)
        doc_meta = {
            "doc_id": did, "company": m["name"], "code": m["code"],
            "report_type": m["report_type"], "report_label": m["period_label"],
        }
        chunks = chunk_document(blocks, doc_meta)
        all_chunks.extend(chunks)
        n_tab = sum(1 for c in chunks if c["type"] == "table")
        stats.append((did, len(blocks), len(chunks), n_tab))
        logger.info("%s: 块 %d → chunks %d（表格 %d）", did, len(blocks), len(chunks), n_tab)

    common.write_jsonl(common.CHUNKS_PATH, all_chunks)
    total_chars = sum(c["n_chars"] for c in all_chunks)
    logger.info("完成：共 %d 个 chunk，平均 %d 字，总计 %.1f 万字",
                len(all_chunks), total_chars // max(1, len(all_chunks)), total_chars / 10000)

    if args.sample:
        import random

        for c in random.sample(all_chunks, min(args.sample, len(all_chunks))):
            print("-" * 100)
            print(f"[{c['chunk_id']}] {c['company']} | {c['section']} | 第{c['page']}页 | {c['type']} | {c['n_chars']}字")
            print(c["text"][:500])


if __name__ == "__main__":
    main()
