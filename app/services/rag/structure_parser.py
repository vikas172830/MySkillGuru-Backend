# ============================================================
# Two jobs, ported from the Flask prototype's rag/ingestion/structure_parser.py
# unchanged (pure text processing, no I/O — no framework adaptation needed):
#   1. classify_doc_type(): decide STRUCTURED vs UNSTRUCTURED so the router
#      can pick tree-index vs vector-index.
#   2. build_tree(): turn STRUCTURED pages into a TreeNode list using
#      heading patterns (numbered sections like "5.2 Assessment Detail",
#      or markdown #/##/###). Tables are attached to the nearest node,
#      never split across nodes.
# ============================================================
from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.services.rag.schemas import DocType, TreeNode, new_id

logger = logging.getLogger("rag.structure_parser")

# Matches: "5.2 Assessment Detail", "1. General Course Information",
# "2.3. Graduate Attributes", markdown "## Heading"
_NUMBERED_HEADING = re.compile(r"^\s{0,3}(\d+(\.\d+)*)\.?\s+(.{3,80})$")
_MD_HEADING = re.compile(r"^(#{1,4})\s+(.{2,80})$")


WORDS_PER_VIRTUAL_PAGE = 500


def classify_doc_type(pages: List[dict]) -> DocType:
    """
    Heuristic: structured admin/course docs have a high density of numbered
    headings and tables relative to page count. Prose textbooks don't.
    Tune the thresholds against your own corpus once you have >20 uploads.

    This backend's extractors (see course_material.py) return the whole
    document as ONE flat page, not true per-page dicts — dividing by
    len(pages) in that case degenerates to a raw count (any single numbered
    line anywhere in the whole document would trip STRUCTURED). Using an
    estimated "effective page count" from word count instead keeps the
    density meaningful regardless of whether real pagination is available.
    """
    heading_hits = 0
    table_hits = 0
    total_words = 0
    for page in pages:
        text = page["text"]
        total_words += len(text.split())
        for line in text.splitlines():
            if _NUMBERED_HEADING.match(line) or _MD_HEADING.match(line):
                heading_hits += 1
        table_hits += len(page.get("tables", []))

    effective_pages = max(total_words / WORDS_PER_VIRTUAL_PAGE, len(pages), 1)

    heading_density = heading_hits / effective_pages
    table_density = table_hits / effective_pages

    # >0.8 headings/page or any real table density => treat as structured.
    doc_type = DocType.STRUCTURED if (heading_density >= 0.8 or table_density >= 0.15) else DocType.UNSTRUCTURED
    logger.info(
        "classify_doc_type: pages=%d effective_pages=%.1f heading_density=%.2f table_density=%.2f -> %s",
        len(pages), effective_pages, heading_density, table_density, doc_type.value,
    )
    return doc_type


def _heading_level(match: re.Match) -> int:
    numbering = match.group(1)
    return numbering.count(".") + 1


def build_tree(pages: List[dict], doc_id: str) -> List[TreeNode]:
    """
    Walk pages line by line. Every heading opens a new node; everything
    until the next heading (at any level) belongs to that node, including
    any table extracted on the pages it spans.
    """
    nodes: List[TreeNode] = []
    stack: List[TreeNode] = []  # ancestor stack, for parent_id assignment

    def close_open_nodes_below(level: int):
        while stack and stack[-1].level >= level:
            stack.pop()

    current: Optional[TreeNode] = None
    buffer: List[str] = []

    def flush():
        if current is not None:
            current.raw_content = "\n".join(buffer).strip()

    for page in pages:
        for line in page["text"].splitlines():
            m_num = _NUMBERED_HEADING.match(line)
            m_md = _MD_HEADING.match(line)
            is_heading = m_num or m_md

            if is_heading:
                flush()
                level = _heading_level(m_num) if m_num else len(m_md.group(1))
                title = (m_num.group(3) if m_num else m_md.group(2)).strip()

                close_open_nodes_below(level)
                parent_id = stack[-1].id if stack else None

                current = TreeNode(
                    id=new_id(),
                    title=title,
                    level=level,
                    parent_id=parent_id,
                    page_start=page["page_num"],
                    page_end=page["page_num"],
                    raw_content="",
                )
                if parent_id:
                    for n in nodes:
                        if n.id == parent_id:
                            n.children_ids.append(current.id)
                            break
                nodes.append(current)
                stack.append(current)
                buffer = []
            else:
                if line.strip():
                    buffer.append(line)
                if current is not None:
                    current.page_end = page["page_num"]

        # attach this page's tables to whatever node is currently open
        for table_md in page.get("tables", []):
            if current is not None:
                buffer.append("\n" + table_md + "\n")
                current.is_table = True

    flush()

    # root fallback: if nothing matched (rare, e.g. loose txt file), make
    # the whole document a single node so the tree path still works.
    if not nodes:
        full_text = "\n".join(p["text"] for p in pages)
        nodes = [
            TreeNode(
                id=new_id(),
                title="Document",
                level=1,
                parent_id=None,
                page_start=1,
                page_end=len(pages),
                raw_content=full_text,
            )
        ]

    # Diagnostic: nodes with barely any content are usually parsing noise —
    # a stray page footer, a table-of-contents dot-leader line, or a garbled
    # table fragment matched the heading regex instead of a real heading.
    # These don't get summarized (tree_index.summarize_nodes skips them) and
    # dilute the outline the retrieval LLM reasons over. This won't block
    # anything, but a high ratio here is the most common reason retrieval
    # later comes back with low/zero confidence on a real-world PDF.
    thin_nodes = [n for n in nodes if len(n.raw_content) <= 40]
    if thin_nodes:
        logger.warning("build_tree: %d/%d nodes have <=40 chars of content (likely parsing noise, "
                        "e.g. page footers/TOC lines misdetected as headings): %s",
                        len(thin_nodes), len(nodes), [n.title for n in thin_nodes[:10]])
    logger.info("build_tree: produced %d node(s) from %d page(s)", len(nodes), len(pages))
    return nodes
