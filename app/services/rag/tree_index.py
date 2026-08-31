# ============================================================
# PageIndex-lite index for structured docs — summarization + skeleton
# rendering. Ported from the Flask prototype's rag/indexing/tree_index.py.
#
# Persistence (save/load) lives in mongo_store.py, not here — this file only
# does in-memory work (summarization, skeleton rendering).
#
# Adapted from the Flask version's sync LLMClient abstraction to call
# app.services.claude.generate_text directly via asyncio.to_thread. Token
# tracking goes through app.services.ai_usage.record_ai_usage, which
# already accepts this module's plain-dict usage shape directly (see
# record_ai_usage._extract_tokens).
# ============================================================
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.claude import generate_text
from app.services.rag.schemas import TreeNode

logger = logging.getLogger("rag.tree_index")


async def summarize_nodes(
    nodes: List[TreeNode], db: Optional[AsyncIOMotorDatabase] = None, user_id: Optional[str] = None
) -> None:
    """
    One cheap batched call to summarize every node in one shot (not one
    call per node — that's how people accidentally 20x their API cost).
    Mutates nodes in place, filling `.summary`.

    db/user_id are optional: when provided, this call's token usage is
    tracked under the uploading user — this is a real Claude call, same as
    the tree retriever's node-picking call.
    """
    # Skip tiny/empty nodes, they don't need a summary.
    targets = [n for n in nodes if len(n.raw_content) > 40]
    if not targets:
        logger.info("summarize_nodes: no nodes with enough content to summarize (%d total nodes)", len(nodes))
        return

    listing = "\n\n".join(
        f"NODE_ID: {n.id}\nTITLE: {n.title}\nCONTENT:\n{n.raw_content[:800]}"
        for n in targets
    )
    prompt = (
        "For each NODE below, write exactly one line, no more than 15 words, "
        "summarizing what it contains (topics, assessments, weightings — "
        "whatever is literally there). Return strict JSON: "
        '{"NODE_ID": "one line summary", ...}\n\n' + listing
    )
    text, token_usage = await asyncio.to_thread(generate_text, prompt, max_tokens=1200)

    if db is not None and user_id is not None:
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model="claude-sonnet-4-6",
            feature=Feature.RAG_SUMMARIZE, usage=token_usage,
        )

    try:
        summaries = json.loads(_strip_code_fence(text))
        logger.info("summarize_nodes: summarized %d/%d eligible nodes", len(summaries), len(targets))
    except json.JSONDecodeError:
        logger.warning("summarize_nodes: failed to parse summary JSON, falling back to titles: %r", text[:200])
        summaries = {}

    for n in targets:
        n.summary = summaries.get(n.id, n.title)


def _strip_code_fence(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def skeleton_view(nodes: List[TreeNode]) -> str:
    """
    What gets shown to the LLM during retrieval reasoning: titles +
    one-line summaries, NOT full content. Keeps the reasoning call cheap
    even for a 30-node document.
    """
    lines = []
    for n in nodes:
        indent = "  " * (n.level - 1)
        tag = " [TABLE]" if n.is_table else ""
        lines.append(f"{indent}- ({n.id}) {n.title}{tag}: {n.summary}")
    return "\n".join(lines)
