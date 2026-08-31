# ============================================================
# LLM reasons over the tree skeleton (titles + summaries only) and picks
# which node ids actually answer the query. Then, and only then, do we pull
# full content for just those nodes.
#
# Ported from the Flask prototype's rag/retrieval/tree_retriever.py, adapted
# to async Motor + app.services.claude (see tree_index.py's header comment
# for why token tracking is done inline rather than via roadmap_ai.py's
# object-attribute-based helper).
# ============================================================
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.claude import generate_text
from app.services.rag.mongo_store import load_tree
from app.services.rag.schemas import RetrievalResult, TreeNode
from app.services.rag.tree_index import skeleton_view

logger = logging.getLogger("rag.tree_retriever")


async def retrieve(
    query: str, doc_id: str, db: AsyncIOMotorDatabase, user_id: Optional[str] = None, max_nodes: int = 3
) -> RetrievalResult:
    nodes = await load_tree(db, doc_id)
    if not nodes:
        logger.warning("tree_retriever: doc_id=%s has no saved tree — returning empty result", doc_id)
        return RetrievalResult(context_text="", source_nodes=[], confidence=0.0, doc_id=doc_id)

    skeleton = skeleton_view(nodes)

    prompt = (
        "You are navigating a document's structure to answer a query. "
        "Below is the document's outline: node ids, titles, and one-line summaries.\n\n"
        f"OUTLINE:\n{skeleton}\n\n"
        f"QUERY: {query}\n\n"
        f"Pick at most {max_nodes} node ids that are directly relevant. "
        "If genuinely nothing in the outline addresses the query, return an empty list. "
        'Return strict JSON: {"node_ids": ["...", ...], "confidence": 0.0-1.0}'
    )
    text, token_usage = await asyncio.to_thread(generate_text, prompt, max_tokens=300)

    # This node-picking call spends real Claude tokens too — track it under the
    # same user, otherwise token_usage undercounts actual API cost.
    if user_id is not None:
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model="claude-sonnet-4-6",
            feature=Feature.RAG_RETRIEVE, usage=token_usage,
        )

    try:
        parsed = json.loads(_strip_fence(text))
        picked_ids = parsed.get("node_ids", [])
        confidence = float(parsed.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError):
        logger.warning("tree_retriever: doc_id=%s failed to parse node-picking response: %r", doc_id, text[:200])
        picked_ids, confidence = [], 0.0

    id_to_node = {n.id: n for n in nodes}
    picked: list[TreeNode] = [id_to_node[i] for i in picked_ids if i in id_to_node]

    logger.info("tree_retriever: doc_id=%s outline_nodes=%d picked=%d confidence=%.2f picked_titles=%s",
                doc_id, len(nodes), len(picked), confidence if picked else 0.0,
                [n.title for n in picked])

    context = "\n\n---\n\n".join(f"### {n.title}\n{n.raw_content}" for n in picked)
    return RetrievalResult(
        context_text=context,
        source_nodes=[n.id for n in picked],
        confidence=confidence if picked else 0.0,
        doc_id=doc_id,
    )


def _strip_fence(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
