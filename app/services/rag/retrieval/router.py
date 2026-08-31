# ============================================================
# The one function every feature (roadmap, notes, etc.) actually calls.
# This is the "layer" — everything upstream of this file is implementation
# detail that can change without breaking callers.
#
# Ported from the Flask prototype's rag/retrieval/router.py.
# ============================================================
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.rag.retrieval import tree_retriever, vector_retriever
from app.services.rag.schemas import DocType, RetrievalResult

logger = logging.getLogger("rag.router")

# Below this confidence, the caller should fall back to no-RAG (pure model
# knowledge) generation instead of forcing weak context into the prompt.
# Tune this once you have real query logs.
MIN_CONFIDENCE = 0.35


async def retrieve(
    query: str, doc_id: str, doc_type: DocType, db: AsyncIOMotorDatabase,
    user_id: Optional[str] = None, vector_store=None, max_nodes: int = 3,
) -> RetrievalResult:
    logger.info("router: doc_id=%s doc_type=%s -> using %s retriever",
                doc_id, doc_type, "tree (vector-less)" if doc_type == DocType.STRUCTURED else "vector")
    if doc_type == DocType.STRUCTURED:
        return await tree_retriever.retrieve(query, doc_id, db, user_id, max_nodes=max_nodes)

    if vector_store is None:
        logger.warning("router: doc_id=%s is UNSTRUCTURED but no vector_store available — treating as ungrounded", doc_id)
        return RetrievalResult(context_text="", source_nodes=[], confidence=0.0, doc_id=doc_id)
    return await vector_retriever.retrieve(query, doc_id, vector_store, db=db, user_id=user_id)


def should_use_rag(result: RetrievalResult) -> bool:
    return bool(result.context_text) and result.confidence >= MIN_CONFIDENCE
