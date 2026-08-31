# ============================================================
# Ported from the Flask prototype's rag/retrieval/vector_retriever.py.
# VectorStore.search() is blocking (Qdrant client + Gemini embedding call),
# run via asyncio.to_thread.
# ============================================================
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.rag.schemas import RetrievalResult
from app.services.rag.vector_store import VectorStore

logger = logging.getLogger("rag.vector_retriever")


async def retrieve(
    query: str, doc_id: str, store: VectorStore, top_k: int = 4,
    db: Optional[AsyncIOMotorDatabase] = None, user_id: Optional[str] = None,
) -> RetrievalResult:
    hits, usage = await asyncio.to_thread(store.search, query, doc_id, top_k)

    # This query-embedding call spends real Gemini usage too — track it
    # under the same user, otherwise usage undercounts actual API cost.
    if db is not None and user_id is not None:
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.GEMINI, model="gemini-embedding-001",
            feature=Feature.RAG_RETRIEVE, usage=usage,
        )

    if not hits:
        logger.warning("vector_retriever: doc_id=%s returned zero hits from Qdrant", doc_id)
        return RetrievalResult(context_text="", source_nodes=[], confidence=0.0, doc_id=doc_id)

    context = "\n\n---\n\n".join(h.payload["text"] for h in hits)
    # qdrant cosine score ~ [0,1] already for normalized vectors; treat top
    # hit's score as a proxy confidence signal for the fallback decision.
    confidence = float(hits[0].score)
    logger.info("vector_retriever: doc_id=%s hits=%d top_score=%.3f", doc_id, len(hits), confidence)
    return RetrievalResult(
        context_text=context,
        source_nodes=[str(h.id) for h in hits],
        confidence=confidence,
        doc_id=doc_id,
    )
