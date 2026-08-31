# ============================================================
# Lazily-constructed VectorStore singleton for the vector (unstructured-doc)
# RAG path. Ported from the Flask prototype's rag/singletons.py.
#
# VectorStore construction talks to Qdrant over the network. get_vector_store()
# never raises: if Qdrant isn't running (e.g. local dev before
# `docker run -p 6333:6333 qdrant/qdrant`), it returns None and callers treat
# that as "vector RAG unavailable for this request" — tree RAG and ungrounded
# generation must keep working regardless of Qdrant's state.
# ============================================================
from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger("rag.singletons")

_vector_store = None


def get_vector_store():
    """
    Blocking — call via asyncio.to_thread() from async callers.
    Returns a ready VectorStore, or None if Qdrant is unreachable. Retries on
    every call while it hasn't succeeded yet (cheap relative to an LLM call)
    so starting Qdrant late in a dev session recovers without a restart.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        from app.services.rag.vector_store import VectorStore

        _vector_store = VectorStore(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        logger.info("Qdrant vector store connected at %s", settings.QDRANT_URL)
    except Exception as exc:
        logger.warning("Vector store unavailable (Qdrant down or embedder init failed): %s", exc)
        return None

    return _vector_store
