# ============================================================
# Document + tree persistence for the RAG layer, ported from the Flask
# prototype's rag/indexing/mongo_store.py and adapted from sync PyMongo to
# async Motor (this backend's convention throughout).
#
# Collections (same `nexus` database, no new Mongo instance):
#   courseMaterials       — one doc per uploaded file (DocumentRecord equivalent)
#   courseMaterialTrees   — one doc per structured file's tree (TreeNode[] equivalent)
#
# Vector chunks for UNSTRUCTURED docs still go to Qdrant (see vector_store.py).
# ============================================================
from __future__ import annotations

import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.rag.schemas import DocType, DocumentRecord, SourceFormat, TreeNode

logger = logging.getLogger("rag.mongo_store")


def _to_document_record(doc: dict) -> DocumentRecord:
    """
    Mongo hands back plain strings for enum fields (it has no concept of
    Python enums) — reconstruct them properly so callers can rely on
    `.doc_type.value` etc. rather than only on DocType's str-equality.
    """
    doc = dict(doc)
    doc.pop("_id", None)
    doc.pop("created_at", None)
    doc["doc_type"] = DocType(doc["doc_type"])
    doc["source_format"] = SourceFormat(doc["source_format"])
    return DocumentRecord(**doc)


async def save_document_record(db: AsyncIOMotorDatabase, record: DocumentRecord) -> None:
    doc = asdict(record)
    doc["_id"] = record.id
    doc["created_at"] = datetime.now(timezone.utc)
    await db.courseMaterials.replace_one({"_id": record.id}, doc, upsert=True)
    logger.info("document record saved doc_id=%s doc_type=%s course_title=%r content_hash=%s",
                record.id, record.doc_type, record.course_title,
                (record.content_hash or "")[:12])


async def find_document_by_hash(db: AsyncIOMotorDatabase, content_hash: str) -> Optional[DocumentRecord]:
    """
    True identity check: same file bytes, regardless of what course_title
    text the uploader typed this time. Checked BEFORE the subject-text
    match below, so re-uploading the exact same file under a different
    subject name is recognized as a duplicate and never gets reprocessed
    (no re-parsing, no re-summarizing, no re-embedding — all of which cost
    real Claude/Gemini tokens).
    """
    if not content_hash:
        return None
    doc = await db.courseMaterials.find_one({"content_hash": content_hash})
    if not doc:
        return None
    return _to_document_record(doc)


async def find_document_by_id(db: AsyncIOMotorDatabase, doc_id: str) -> Optional[DocumentRecord]:
    """
    Precise lookup by _id — lets a roadmap's stored `grounded_doc_id` be
    trusted directly instead of re-doing a subject-text match on every
    generation call (see roadmap.py's `_resolve_grounding`). Returning None
    here (not raising) also serves as the staleness check: if the material
    was deleted since the roadmap was created, callers fall back to the
    subject-match path automatically.
    """
    if not doc_id:
        return None
    doc = await db.courseMaterials.find_one({"_id": doc_id})
    if not doc:
        return None
    return _to_document_record(doc)


async def find_document_for_subject(db: AsyncIOMotorDatabase, subject: str) -> Optional[DocumentRecord]:
    """
    Called before roadmap/notes generation: given the user-typed `subject`,
    find a matching uploaded course material (case-insensitive substring
    match on course_title/course_code — swap for a real search index once
    you have more than a handful of uploads).
    """
    if not subject:
        return None
    pattern = re.escape(subject)  # subject is free user text — must not be interpreted as a regex
    doc = await db.courseMaterials.find_one({
        "$or": [
            {"course_title": {"$regex": pattern, "$options": "i"}},
            {"course_code": {"$regex": pattern, "$options": "i"}},
        ]
    })
    if not doc:
        logger.info("find_document_for_subject: no match for subject=%r", subject)
        return None
    record = _to_document_record(doc)
    logger.info("find_document_for_subject: matched subject=%r -> doc_id=%s (course_title=%r)",
                subject, record.id, record.course_title)
    return record


async def save_tree(db: AsyncIOMotorDatabase, doc_id: str, nodes: List[TreeNode]) -> None:
    await db.courseMaterialTrees.replace_one(
        {"_id": doc_id},
        {"_id": doc_id, "nodes": [asdict(n) for n in nodes]},
        upsert=True,
    )
    logger.info("tree saved doc_id=%s nodes=%d", doc_id, len(nodes))


async def load_tree(db: AsyncIOMotorDatabase, doc_id: str) -> List[TreeNode]:
    doc = await db.courseMaterialTrees.find_one({"_id": doc_id})
    if not doc:
        logger.warning("load_tree: no tree found for doc_id=%s", doc_id)
        return []
    nodes = [TreeNode(**n) for n in doc["nodes"]]
    logger.info("tree loaded doc_id=%s nodes=%d", doc_id, len(nodes))
    return nodes
