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
    # Records written before ownership existed have no owner_user_ids field
    # at all; let the dataclass default supply an empty list rather than
    # raising on a missing kwarg.
    doc.setdefault("owner_user_ids", [])
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


async def find_document_by_id(
    db: AsyncIOMotorDatabase, doc_id: str, owner_user_id: Optional[str] = None
) -> Optional[DocumentRecord]:
    """
    Precise lookup by _id — lets a roadmap's stored `grounded_doc_id` be
    trusted directly instead of re-doing a subject-text match on every
    generation call (see roadmap.py's `_resolve_grounding`). Returning None
    here (not raising) also serves as the staleness check: if the material
    was deleted since the roadmap was created, callers fall back to the
    subject-match path automatically.

    Pass owner_user_id whenever the doc_id came from the client. Roadmap
    creation accepts one straight off the request body, so without this
    check any caller could name another user's doc_id and read its contents
    back out through generated notes, practice questions or auto-tests.
    Access is deliberately expressed as "returns None" rather than an
    exception so that path stays identical to a deleted or unknown document
    — callers already fall back to ungrounded generation, and a distinct
    error would tell an attacker their guessed id exists.

    Legacy records carrying no owner_user_ids match no owner and so resolve
    to None; scripts/backfill_course_material_owners.py recovers ownership
    for every document a roadmap still references.
    """
    if not doc_id:
        return None

    query: dict = {"_id": doc_id}
    if owner_user_id is not None:
        query["owner_user_ids"] = owner_user_id

    doc = await db.courseMaterials.find_one(query)
    if not doc:
        return None
    return _to_document_record(doc)


async def add_document_owner(db: AsyncIOMotorDatabase, doc_id: str, user_id: str) -> None:
    """
    Grant user_id access to an already-indexed document. Called when an
    upload dedups against existing bytes: the uploader clearly holds the
    file, so they get access to the indexed copy, and the expensive
    parse/summarize/embed work is not repeated. $addToSet keeps it
    idempotent across repeat uploads by the same user.
    """
    if not doc_id or not user_id:
        return
    await db.courseMaterials.update_one({"_id": doc_id}, {"$addToSet": {"owner_user_ids": user_id}})
    logger.info("document owner added doc_id=%s user_id=%s", doc_id, user_id)


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
