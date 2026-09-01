# ============================================================
# SELF-LEARNER COURSE MATERIAL ROUTER
#
# The RAG ingestion pipeline (hash-dedup -> extract -> classify
# STRUCTURED/UNSTRUCTURED -> tree-index or chunk+embed) that grounds a
# self-learner's roadmap in a syllabus/textbook they upload at creation
# time (see roadmap/create's "Ground it in your own material" step).
#
# Originally shared with an institute/faculty-facing course-material
# router via a common _run_ingest_job() helper; that router isn't part of
# MySkillGuru's scope (institute/faculty features removed), so the
# ingestion job now lives here directly rather than being imported
# cross-file from a router that no longer exists.
#
# Mounted at /api/self-learner/course-material — one of the two
# prefix-preserving rewrite rules in next.config.mjs (the other being
# /api/self-learner/roadmap), since this router's own url_prefix already
# includes "/api".
# ============================================================

import asyncio
import hashlib
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.deps import get_current_identity, require_myskillguru_access
from app.core.rate_limit import ai_rate_limit
from app.db.mongodb import get_database
from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.gemini import extract_text_from_file
from app.services.job_store import get_job, set_job, update_job
from app.services.rag import mongo_store, singletons, structure_parser, tree_index
from app.services.rag.pdf_extract import extract_pdf_text
from app.services.rag.schemas import DocType, DocumentRecord, SourceFormat, new_id
from app.services.rag.vector_store import chunk_text

router = APIRouter(
    prefix="/api/self-learner/course-material",
    dependencies=[Depends(get_current_identity), Depends(require_myskillguru_access)],
    tags=["self-learner-course-material"],
)

logger = logging.getLogger(__name__)

SL_CM_JOB_PREFIX = "self_learner_course_material_job:"

_EXT_TO_FORMAT = {
    "pdf": SourceFormat.PDF, "docx": SourceFormat.DOCX, "doc": SourceFormat.DOCX,
    "md": SourceFormat.MD, "txt": SourceFormat.TXT,
}


def _file_ext(filename: str) -> str:
    return (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()


# ============================================================
# BACKGROUND JOB — INGEST (extract -> classify -> tree/vector index)
# ============================================================

async def _run_ingest_job(
    job_id: str, file_bytes: bytes, filename: str,
    course_title: Optional[str], course_code: Optional[str], user_id: str,
    job_prefix: str = SL_CM_JOB_PREFIX,
) -> None:
    db = get_database()
    try:
        await update_job(job_prefix, job_id, {"step": "Checking for duplicates…"})

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = await mongo_store.find_document_by_hash(db, content_hash)
        if existing:
            logger.info("course_material ingest: doc_id=%s is a duplicate of existing doc_id=%s (hash match)",
                        job_id, existing.id)
            await update_job(job_prefix, job_id, {
                "status": "done", "doc_id": existing.id, "duplicate": True, "step": "Already indexed",
            })
            return

        await update_job(job_prefix, job_id, {"step": "Extracting text…"})

        ext = _file_ext(filename)
        source_format = _EXT_TO_FORMAT.get(ext, SourceFormat.TXT)

        if ext == "pdf":
            # Local-first: pdfplumber pulls the text layer straight out of
            # the PDF, no LLM call, no token limit — covers the large
            # majority of uploads (typed syllabi, exported docs, digital
            # textbooks). Gemini OCR is only invoked for pages that come
            # back empty (scanned/image-only), and even then batched a few
            # pages at a time so no single call risks Gemini's 65,536-token
            # output cap regardless of total document length.
            text, extract_usage, extract_truncated = await asyncio.to_thread(extract_pdf_text, file_bytes)
            await record_ai_usage(
                db, user_id=user_id, provider=Provider.GEMINI, model="gemini-2.5-flash",
                feature=Feature.RAG_INGEST_EXTRACTION, usage=extract_usage, job_id=job_id,
            )
            if extract_truncated:
                logger.warning(
                    "course_material ingest: doc_id=%s OCR extraction hit Gemini's output-token cap on at "
                    "least one page batch — extracted text for those pages may be incomplete", job_id,
                )
        else:
            text = await asyncio.to_thread(extract_text_from_file, file_bytes, filename)

        if not text or not text.strip():
            await update_job(job_prefix, job_id, {
                "status": "error", "error": "No text could be extracted from this file.",
            })
            return

        # No true per-page-with-tables extraction utility exists on the
        # FastAPI side (this backend's Gemini/docx extractors return one
        # flat string, not paginated dicts) — treat the whole document as a
        # single page. Fine for the small structured course-outline
        # documents this feature targets; heading-density classification
        # still works at document granularity.
        pages = [{"text": text, "page_num": 1, "tables": []}]

        doc_id = new_id()
        doc_type = structure_parser.classify_doc_type(pages)
        await update_job(job_prefix, job_id, {"step": f"Indexing as {doc_type.value}…"})

        if doc_type == DocType.STRUCTURED:
            nodes = structure_parser.build_tree(pages, doc_id)
            await tree_index.summarize_nodes(nodes, db=db, user_id=user_id)
            await mongo_store.save_tree(db, doc_id, nodes)
        else:
            store = await asyncio.to_thread(singletons.get_vector_store)
            if store is None:
                await update_job(job_prefix, job_id, {
                    "status": "error",
                    "error": "Vector store (Qdrant) is unavailable — start it and try again, or upload a "
                             "more structured document (with numbered headings) to use the tree-index path instead.",
                })
                return
            chunks = chunk_text(doc_id, pages)
            embed_usage = await asyncio.to_thread(store.upsert, chunks)
            await record_ai_usage(
                db, user_id=user_id, provider=Provider.GEMINI, model="gemini-embedding-001",
                feature=Feature.RAG_EMBEDDING, usage=embed_usage, job_id=job_id,
            )

        record = DocumentRecord(
            id=doc_id, filename=filename, source_format=source_format, doc_type=doc_type,
            course_code=course_code, course_title=course_title, content_hash=content_hash,
        )
        await mongo_store.save_document_record(db, record)

        logger.info("course_material ingest done: doc_id=%s doc_type=%s course_title=%r",
                    doc_id, doc_type.value, course_title)
        await update_job(job_prefix, job_id, {
            "status": "done", "doc_id": doc_id, "doc_type": doc_type.value, "duplicate": False, "step": "Done",
        })

    except Exception as e:
        logger.error("course_material ingest job %s failed: %s", job_id, e, exc_info=True)
        await update_job(job_prefix, job_id, {
            "status": "error", "error": "Internal server error during course material indexing.",
        })


# ============================================================
# ROUTES
# ============================================================

@router.get("/status/{job_id}")
async def get_upload_status(job_id: str, identity: dict = Depends(get_current_identity)):
    job = await get_job(SL_CM_JOB_PREFIX, job_id)
    if job is None or job.get("user_id") != identity["user_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("", dependencies=[Depends(ai_rate_limit)])
async def upload_course_material(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    course_title: Optional[str] = Form(None),
    course_code: Optional[str] = Form(None),
    identity: dict = Depends(get_current_identity),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    job_id = str(uuid.uuid4())
    await set_job(SL_CM_JOB_PREFIX, job_id, {
        "status": "processing", "step": "Starting…", "user_id": identity["user_id"],
    })

    background_tasks.add_task(
        _run_ingest_job, job_id, file_bytes, file.filename or "upload",
        course_title, course_code, identity["user_id"],
        SL_CM_JOB_PREFIX,
    )

    logger.info(
        "self-learner course material upload queued: job_id=%s filename=%r user_id=%s",
        job_id, file.filename, identity["user_id"],
    )
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing"})
