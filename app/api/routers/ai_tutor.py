# ============================================================
# AI TUTOR ROUTER — Homework Help & Notes Generation
# Ported from controllers/institute/homework_help_controller.py +
# controllers/institute/notes_generate_controller.py — the implementation
# the frontend actually calls (confirmed via hardcoded absolute URLs in
# src/app/self-learner/ai-tutor/{homework-help,notes-generate}/page.js).
# The MongoDB-backed v1 (ai_tutor_controller.py) has no frontend caller and
# was deliberately not ported — see the Phase 3a plan.
#
# Deviation from Flask: these endpoints require authentication here
# (Depends(get_current_identity)) — Flask had none, flagged as a High
# severity finding in the original project analysis (unauthenticated
# callers could trigger billed Gemini/Claude calls).
#
# Response bodies/status codes are mirrored exactly (via JSONResponse)
# rather than left to FastAPI's default error shape, since the frontend
# calls these endpoints directly against a hardcoded backend URL and may
# depend on the exact field names.
# ============================================================

import asyncio
import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from typing import Optional, Set, Tuple

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.deps import get_current_identity, require_myskillguru_access
from app.core.rate_limit import ai_rate_limit
from app.db.mongodb import get_database
from app.models.ai_tutor_note import create_note_document, serialize_note, serialize_note_summary
from app.models.ai_usage_event import Feature, Provider
from app.services.ai_usage import record_ai_usage
from app.services.claude import generate_html
from app.services.gemini import (
    extract_text_from_file,
    generate_content_from_file_checked,
    generate_html_from_prompt,
)
# Shared provider-failover predicate — first introduced for roadmap
# generation, reused here rather than duplicated.
from app.services.roadmap_ai import is_gemini_quota_error
from app.services.imagekit import delete_imagekit_file, upload_file_to_imagekit
from app.services.job_store import get_job, set_job, update_job
from app.services.pdf_render import render_html_to_pdf
from app.services.rag.pdf_extract import extract_pdf_text

router = APIRouter(
    prefix="/api/ai-tutor",
    dependencies=[Depends(get_current_identity), Depends(require_myskillguru_access)],
    tags=["ai-tutor"],
)

HW_JOB_PREFIX = "hw_job:"
NOTES_JOB_PREFIX = "notes_job:"

# ============================================================
# FILE-TYPE HELPERS
# ============================================================

_EXT_MIME_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
}
_DOCX_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
_HOMEWORK_GEMINI_DIRECT_MIMES = {"application/pdf", "image/png", "image/jpeg", "image/jpg", "image/webp"}
_NOTES_GEMINI_DIRECT_MIMES = {"application/pdf"}
_HOMEWORK_ACCEPTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp", "docx", "doc"}
_NOTES_ACCEPTED_EXTENSIONS = {"pdf", "docx", "doc"}


def _get_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXT_MIME_MAP.get(ext, mimetypes.guess_type(filename)[0] or "application/octet-stream")


def _is_accepted(filename: str, accepted: Set[str]) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in accepted


_EMPTY_GEMINI_USAGE = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}

_HOMEWORK_EXTRACTION_PROMPT = (
    "Extract ALL text and questions from this homework document with high accuracy.\n"
    "Preserve question numbers, sub-questions, marks allocation, and any formulas.\n"
    "Return plain text only. Do not use markdown."
)

_STUDY_EXTRACTION_PROMPT = (
    "Extract ALL text from this study material/document with high accuracy.\n"
    "Preserve headings, subheadings, topics, definitions, formulas, and examples.\n"
    "Return plain text only. Do not use markdown."
)


def _empty_usage() -> dict:
    """Fresh copy per call — the returned dict flows into record_ai_usage and
    must never be a shared module-level object callers could mutate."""
    return dict(_EMPTY_GEMINI_USAGE)


def _extract_pdf_local_first(file_bytes: bytes, gemini_prompt: str) -> Tuple[str, dict, bool]:
    """
    Local-first PDF extraction, shared by both pipelines.

    pdfplumber pulls the embedded text layer straight out of the PDF for
    free, covering the large majority of uploads here (typed question banks,
    exported Word documents, digitally produced textbooks). Gemini OCR is
    spent only on the pages that come back empty — scanned or image-only —
    and even then in batches, so no single call risks the model's
    output-token ceiling regardless of how long the document is.

    Previously every PDF went to Gemini whole and uncapped, so anything
    larger than that ceiling came back silently truncated and notes were
    generated from whichever fragment happened to fit, with nothing
    surfaced to the student. This is the same fix app/services/rag/
    pdf_extract.py already applied on the RAG ingestion path; this router
    never received it.

    Falls back to the old whole-file Gemini call — now truncation-checked —
    only when pdfplumber cannot open the file at all (encrypted or
    malformed PDFs that it rejects but Gemini's vision path may still read),
    so this is strictly additive to what previously worked.
    """
    try:
        return extract_pdf_text(file_bytes)
    except Exception as e:
        logging.warning(
            "pdfplumber could not open PDF (%s) — falling back to whole-file Gemini extraction", e
        )
        return generate_content_from_file_checked(file_bytes, "application/pdf", gemini_prompt)


def _extract_homework_content(file_bytes: bytes, filename: str) -> Tuple[str, dict, bool]:
    """Returns (text, gemini_usage, truncated)."""
    mime_type = _get_mime(filename)

    if mime_type in _DOCX_MIMES:
        return extract_text_from_file(file_bytes, filename), _empty_usage(), False

    if mime_type == "application/pdf":
        return _extract_pdf_local_first(file_bytes, _HOMEWORK_EXTRACTION_PROMPT)

    if mime_type not in _HOMEWORK_GEMINI_DIRECT_MIMES:
        return "", _empty_usage(), False

    # Images (png/jpeg/webp) carry no text layer to read locally, so Gemini
    # vision stays the only option for them — sent whole, but now with the
    # truncation signal surfaced rather than discarded.
    return generate_content_from_file_checked(file_bytes, mime_type, _HOMEWORK_EXTRACTION_PROMPT)


def _extract_study_content(file_bytes: bytes, filename: str) -> Tuple[str, dict, bool]:
    """Returns (text, gemini_usage, truncated)."""
    mime_type = _get_mime(filename)

    if mime_type in _DOCX_MIMES:
        return extract_text_from_file(file_bytes, filename), _empty_usage(), False

    # Notes accept PDF/DOCX only (_NOTES_ACCEPTED_EXTENSIONS), so anything
    # reaching here that isn't a PDF has no extractor.
    if mime_type not in _NOTES_GEMINI_DIRECT_MIMES:
        return "", _empty_usage(), False

    return _extract_pdf_local_first(file_bytes, _STUDY_EXTRACTION_PROMPT)


# ============================================================
# CLAUDE PROMPT BUILDERS
# ============================================================

_HOMEWORK_TYPE_INSTRUCTIONS = {
    "Detailed Solution": "Provide a thorough, comprehensive solution with full explanations for every step.",
    "Short Explanation": "Provide concise, clear explanations. Get to the point quickly.",
    "Step By Step": "Break down the solution into clearly numbered steps. Each step on its own line.",
    "Summary Answer": "Give a brief summary answer. Focus only on the final result and key points.",
}

_RESPONSE_STYLE_INSTRUCTIONS = {
    "Simple": "Use simple, easy-to-understand language suitable for all levels.",
    "Student Friendly": "Use encouraging, friendly language with relatable examples.",
    "Technical": "Use precise technical language, terminology, and formal notation.",
    "Exam Style": "Format answers exactly as expected in an exam. Be concise and structured.",
}


def _build_homework_prompt(prompt: str, extracted_text: str, homework_type: str, response_style: str) -> str:
    current_date = datetime.now().strftime("%B %d, %Y")
    type_instr = _HOMEWORK_TYPE_INSTRUCTIONS.get(homework_type, _HOMEWORK_TYPE_INSTRUCTIONS["Detailed Solution"])
    style_instr = _RESPONSE_STYLE_INSTRUCTIONS.get(response_style, _RESPONSE_STYLE_INSTRUCTIONS["Simple"])

    homework_content = ""
    if extracted_text:
        homework_content = f"\n\nHOMEWORK CONTENT (extracted from uploaded file):\n{extracted_text}"
    if prompt:
        homework_content += f"\n\nADDITIONAL INSTRUCTIONS FROM STUDENT:\n{prompt}"

    return f"""You are an expert academic tutor helping a student with their homework.
    CURRENT DATE: {current_date}

HOMEWORK TYPE: {homework_type}
INSTRUCTION: {type_instr}

RESPONSE STYLE: {response_style}
INSTRUCTION: {style_instr}
{homework_content}

Generate a complete, standalone HTML document with the homework solution.

STYLING REQUIREMENTS:
- Clean, professional, student-friendly design
- Use a blue/violet color scheme (header: #1E1B4B or similar)
- Clearly numbered questions and answers
- Code blocks for programming answers (use <pre><code> tags)
- Mathematical expressions in readable format
- Print-friendly layout
- Font: Arial or similar sans-serif, 12pt body text

HTML STRUCTURE:
- Complete <!DOCTYPE html> document with <head> and inline <style>
- Header with "Homework Solution" title and generation date
-Show the exact generation date: {current_date}
- Each question clearly labeled (Q1, Q2, etc.)
- Answers in styled boxes/cards
- Footer: "Generated by Gradelytics AI Tutor"

Return ONLY the complete HTML document. No markdown fences, no explanations.
Start with <!DOCTYPE html> and end with </html>."""


_NOTES_TYPE_INSTRUCTIONS = {
    "Short Notes": (
        "Create concise, to-the-point notes. Use bullet points, key terms, and short explanations. "
        "Avoid unnecessary detail. Perfect for quick revision."
    ),
    "Detailed Notes": (
        "Create comprehensive, in-depth notes covering all topics thoroughly. "
        "Include definitions, explanations, examples, and sub-points. "
        "Suitable for deep understanding and exam preparation."
    ),
    "Presentation Style": (
        "Create notes in a slide/presentation format. Each topic as a heading, "
        "followed by 4-6 concise bullet points. Clean, visual, and scannable. "
        "Suitable for presenting or quick revision."
    ),
    "Summary Notes": (
        "Create a structured summary capturing only the most essential points. "
        "Each topic in 2-3 lines maximum. Great for last-minute revision."
    ),
}

# Output-token budgets per requested length. Bumped from the original flat
# 3000/5000/7000/5000: measured directly against this same prompt, a "5
# Pages" run needs ~3.4k-5.5k output tokens for a complete styled HTML
# document (table of contents, per-topic cards, takeaways section) — the
# original budget left generation hitting the cap and returning HTML cut off
# mid-<style>, which renders as a BLANK preview (an unterminated style block
# swallows everything after it) rather than merely a short one. Matters even
# more now that prompt-only notes route to Gemini (_generate_notes_html
# below): gemini-2.5-flash is a thinking model whose internal reasoning
# shares this same budget with the visible output.
_NOTES_LENGTH_TOKENS = {"5 Pages": 20000, "10 Pages": 32000, "15 Pages": 48000, "Custom": 24000}


def _build_notes_prompt(prompt: str, extracted_text: str, notes_type: str, notes_length: str) -> str:
    current_date = datetime.now().strftime("%B %d, %Y")
    type_instr = _NOTES_TYPE_INSTRUCTIONS.get(notes_type, _NOTES_TYPE_INSTRUCTIONS["Short Notes"])

    study_content = ""
    if extracted_text:
        study_content += f"\n\nSTUDY MATERIAL (extracted from uploaded file):\n{extracted_text}"
    if prompt:
        study_content += f"\n\nTOPIC / INSTRUCTIONS FROM STUDENT:\n{prompt}"

    return f"""You are an expert academic notes creator and study assistant.
CURRENT DATE: {current_date}

NOTES TYPE: {notes_type}
INSTRUCTION: {type_instr}

NOTES LENGTH: {notes_length}
{study_content}

Generate a complete, standalone HTML document with well-structured academic notes.

═══════════════════════════════════════════════
CONTENT REQUIREMENTS
═══════════════════════════════════════════════
1. Cover all major topics from the study material / prompt
2. Use clear headings for each topic (H2) and sub-topics (H3)
3. Include:
   - Key definitions with proper formatting
   - Important formulas in readable format
   - Examples where relevant
   - Bullet points for lists and properties
   - Numbered lists for steps/processes
4. For programming/code topics: use <pre><code> blocks
5. End with a "Key Takeaways" or "Quick Revision" section

═══════════════════════════════════════════════
HTML & STYLING REQUIREMENTS
═══════════════════════════════════════════════
- Complete <!DOCTYPE html> document with inline <style> in <head>
- Color scheme: Deep navy header (#1E1B4B), violet accents (#7C3AED)
- Clean white cards for each topic section
- Header: "Study Notes" title + topic name + generation date ({current_date})
- Each topic in a styled card/box with left border accent
- Definitions highlighted with a light background
- Formulas in a distinct styled block
- Code in dark-background code blocks
- Footer: "Generated by Gradelytics AI Tutor | {current_date}"
- Print-friendly (good margins, readable fonts)
- Font: Arial or similar, 12pt body

═══════════════════════════════════════════════
STRUCTURE TEMPLATE
═══════════════════════════════════════════════
Header → Table of Contents → Topic sections (each as a card) → Key Takeaways → Footer

Return ONLY the complete HTML document.
No markdown fences. No explanations outside HTML.
Start with <!DOCTYPE html> and end with </html>."""


# ============================================================
# BACKGROUND JOBS
# ============================================================

# Surfaced on the completed job (and through the status endpoint) whenever
# extraction came back short, so a student reading partial output is told
# so rather than being left to assume their whole document was covered.
# Additive key — existing frontend callers that don't read it are unaffected.
_TRUNCATION_WARNING = (
    "This document was too large to read in full — some pages could not be extracted, "
    "so the result may not cover the entire document."
)

_GEMINI_NOTES_MODEL = "gemini-2.5-flash"
# MySkillGuru's own existing Claude model choice for notes/homework — kept
# as-is rather than switched to whatever lms-evaluation-backend currently
# uses, since that's an unrelated model-pin change, not part of this port.
_CLAUDE_NOTES_MODEL = "claude-sonnet-4-20250514"


async def _generate_notes_html(
    job_id: str, full_prompt: str, max_tokens: int, has_document: bool,
) -> Tuple[str, dict, str, str]:
    """
    Picks the provider for a notes run and returns
    (html, usage, provider, model) so the caller records against whichever
    one actually served the request.

    Prompt-only notes ("make me notes on photosynthesis") are written from
    the model's own knowledge, which Gemini Flash handles well at a fraction
    of the cost — that path is the majority of notes traffic and does not
    need Claude. Document-backed notes stay on Claude: those require
    reasoning over a long piece of source material the student supplied, and
    faithfulness to that text is worth more than the saving.

    Gemini quota exhaustion fails over to Claude rather than failing the
    job, mirroring roadmap.py's treatment of the same situation. Only quota
    errors fail over — see is_gemini_quota_error.
    """
    if has_document:
        logging.info("[notes:%s] Document-backed — calling Claude…", job_id)
        html, usage = await asyncio.to_thread(
            generate_html, full_prompt, _CLAUDE_NOTES_MODEL, max_tokens
        )
        logging.info("[notes:%s] Claude done — %d tokens", job_id, usage["total_tokens"])
        return html, usage, Provider.CLAUDE, _CLAUDE_NOTES_MODEL

    try:
        logging.info("[notes:%s] Prompt-only — calling Gemini…", job_id)
        html, usage = await asyncio.to_thread(
            generate_html_from_prompt, full_prompt, _GEMINI_NOTES_MODEL, max_tokens
        )
        logging.info("[notes:%s] Gemini done — %d tokens", job_id, usage["total_tokens"])
        return html, usage, Provider.GEMINI, _GEMINI_NOTES_MODEL
    except Exception as e:
        if not is_gemini_quota_error(e):
            raise
        logging.warning("[notes:%s] Gemini quota exhausted — failing over to Claude: %s", job_id, e)

    html, usage = await asyncio.to_thread(
        generate_html, full_prompt, _CLAUDE_NOTES_MODEL, max_tokens
    )
    logging.info("[notes:%s] Claude failover done — %d tokens", job_id, usage["total_tokens"])
    return html, usage, Provider.CLAUDE, _CLAUDE_NOTES_MODEL


async def _run_homework_job(
    job_id: str, params: dict, file_bytes: Optional[bytes], filename: str,
    db: AsyncIOMotorDatabase, user_id: str,
) -> None:
    try:
        extracted_text = ""
        warning = None
        if file_bytes:
            await update_job(HW_JOB_PREFIX, job_id, {"status": "processing", "step": "extracting_homework"})
            logging.info("[hw:%s] Extracting file content…", job_id)
            extracted_text, extract_usage, extract_truncated = await asyncio.to_thread(
                _extract_homework_content, file_bytes, filename
            )
            logging.info("[hw:%s] Extraction done — %d chars", job_id, len(extracted_text))
            if extract_truncated:
                warning = _TRUNCATION_WARNING
                logging.warning(
                    "[hw:%s] Extraction hit Gemini's output-token cap — the solution covers only "
                    "the part of the document that could be read", job_id,
                )
            await record_ai_usage(
                db, user_id=user_id, provider=Provider.GEMINI, model="gemini-2.5-flash",
                feature=Feature.SELF_REVIEW_HOMEWORK_EXTRACTION, usage=extract_usage, job_id=job_id,
            )

        await update_job(HW_JOB_PREFIX, job_id, {"step": "generating_solution"})
        logging.info("[hw:%s] Calling Claude…", job_id)
        full_prompt = _build_homework_prompt(
            params["prompt"], extracted_text, params["homeworkType"], params["responseStyle"]
        )
        claude_model = _CLAUDE_NOTES_MODEL
        html_content, c_usage = await asyncio.to_thread(generate_html, full_prompt, claude_model, 5000)
        logging.info("[hw:%s] Claude done — %d tokens", job_id, c_usage["total_tokens"])
        await record_ai_usage(
            db, user_id=user_id, provider=Provider.CLAUDE, model=claude_model,
            feature=Feature.SELF_REVIEW_HOMEWORK_HELP, usage=c_usage, job_id=job_id,
        )

        await update_job(HW_JOB_PREFIX, job_id, {"step": "building_pdf"})
        logging.info("[hw:%s] Rendering PDF…", job_id)
        pdf_binary = await asyncio.to_thread(render_html_to_pdf, html_content)
        logging.info("[hw:%s] PDF built — %d bytes", job_id, len(pdf_binary))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        upload = await asyncio.to_thread(
            upload_file_to_imagekit, pdf_binary, f"homework_solution_{ts}.pdf",
            "/homework-solutions", ["homework", "ai-tutor"],
        )
        logging.info("[hw:%s] Uploaded -> %s", job_id, upload["url"])

        await update_job(HW_JOB_PREFIX, job_id, {
            "status": "completed",
            "step": "done",
            "solution_url": upload["url"],
            "file_id": upload["file_id"],
            "html_content": html_content,
            "token_usage": c_usage,
            "warning": warning,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        logging.info("[hw:%s] Job completed.", job_id)

    except Exception as e:
        logging.error("[hw:%s] Job failed: %s", job_id, e, exc_info=True)
        await update_job(HW_JOB_PREFIX, job_id, {"status": "failed", "step": "error", "error": str(e)})


async def _run_notes_job(
    job_id: str, params: dict, file_bytes: Optional[bytes], filename: str,
    db: AsyncIOMotorDatabase, user_id: str,
) -> None:
    try:
        extracted_text = ""
        warning = None
        if file_bytes:
            await update_job(NOTES_JOB_PREFIX, job_id, {"status": "processing", "step": "extracting_notes"})
            logging.info("[notes:%s] Extracting file content…", job_id)
            extracted_text, extract_usage, extract_truncated = await asyncio.to_thread(
                _extract_study_content, file_bytes, filename
            )
            logging.info("[notes:%s] Extraction done — %d chars", job_id, len(extracted_text))
            if extract_truncated:
                warning = _TRUNCATION_WARNING
                logging.warning(
                    "[notes:%s] Extraction hit Gemini's output-token cap — these notes cover only "
                    "the part of the document that could be read", job_id,
                )
            await record_ai_usage(
                db, user_id=user_id, provider=Provider.GEMINI, model="gemini-2.5-flash",
                feature=Feature.SELF_REVIEW_NOTES_EXTRACTION, usage=extract_usage, job_id=job_id,
            )

        await update_job(NOTES_JOB_PREFIX, job_id, {"step": "generating_notes"})
        full_prompt = _build_notes_prompt(
            params["prompt"], extracted_text, params["notesType"], params["notesLength"]
        )
        max_tokens = _NOTES_LENGTH_TOKENS.get(params["notesLength"], 5000)
        html_content, c_usage, provider, model = await _generate_notes_html(
            job_id, full_prompt, max_tokens, has_document=bool(extracted_text)
        )
        await record_ai_usage(
            db, user_id=user_id, provider=provider, model=model,
            feature=Feature.SELF_REVIEW_NOTES, usage=c_usage, job_id=job_id,
            grounded=bool(extracted_text),
        )

        await update_job(NOTES_JOB_PREFIX, job_id, {"step": "building_pdf"})
        logging.info("[notes:%s] Rendering PDF…", job_id)
        pdf_binary = await asyncio.to_thread(render_html_to_pdf, html_content)
        logging.info("[notes:%s] PDF built — %d bytes", job_id, len(pdf_binary))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        upload = await asyncio.to_thread(
            upload_file_to_imagekit, pdf_binary, f"ai_notes_{ts}.pdf", "/ai-notes", ["notes", "ai-tutor"]
        )
        logging.info("[notes:%s] Uploaded -> %s", job_id, upload["url"])

        # Durable record. The Redis job above is the polling contract while
        # generation is in flight and expires after JOB_TTL (1 hour) — this
        # is what makes the result reachable afterward: a page refresh, a
        # return visit, the history list. Best-effort: the PDF is already
        # generated and uploaded by this point, so a Mongo hiccup here must
        # not turn an already-successful generation into a failed job — the
        # student still gets html_content/solution_url back from this same
        # response, just without a history entry. note_id stays None in
        # that case, which the frontend can treat as "not saved."
        note_id = None
        try:
            note_doc = create_note_document(
                user_id=user_id,
                prompt=params["prompt"],
                notes_type=params["notesType"],
                notes_length=params["notesLength"],
                source_filename=filename or None,
                html_content=html_content,
                solution_url=upload["url"],
                file_id=upload["file_id"],
                provider=provider,
                model=model,
                grounded=bool(extracted_text),
                warning=warning,
                token_usage=c_usage,
            )
            insert_result = await db.selfLearnerNotes.insert_one(note_doc)
            note_id = str(insert_result.inserted_id)
        except Exception as e:
            logging.error("[notes:%s] Failed to persist note to history: %s", job_id, e, exc_info=True)

        await update_job(NOTES_JOB_PREFIX, job_id, {
            "status": "completed",
            "step": "done",
            "solution_url": upload["url"],
            "file_id": upload["file_id"],
            "html_content": html_content,
            "token_usage": c_usage,
            "warning": warning,
            "note_id": note_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        logging.info("[notes:%s] Job completed.", job_id)

    except Exception as e:
        logging.error("[notes:%s] Job failed: %s", job_id, e, exc_info=True)
        await update_job(NOTES_JOB_PREFIX, job_id, {"status": "failed", "step": "error", "error": str(e)})


# ============================================================
# ROUTES — HOMEWORK HELP
# ============================================================

@router.post("/homework-help", dependencies=[Depends(ai_rate_limit)])
async def homework_help(
    background_tasks: BackgroundTasks,
    prompt: str = Form(""),
    homeworkType: str = Form("Detailed Solution"),
    responseStyle: str = Form("Simple"),
    file: Optional[UploadFile] = File(None),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    try:
        prompt = prompt.strip()
        homework_type = homeworkType.strip()
        response_style = responseStyle.strip()

        if not prompt and not file:
            return JSONResponse(status_code=400, content={"error": "Please enter homework details or upload a file."})

        file_bytes = None
        filename = ""
        if file:
            fname = file.filename or "upload"
            if not _is_accepted(fname, _HOMEWORK_ACCEPTED_EXTENSIONS):
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "unknown"
                return JSONResponse(status_code=400, content={
                    "error": f"Unsupported file type '.{ext}'. Accepted: PDF, DOCX, PNG, JPG, JPEG, WEBP."
                })
            file_bytes = await file.read()
            filename = fname

        job_id = str(uuid.uuid4())
        await set_job(HW_JOB_PREFIX, job_id, {
            "status": "processing", "step": "starting", "job_id": job_id,
            "user_id": identity["user_id"],
        })

        params = {"prompt": prompt, "homeworkType": homework_type, "responseStyle": response_style}
        background_tasks.add_task(_run_homework_job, job_id, params, file_bytes, filename, db, identity["user_id"])

        logging.info("Homework job %s started.", job_id)
        return JSONResponse(status_code=202, content={
            "success": True,
            "jobId": job_id,
            "message": "Generation started. Poll /api/ai-tutor/homework-help/status/<jobId>.",
        })

    except Exception as e:
        logging.error("homework_help error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/homework-help/status/{job_id}")
async def homework_help_status(job_id: str, identity: dict = Depends(get_current_identity)):
    job = await get_job(HW_JOB_PREFIX, job_id)
    if job is None or job.get("user_id") != identity["user_id"]:
        return JSONResponse(status_code=404, content={"error": "Job not found. It may have expired or never existed."})

    status = job.get("status", "unknown")

    if status == "completed":
        return {
            "success": True,
            "status": "completed",
            "step": "done",
            "solution_url": job.get("solution_url"),
            "file_id": job.get("file_id"),
            "html_content": job.get("html_content"),
            "token_usage": job.get("token_usage"),
            # None on a normal run; set only when extraction came back short.
            "warning": job.get("warning"),
            "generated_at": job.get("generated_at"),
        }

    if status == "failed":
        # 200, not 500: the status check itself succeeded — the job failed.
        # Returning 5xx makes the frontend poller treat it as "couldn't reach
        # the server" instead of surfacing job.error.
        return JSONResponse(status_code=200, content={
            "success": False, "status": "failed", "error": job.get("error", "Unknown error."),
        })

    return JSONResponse(status_code=202, content={
        "success": True, "status": status, "step": job.get("step", ""), "message": "Generation in progress...",
    })


# ============================================================
# ROUTES — GENERATE NOTES
# ============================================================

@router.post("/generate-notes", dependencies=[Depends(ai_rate_limit)])
async def generate_notes(
    background_tasks: BackgroundTasks,
    prompt: str = Form(""),
    notesType: str = Form("Short Notes"),
    notesLength: str = Form("5 Pages"),
    file: Optional[UploadFile] = File(None),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    try:
        prompt = prompt.strip()
        notes_type = notesType.strip()
        notes_length = notesLength.strip()

        if not prompt and not file:
            return JSONResponse(status_code=400, content={"error": "Please enter a topic or upload study material."})

        file_bytes = None
        filename = ""
        if file:
            fname = file.filename or "upload"
            if not _is_accepted(fname, _NOTES_ACCEPTED_EXTENSIONS):
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "unknown"
                return JSONResponse(status_code=400, content={
                    "error": f"Unsupported file type '.{ext}'. Accepted: PDF, DOCX."
                })
            file_bytes = await file.read()
            filename = fname

        job_id = str(uuid.uuid4())
        await set_job(NOTES_JOB_PREFIX, job_id, {
            "status": "processing", "step": "starting", "job_id": job_id,
            "user_id": identity["user_id"],
        })

        params = {"prompt": prompt, "notesType": notes_type, "notesLength": notes_length}
        background_tasks.add_task(_run_notes_job, job_id, params, file_bytes, filename, db, identity["user_id"])

        logging.info("Notes job %s started — type=%s length=%s", job_id, notes_type, notes_length)
        return JSONResponse(status_code=202, content={
            "success": True,
            "jobId": job_id,
            "message": "Generation started. Poll /api/ai-tutor/generate-notes/status/<jobId>.",
        })

    except Exception as e:
        logging.error("generate_notes error: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/generate-notes/status/{job_id}")
async def generate_notes_status(job_id: str, identity: dict = Depends(get_current_identity)):
    job = await get_job(NOTES_JOB_PREFIX, job_id)
    if job is None or job.get("user_id") != identity["user_id"]:
        return JSONResponse(status_code=404, content={"error": "Job not found. It may have expired or never existed."})

    status = job.get("status", "unknown")

    if status == "completed":
        return {
            "success": True,
            "status": "completed",
            "step": "done",
            "solution_url": job.get("solution_url"),
            "file_id": job.get("file_id"),
            "html_content": job.get("html_content"),
            "token_usage": job.get("token_usage"),
            # None on a normal run; set only when extraction came back short.
            "warning": job.get("warning"),
            # Id of the durable selfLearnerNotes record this run created —
            # None if persistence failed (see _run_notes_job), in which case
            # html_content/solution_url above are still this response's own
            # copy and remain usable, they just won't show up in history.
            "note_id": job.get("note_id"),
            "generated_at": job.get("generated_at"),
        }

    if status == "failed":
        # 200, not 500: the status check itself succeeded — the job failed.
        # Returning 5xx makes the frontend poller treat it as "couldn't reach
        # the server" instead of surfacing job.error.
        return JSONResponse(status_code=200, content={
            "success": False, "status": "failed", "error": job.get("error", "Unknown error."),
        })

    return JSONResponse(status_code=202, content={
        "success": True, "status": status, "step": job.get("step", ""), "message": "Generation in progress...",
    })


# ============================================================
# ROUTES — NOTES HISTORY (persisted in Mongo — see app/models/ai_tutor_note.py)
# ============================================================

@router.get("/notes")
async def list_notes(
    limit: int = Query(50, ge=1, le=200),
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    # html_content excluded at the query level, not just the serializer —
    # a single generated document runs tens of KB, and a list of up to 200
    # of them has no use for the full body until one is actually opened.
    cursor = db.selfLearnerNotes.find(
        {"user_id": ObjectId(identity["user_id"])},
        {"html_content": 0},
    ).sort("created_at", -1).limit(limit)

    notes = [serialize_note_summary(doc) async for doc in cursor]
    return {"success": True, "notes": notes}


@router.get("/notes/{note_id}")
async def get_note(
    note_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=404, detail="Note not found")

    doc = await db.selfLearnerNotes.find_one(
        {"_id": ObjectId(note_id), "user_id": ObjectId(identity["user_id"])}
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Note not found")

    return {"success": True, "note": serialize_note(doc)}


@router.delete("/notes/{note_id}")
async def delete_note(
    note_id: str,
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=404, detail="Note not found")

    # find_one_and_delete rather than delete_one so the ImageKit file_id is
    # available for cleanup below without a second round trip.
    doc = await db.selfLearnerNotes.find_one_and_delete(
        {"_id": ObjectId(note_id), "user_id": ObjectId(identity["user_id"])}
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Note not found")

    file_id = doc.get("file_id")
    if file_id:
        try:
            await asyncio.to_thread(delete_imagekit_file, file_id)
        except Exception as e:
            # The Mongo record is already gone — the note is deleted from
            # the student's point of view either way. An orphaned PDF on
            # ImageKit is a storage-cost concern, not a correctness one, so
            # this is logged and swallowed rather than surfaced as a 500.
            logging.warning(
                "delete_note: ImageKit cleanup failed for file_id=%s (note_id=%s): %s",
                file_id, note_id, e,
            )

    return {"success": True}
