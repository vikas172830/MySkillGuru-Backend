# ============================================================
# SELF-LEARNER NOTES — persisted history for AI Tutor's "Generate Notes"
# feature (app/api/routers/ai_tutor.py).
#
# Generated notes previously lived only in the Redis job record written by
# job_store.py (see set_job/update_job), which expires after JOB_TTL (1
# hour) with no way to list or revisit a past generation — a page refresh,
# or a return visit an hour later, lost the notes entirely even though the
# rendered PDF itself survives indefinitely on ImageKit. This collection is
# the durable record; the Redis job is unchanged and still drives the
# generate -> poll -> complete contract while a job is in flight.
#
# Plain-dict convention (no ODM), matching every other app/models/*.py file.
# ============================================================
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bson import ObjectId


def create_note_document(
    *,
    user_id: str,
    prompt: str,
    notes_type: str,
    notes_length: str,
    source_filename: Optional[str],
    html_content: str,
    solution_url: str,
    file_id: str,
    provider: str,
    model: str,
    grounded: bool,
    warning: Optional[str],
    token_usage: Dict[str, Any],
) -> Dict[str, Any]:
    # A prompt-only run has a natural title already; a file-only run (no
    # prompt, e.g. "just summarize this PDF") falls back to the filename so
    # the history list never shows a blank entry.
    title = prompt.strip() if prompt and prompt.strip() else (source_filename or "Untitled Notes")
    return {
        "user_id": ObjectId(user_id),
        "title": title[:200],
        "prompt": prompt,
        "notes_type": notes_type,
        "notes_length": notes_length,
        "source_filename": source_filename,
        "html_content": html_content,
        "solution_url": solution_url,
        "file_id": file_id,
        "provider": provider,
        "model": model,
        "grounded": grounded,
        "warning": warning,
        "token_usage": token_usage,
        "created_at": datetime.now(timezone.utc),
    }


def serialize_note(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Full record, html_content included — the single-note preview endpoint."""
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title"),
        "prompt": doc.get("prompt"),
        "notes_type": doc.get("notes_type"),
        "notes_length": doc.get("notes_length"),
        "source_filename": doc.get("source_filename"),
        "html_content": doc.get("html_content"),
        "solution_url": doc.get("solution_url"),
        "file_id": doc.get("file_id"),
        "provider": doc.get("provider"),
        "model": doc.get("model"),
        "grounded": doc.get("grounded", False),
        "warning": doc.get("warning"),
        "token_usage": doc.get("token_usage"),
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
    }


def serialize_note_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight record for the history list. html_content is deliberately
    omitted — a generated document runs tens of KB, and the list endpoint
    already excludes it at the query level (see list_notes' projection);
    this just keeps the two paths' shapes consistent regardless of what the
    caller queried for."""
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title"),
        "notes_type": doc.get("notes_type"),
        "notes_length": doc.get("notes_length"),
        "solution_url": doc.get("solution_url"),
        "grounded": doc.get("grounded", False),
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
    }
