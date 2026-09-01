# ============================================================
# Local-first PDF text extraction for course-material RAG ingestion.
#
# Most uploaded PDFs (typed syllabi, exported Word/Docs, digitally-created
# textbooks) already carry an embedded text layer — pdfplumber pulls it out
# directly, for free, with no LLM token limit. Only pages with no
# extractable text (scanned/image-only pages) fall back to Gemini OCR, and
# even then only those pages are sent — batched as contiguous runs of a few
# pages at a time — so no single Gemini call risks hitting gemini-2.5-flash's
# 65,536-token output cap regardless of total document length.
#
# Previously EVERY uploaded PDF was sent to Gemini whole, uncapped, which
# both silently truncated large documents (no truncation detection existed)
# and spent an LLM call on documents that never needed one.
# ============================================================
from __future__ import annotations

import io
import logging
from typing import Dict, List, Tuple

import pdfplumber
import PyPDF2
from google.genai import types as genai_types

from app.services.gemini import _call_generate_content_with_retry, _get_client, _parse_token_usage

logger = logging.getLogger("rag.pdf_extract")

_MIN_CHARS_PER_PAGE = 20  # below this, a page is treated as having no real text layer (scanned/image)
_OCR_BATCH_PAGES = 15     # pages per Gemini OCR call — keeps output comfortably under the 65K-token cap


def extract_pdf_text(file_bytes: bytes) -> Tuple[str, dict, bool]:
    """
    Returns (text, usage, truncated).

    usage is a token-usage dict (all zeros if no Gemini calls were needed —
    i.e. the whole document already had a real text layer). truncated is
    True if any OCR batch hit Gemini's MAX_TOKENS output cap, meaning that
    batch's text may be incomplete.

    Blocking — call via asyncio.to_thread() from async callers.
    """
    pages_text: List[str] = []
    scanned_indices: List[int] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            pages_text.append(text)
            if len(text) < _MIN_CHARS_PER_PAGE:
                scanned_indices.append(i)

    total_usage = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
    truncated = False

    if scanned_indices:
        logger.info(
            "extract_pdf_text: %d/%d pages have no text layer — OCR fallback via Gemini",
            len(scanned_indices), len(pages_text),
        )
        ocr_text_by_page, ocr_usage, truncated = _ocr_scanned_pages(file_bytes, scanned_indices)
        for key in total_usage:
            total_usage[key] += ocr_usage.get(key, 0)
        for idx, text in ocr_text_by_page.items():
            pages_text[idx] = text

    return "\n\n".join(t for t in pages_text if t), total_usage, truncated


def _group_contiguous(indices: List[int], max_batch: int) -> List[List[int]]:
    """Groups page indices into contiguous runs (capped at max_batch each) so
    a batch's OCR'd text lands back in roughly the right position in the
    document instead of jumping between unrelated pages."""
    batches: List[List[int]] = []
    current: List[int] = []
    for idx in indices:
        if current and (idx != current[-1] + 1 or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(idx)
    if current:
        batches.append(current)
    return batches


def _is_truncated(response) -> bool:
    try:
        finish_reason = response.candidates[0].finish_reason
        return finish_reason is not None and "MAX_TOKENS" in str(finish_reason)
    except Exception:
        return False


def _ocr_scanned_pages(file_bytes: bytes, page_indices: List[int]) -> Tuple[Dict[int, str], dict, bool]:
    """OCR only the pages that lack a text layer, batched as contiguous runs
    of at most _OCR_BATCH_PAGES so no single Gemini call risks the model's
    output-token ceiling regardless of total document length."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    client = _get_client()

    results: Dict[int, str] = {}
    total_usage = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
    truncated = False

    for batch in _group_contiguous(page_indices, _OCR_BATCH_PAGES):
        writer = PyPDF2.PdfWriter()
        for idx in batch:
            writer.add_page(reader.pages[idx])
        buf = io.BytesIO()
        writer.write(buf)

        batch_part = genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="application/pdf")
        response = _call_generate_content_with_retry(
            client, "gemini-2.5-flash",
            [batch_part, "Extract ALL text from these course document pages with high accuracy. "
                         "Preserve the original structure including section headings, numbering, and tables. "
                         "Return plain text only. Do not use markdown."],
        )

        text = ""
        try:
            text = response.text.strip()
        except Exception:
            pass

        if _is_truncated(response):
            truncated = True
            logger.warning("extract_pdf_text: OCR batch pages=%s hit MAX_TOKENS — text may be incomplete", batch)

        usage = _parse_token_usage(getattr(response, "usage_metadata", None))
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

        # The whole document is already flattened into one page downstream
        # (see self_learner_course_material.py) so exact per-page placement
        # doesn't need to survive past this function — landing a batch's
        # combined text at its first page's slot and leaving the rest blank
        # keeps document order correct without needing to split Gemini's
        # response back apart per page.
        results[batch[0]] = text
        for idx in batch[1:]:
            results[idx] = ""

    return results, total_usage, truncated
