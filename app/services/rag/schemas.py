# ============================================================
# Shared data contracts for the hybrid RAG layer (course-material grounding
# for AI roadmap generation). Ported from the Flask prototype's
# rag/models/schemas.py — pure dataclasses, no change needed for FastAPI.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4


class DocType(str, Enum):
    STRUCTURED = "structured"      # course profile, syllabus, assessment plan
    UNSTRUCTURED = "unstructured"  # textbook, reference material, notes dump


class SourceFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MD = "md"
    TXT = "txt"


@dataclass
class TreeNode:
    """A node in a structure-derived document tree (PageIndex-lite)."""
    id: str
    title: str
    level: int                      # heading depth: 1 = H1/section, 2 = subsection...
    parent_id: Optional[str]
    page_start: int
    page_end: int
    raw_content: str                # full text of just this node (tables included whole)
    is_table: bool = False
    summary: str = ""               # 1-line summary, filled at index-build time
    children_ids: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A retrieval unit for the vector path."""
    id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentRecord:
    id: str
    filename: str
    source_format: SourceFormat
    doc_type: DocType                # decided by the structure heuristic
    course_code: Optional[str] = None
    course_title: Optional[str] = None
    content_hash: Optional[str] = None  # sha256 of file bytes — true identity, independent of what course_title text was typed


@dataclass
class RetrievalResult:
    """What the retriever hands to the generation layer."""
    context_text: str               # the actual content to inject into the prompt
    source_nodes: list[str]         # node/chunk ids used, for traceability/logging
    confidence: float                # 0-1, used to decide RAG-grounded vs fallback
    doc_id: Optional[str] = None


def new_id() -> str:
    return str(uuid4())
