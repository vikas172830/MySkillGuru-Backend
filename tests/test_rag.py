# ============================================================
# Phase 9 (hybrid RAG) coverage: the heading/table-density classifier and
# tree-building heuristic in structure_parser.py (pure functions, no
# mocking), the tree/vector dispatch in retrieval/router.py, and the two
# retrievers themselves. app.services.claude.generate_text and
# VectorStore.search are always mocked/faked here — no real Claude, Qdrant,
# or Gemini call in this file.
# ============================================================
from unittest.mock import AsyncMock, patch

from bson import ObjectId

from app.services.rag import mongo_store
from app.services.rag.retrieval import router, tree_retriever, vector_retriever
from app.services.rag.schemas import DocType, RetrievalResult, TreeNode
from app.services.rag.structure_parser import build_tree, classify_doc_type

_GENERATE_TEXT_PATCH = "app.services.rag.retrieval.tree_retriever.generate_text"


def _page(text: str, page_num: int = 1, tables=None) -> dict:
    return {"text": text, "page_num": page_num, "tables": tables or []}


# ---------------------------------------------------------------- classify_doc_type

def test_prose_textbook_is_unstructured():
    pages = [_page("This is a long paragraph of narrative prose. " * 50)]
    assert classify_doc_type(pages) == DocType.UNSTRUCTURED


def test_dense_numbered_headings_is_structured():
    text = "\n".join(f"{i}. Section {i}" for i in range(1, 5))
    assert classify_doc_type([_page(text)]) == DocType.STRUCTURED


def test_markdown_headings_count_too():
    text = "\n".join(f"## Heading {i}" for i in range(1, 5))
    assert classify_doc_type([_page(text)]) == DocType.STRUCTURED


def test_real_table_density_alone_is_structured():
    # Below the heading-density threshold on its own, but table_density >=
    # 0.15 is sufficient by itself per the OR in classify_doc_type.
    pages = [_page("Ordinary prose with no headings at all. " * 20, tables=["| a | b |"])]
    assert classify_doc_type(pages) == DocType.STRUCTURED


def test_single_stray_heading_in_long_doc_stays_unstructured():
    # Guards the "effective page count via word count" heuristic described
    # in the docstring: a single-page flat-text extraction with exactly one
    # numbered line must not alone trip STRUCTURED once the doc is long.
    text = "1. Introduction\n" + ("Long prose paragraph without further structure. " * 300)
    assert classify_doc_type([_page(text)]) == DocType.UNSTRUCTURED


def test_empty_pages_default_unstructured():
    assert classify_doc_type([_page("")]) == DocType.UNSTRUCTURED


# ---------------------------------------------------------------- build_tree

def test_single_top_level_heading():
    pages = [_page("1. Course Overview\nSome content here.")]
    nodes = build_tree(pages, doc_id="doc1")
    assert len(nodes) == 1
    assert nodes[0].title == "Course Overview"
    assert nodes[0].level == 1
    assert nodes[0].parent_id is None
    assert "Some content here." in nodes[0].raw_content


def test_nested_headings_set_parent_and_children():
    pages = [_page(
        "1. Course Overview\n"
        "Top-level content.\n"
        "1.1 Objectives\n"
        "Nested content.\n"
    )]
    nodes = build_tree(pages, doc_id="doc1")
    assert len(nodes) == 2
    parent, child = nodes
    assert parent.level == 1
    assert child.level == 2
    assert child.parent_id == parent.id
    assert child.id in parent.children_ids


def test_markdown_heading_levels_use_hash_count():
    pages = [_page("# Top\nA\n## Sub\nB")]
    nodes = build_tree(pages, doc_id="doc1")
    assert [n.level for n in nodes] == [1, 2]


def test_sibling_at_shallower_level_closes_deeper_node():
    pages = [_page(
        "1 Section One\n"
        "1.1 Sub A\n"
        "content a\n"
        "2 Section Two\n"
        "content b\n"
    )]
    nodes = build_tree(pages, doc_id="doc1")
    section_two = next(n for n in nodes if n.title == "Section Two")
    assert section_two.level == 1
    assert section_two.parent_id is None  # not nested under Sub A


def test_table_attaches_to_open_node_and_is_not_split():
    pages = [
        {"text": "1. Assessment\nintro line", "page_num": 1, "tables": []},
        {"text": "more text", "page_num": 2, "tables": ["| Component | Weight |\n| Exam | 60% |"]},
    ]
    nodes = build_tree(pages, doc_id="doc1")
    assert len(nodes) == 1
    assert nodes[0].is_table is True
    assert "Component" in nodes[0].raw_content
    assert nodes[0].page_start == 1
    assert nodes[0].page_end == 2


def test_no_headings_falls_back_to_single_root_node():
    pages = [_page("Just plain text, no headings at all.")]
    nodes = build_tree(pages, doc_id="doc1")
    assert len(nodes) == 1
    assert nodes[0].title == "Document"
    assert nodes[0].parent_id is None
    assert nodes[0].raw_content == pages[0]["text"]


# ---------------------------------------------------------------- router.retrieve

async def test_structured_doc_dispatches_to_tree_retriever():
    expected = RetrievalResult(context_text="ctx", source_nodes=["n1"], confidence=0.9, doc_id="doc1")
    with patch.object(router.tree_retriever, "retrieve", new=AsyncMock(return_value=expected)) as mock_tree, \
         patch.object(router.vector_retriever, "retrieve", new=AsyncMock()) as mock_vector:
        result = await router.retrieve("query", "doc1", DocType.STRUCTURED, db=None)
    mock_tree.assert_awaited_once()
    mock_vector.assert_not_called()
    assert result is expected


async def test_unstructured_doc_without_vector_store_is_ungrounded():
    with patch.object(router.tree_retriever, "retrieve", new=AsyncMock()) as mock_tree, \
         patch.object(router.vector_retriever, "retrieve", new=AsyncMock()) as mock_vector:
        result = await router.retrieve("query", "doc1", DocType.UNSTRUCTURED, db=None, vector_store=None)
    mock_tree.assert_not_called()
    mock_vector.assert_not_called()
    assert result.context_text == ""
    assert result.confidence == 0.0


async def test_unstructured_doc_with_vector_store_dispatches_to_vector_retriever():
    expected = RetrievalResult(context_text="ctx", source_nodes=["c1"], confidence=0.6, doc_id="doc1")
    fake_store = object()
    with patch.object(router.tree_retriever, "retrieve", new=AsyncMock()) as mock_tree, \
         patch.object(router.vector_retriever, "retrieve", new=AsyncMock(return_value=expected)) as mock_vector:
        result = await router.retrieve("query", "doc1", DocType.UNSTRUCTURED, db=None, vector_store=fake_store)
    mock_tree.assert_not_called()
    mock_vector.assert_awaited_once_with("query", "doc1", fake_store, db=None, user_id=None)
    assert result is expected


def test_should_use_rag_below_min_confidence_is_false():
    result = RetrievalResult(context_text="something", source_nodes=[], confidence=0.34, doc_id="d")
    assert router.should_use_rag(result) is False


def test_should_use_rag_at_min_confidence_is_true():
    result = RetrievalResult(context_text="something", source_nodes=[], confidence=0.35, doc_id="d")
    assert router.should_use_rag(result) is True


def test_should_use_rag_empty_context_is_false_even_at_high_confidence():
    result = RetrievalResult(context_text="", source_nodes=[], confidence=0.99, doc_id="d")
    assert router.should_use_rag(result) is False


# ---------------------------------------------------------------- tree_retriever

async def test_tree_retriever_missing_tree_returns_empty_without_calling_claude(test_db):
    with patch(_GENERATE_TEXT_PATCH) as mock_generate:
        result = await tree_retriever.retrieve("query", "missing-doc", test_db)
    mock_generate.assert_not_called()
    assert result == RetrievalResult(context_text="", source_nodes=[], confidence=0.0, doc_id="missing-doc")


async def test_tree_retriever_picks_nodes_from_well_formed_response(test_db):
    nodes = [
        TreeNode(id="n1", title="Grading", level=1, parent_id=None, page_start=1, page_end=1,
                  raw_content="Exams are 60% of the grade.", summary="Grading breakdown"),
        TreeNode(id="n2", title="Schedule", level=1, parent_id=None, page_start=2, page_end=2,
                  raw_content="Class meets Tuesdays.", summary="Meeting schedule"),
    ]
    await mongo_store.save_tree(test_db, "doc1", nodes)
    fake_response = ('{"node_ids": ["n1"], "confidence": 0.8}', {"input_tokens": 10, "output_tokens": 5})

    with patch(_GENERATE_TEXT_PATCH, return_value=fake_response) as mock_generate:
        result = await tree_retriever.retrieve("How is grading weighted?", "doc1", test_db)

    mock_generate.assert_called_once()
    assert result.source_nodes == ["n1"]
    assert result.confidence == 0.8
    assert "Exams are 60%" in result.context_text
    assert "Class meets Tuesdays" not in result.context_text


async def test_tree_retriever_malformed_json_falls_back_to_empty(test_db):
    nodes = [TreeNode(id="n1", title="X", level=1, parent_id=None, page_start=1, page_end=1, raw_content="content")]
    await mongo_store.save_tree(test_db, "doc1", nodes)
    fake_response = ("not valid json at all", {})

    with patch(_GENERATE_TEXT_PATCH, return_value=fake_response):
        result = await tree_retriever.retrieve("query", "doc1", test_db)

    assert result.source_nodes == []
    assert result.confidence == 0.0
    assert result.context_text == ""


async def test_tree_retriever_drops_unknown_node_ids(test_db):
    nodes = [TreeNode(id="n1", title="X", level=1, parent_id=None, page_start=1, page_end=1, raw_content="content")]
    await mongo_store.save_tree(test_db, "doc1", nodes)
    fake_response = ('{"node_ids": ["n1", "does-not-exist"], "confidence": 0.5}', {})

    with patch(_GENERATE_TEXT_PATCH, return_value=fake_response):
        result = await tree_retriever.retrieve("query", "doc1", test_db)

    assert result.source_nodes == ["n1"]


async def test_tree_retriever_empty_pick_list_forces_zero_confidence(test_db):
    # Confidence in the response is ignored unless at least one node was
    # actually picked (`confidence if picked else 0.0` in tree_retriever.py).
    nodes = [TreeNode(id="n1", title="X", level=1, parent_id=None, page_start=1, page_end=1, raw_content="content")]
    await mongo_store.save_tree(test_db, "doc1", nodes)
    fake_response = ('{"node_ids": [], "confidence": 0.9}', {})

    with patch(_GENERATE_TEXT_PATCH, return_value=fake_response):
        result = await tree_retriever.retrieve("query", "doc1", test_db)

    assert result.source_nodes == []
    assert result.confidence == 0.0


# ---------------------------------------------------------------- vector_retriever

class _FakeHit:
    def __init__(self, id_, text, score):
        self.id = id_
        self.payload = {"text": text}
        self.score = score


class _FakeVectorStore:
    def __init__(self, hits, usage=None):
        self._hits = hits
        self._usage = usage or {"input_tokens": 3, "output_tokens": 0}
        self.calls = []

    def search(self, query, doc_id, top_k):
        self.calls.append((query, doc_id, top_k))
        return self._hits, self._usage


async def test_vector_retriever_no_hits_returns_empty_result():
    store = _FakeVectorStore(hits=[])
    result = await vector_retriever.retrieve("query", "doc1", store)
    assert result.context_text == ""
    assert result.confidence == 0.0
    assert result.source_nodes == []


async def test_vector_retriever_joins_hits_and_uses_top_score_as_confidence():
    hits = [_FakeHit("c1", "first chunk", 0.91), _FakeHit("c2", "second chunk", 0.4)]
    store = _FakeVectorStore(hits=hits)

    result = await vector_retriever.retrieve("query", "doc1", store, top_k=4)

    assert result.confidence == 0.91  # top hit's score, not an average
    assert "first chunk" in result.context_text
    assert "second chunk" in result.context_text
    assert result.source_nodes == ["c1", "c2"]
    assert store.calls == [("query", "doc1", 4)]


async def test_vector_retriever_tracks_query_embedding_usage_when_db_and_user_given(test_db):
    store = _FakeVectorStore(hits=[_FakeHit("c1", "chunk", 0.5)], usage={"input_tokens": 7, "output_tokens": 0})
    await test_db["users"].insert_one({"_id": ObjectId("507f1f77bcf86cd799439011"), "role": 7})

    await vector_retriever.retrieve("query", "doc1", store, db=test_db, user_id="507f1f77bcf86cd799439011")

    event = await test_db["aiUsageEvents"].find_one({"feature": "rag_retrieve"})
    assert event is not None
    assert event["input_tokens"] == 7
    assert event["provider"] == "gemini"


async def test_vector_retriever_skips_tracking_without_db_or_user_id():
    # Default call signature (no db/user_id) must keep working unchanged —
    # every existing caller before this feature was added relies on that.
    store = _FakeVectorStore(hits=[_FakeHit("c1", "chunk", 0.5)])
    result = await vector_retriever.retrieve("query", "doc1", store)
    assert result.confidence == 0.5


# ---------------------------------------------------------------- document ownership
#
# Access to uploaded course material is scoped to its owners. Dedup stays
# global (identity is the file's bytes), so ownership is a set: uploading a
# file someone else already indexed adds you to it and reuses their indexed
# copy, rather than paying for a second parse/summarize/embed pass.

_OWNER = "507f1f77bcf86cd799439011"
_STRANGER = "507f1f77bcf86cd799439012"


async def _insert_material(test_db, doc_id: str, owners=None) -> None:
    doc = {
        "_id": doc_id,
        "id": doc_id,
        "filename": "syllabus.pdf",
        "source_format": "pdf",
        "doc_type": "structured",
        "content_hash": f"hash-of-{doc_id}",
    }
    if owners is not None:
        doc["owner_user_ids"] = owners
    await test_db.courseMaterials.insert_one(doc)


async def test_owner_can_resolve_own_document(test_db):
    await _insert_material(test_db, "doc-owned", owners=[_OWNER])
    found = await mongo_store.find_document_by_id(test_db, "doc-owned", owner_user_id=_OWNER)
    assert found is not None
    assert found.owner_user_ids == [_OWNER]


async def test_stranger_cannot_resolve_another_users_document(test_db):
    """The IDOR this closes: roadmap creation takes doc_id straight off the
    request body, so an unowned id must resolve to None — indistinguishable
    from one that doesn't exist, and generation continues ungrounded."""
    await _insert_material(test_db, "doc-owned", owners=[_OWNER])
    assert await mongo_store.find_document_by_id(test_db, "doc-owned", owner_user_id=_STRANGER) is None


async def test_lookup_without_owner_arg_stays_unscoped(test_db):
    """Internal callers that already established access don't pass an owner."""
    await _insert_material(test_db, "doc-owned", owners=[_OWNER])
    assert await mongo_store.find_document_by_id(test_db, "doc-owned") is not None


async def test_legacy_document_without_owners_resolves_to_none(test_db):
    """Records predating ownership have no owner_user_ids and match nobody;
    scripts/backfill_course_material_owners.py recovers the ones roadmaps
    still reference."""
    await _insert_material(test_db, "doc-legacy")  # field absent entirely
    assert await mongo_store.find_document_by_id(test_db, "doc-legacy", owner_user_id=_OWNER) is None

    legacy = await mongo_store.find_document_by_id(test_db, "doc-legacy")
    assert legacy is not None and legacy.owner_user_ids == []


async def test_add_document_owner_is_idempotent_and_additive(test_db):
    await _insert_material(test_db, "doc-shared", owners=[_OWNER])

    await mongo_store.add_document_owner(test_db, "doc-shared", _STRANGER)
    await mongo_store.add_document_owner(test_db, "doc-shared", _STRANGER)  # repeat upload

    doc = await test_db.courseMaterials.find_one({"_id": "doc-shared"})
    assert sorted(doc["owner_user_ids"]) == sorted([_OWNER, _STRANGER])

    # Both parties can now resolve it; neither lost access.
    assert await mongo_store.find_document_by_id(test_db, "doc-shared", owner_user_id=_OWNER) is not None
    assert await mongo_store.find_document_by_id(test_db, "doc-shared", owner_user_id=_STRANGER) is not None


async def test_dedup_stays_global_across_owners(test_db):
    """find_document_by_hash must NOT be owner-scoped — that is what keeps a
    shared textbook to a single indexing pass instead of one per student."""
    await _insert_material(test_db, "doc-shared", owners=[_OWNER])
    found = await mongo_store.find_document_by_hash(test_db, "hash-of-doc-shared")
    assert found is not None and found.id == "doc-shared"
