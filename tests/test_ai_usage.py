# ============================================================
# Coverage for the AI Usage Events ledger (app/services/ai_usage.py):
# the four usage-shape normalizer, the aiUsageEvents + users.token_usage
# dual-write, the best-effort/non-fatal contract every other tracking
# helper in this codebase follows, and the price table's cost math.
# ============================================================
from types import SimpleNamespace

from bson import ObjectId

from app.services.ai_usage import _extract_tokens, ensure_ai_usage_indexes, record_ai_usage
from app.utils.ai_pricing import estimate_cost_usd


# ---------------------------------------------------------------- _extract_tokens

def test_extract_tokens_from_claude_dict_shape():
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert _extract_tokens(usage) == (100, 50)


def test_extract_tokens_from_gemini_dict_shape():
    usage = {"prompt_tokens": 200, "candidate_tokens": 75, "total_tokens": 275}
    assert _extract_tokens(usage) == (200, 75)


def test_extract_tokens_from_anthropic_sdk_object_shape():
    usage = SimpleNamespace(input_tokens=30, output_tokens=20)
    assert _extract_tokens(usage) == (30, 20)


def test_extract_tokens_from_gemini_sdk_metadata_shape():
    usage = SimpleNamespace(prompt_token_count=40, candidates_token_count=10, total_token_count=50)
    assert _extract_tokens(usage) == (40, 10)


def test_extract_tokens_handles_none():
    assert _extract_tokens(None) == (0, 0)


def test_extract_tokens_handles_missing_fields_as_zero():
    assert _extract_tokens({}) == (0, 0)
    assert _extract_tokens(SimpleNamespace()) == (0, 0)


# ---------------------------------------------------------------- estimate_cost_usd

def test_estimate_cost_usd_known_model():
    # claude-sonnet-4-6: $3/$15 per 1M input/output tokens.
    cost = estimate_cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 18.0


def test_estimate_cost_usd_unknown_model_returns_zero():
    assert estimate_cost_usd("some-future-model-nobody-has-priced-yet", 1000, 1000) == 0.0


def test_estimate_cost_usd_zero_tokens_is_zero():
    assert estimate_cost_usd("claude-sonnet-4-6", 0, 0) == 0.0


# ---------------------------------------------------------------- record_ai_usage

async def _seed_user(test_db) -> str:
    result = await test_db["users"].insert_one({"fullName": "Test Learner", "email": "learner@test.local", "role": 7})
    return str(result.inserted_id)


async def test_record_ai_usage_writes_event_and_rollup(test_db):
    user_id = await _seed_user(test_db)

    await record_ai_usage(
        test_db, user_id=user_id, provider="claude", model="claude-sonnet-4-6",
        feature="roadmap_curriculum", usage={"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
    )

    events = [e async for e in test_db["aiUsageEvents"].find({"user_id": ObjectId(user_id)})]
    assert len(events) == 1
    event = events[0]
    assert event["provider"] == "claude"
    assert event["model"] == "claude-sonnet-4-6"
    assert event["feature"] == "roadmap_curriculum"
    assert event["input_tokens"] == 500
    assert event["output_tokens"] == 200
    assert event["total_tokens"] == 700
    assert event["cost_usd"] > 0
    assert event["tenant_type"] == "individual"
    assert event["institute_id"] is None

    user = await test_db["users"].find_one({"_id": ObjectId(user_id)})
    assert user["token_usage"]["claude"]["input_tokens"] == 500
    assert user["token_usage"]["claude"]["output_tokens"] == 200


async def test_record_ai_usage_accumulates_rollup_across_calls(test_db):
    user_id = await _seed_user(test_db)

    for _ in range(3):
        await record_ai_usage(
            test_db, user_id=user_id, provider="gemini", model="gemini-2.5-flash",
            feature="roadmap_notes", usage={"prompt_tokens": 10, "candidate_tokens": 5, "total_tokens": 15},
        )

    user = await test_db["users"].find_one({"_id": ObjectId(user_id)})
    assert user["token_usage"]["gemini"]["input_tokens"] == 30
    assert user["token_usage"]["gemini"]["output_tokens"] == 15

    events = [e async for e in test_db["aiUsageEvents"].find({"user_id": ObjectId(user_id)})]
    assert len(events) == 3


async def test_record_ai_usage_skips_zero_token_usage(test_db):
    # e.g. a local-extraction path (docx) that never actually called an AI
    # model — must not pollute the ledger with a zero-cost row.
    user_id = await _seed_user(test_db)

    await record_ai_usage(
        test_db, user_id=user_id, provider="gemini", model="gemini-2.5-flash",
        feature="self_review_homework_extraction",
        usage={"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0},
    )

    events = [e async for e in test_db["aiUsageEvents"].find({"user_id": ObjectId(user_id)})]
    assert events == []

    user = await test_db["users"].find_one({"_id": ObjectId(user_id)})
    assert "token_usage" not in user


async def test_record_ai_usage_is_non_fatal_on_bad_user_id(test_db):
    # Must never raise and break the AI feature it's tracking — matches
    # every other tracking helper's best-effort contract in this codebase.
    await record_ai_usage(
        test_db, user_id="not-a-valid-object-id", provider="claude", model="claude-sonnet-4-6",
        feature="roadmap_curriculum", usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    assert await test_db["aiUsageEvents"].count_documents({}) == 0


async def test_record_ai_usage_institute_context_sets_tenant_type(test_db):
    user_id = await _seed_user(test_db)
    institute_id = str(ObjectId())
    school_id = str(ObjectId())

    await record_ai_usage(
        test_db, user_id=user_id, provider="gemini", model="gemini-2.5-flash",
        feature="rag_ingest_extraction", usage={"prompt_tokens": 100, "candidate_tokens": 0, "total_tokens": 100},
        institute_id=institute_id, school_id=school_id,
    )

    event = await test_db["aiUsageEvents"].find_one({"user_id": ObjectId(user_id)})
    assert event["tenant_type"] == "institute"
    assert event["institute_id"] == ObjectId(institute_id)
    assert event["school_id"] == ObjectId(school_id)


async def test_ensure_ai_usage_indexes_is_idempotent(test_db):
    await ensure_ai_usage_indexes(test_db)
    await ensure_ai_usage_indexes(test_db)  # must not raise on the second call

    index_info = await test_db["aiUsageEvents"].index_information()
    assert len(index_info) >= 4  # the default _id index plus the three created
