"""Isolation from the LMS, which may share the same host, Redis server and
Qdrant cluster. Browsers scope cookies by host (not port), so these defaults
are what stop two products from overwriting each other's sessions or data."""

from app.core.config import Settings, settings
from app.core.redis_client import get_redis, prefixed, revoke_user_tokens
from app.services.job_store import set_job

LMS_COOKIE = "access_token_cookie"
LMS_QDRANT_COLLECTION = "lms_course_material_chunks"
LMS_DB_NAMES = {"nexus", "lms_evaluation"}


def test_code_defaults_never_match_the_lms():
    s = Settings(_env_file=None)
    assert s.JWT_COOKIE_NAME != LMS_COOKIE
    assert s.QDRANT_COLLECTION != LMS_QDRANT_COLLECTION
    assert s.DB_NAME not in LMS_DB_NAMES
    assert s.REDIS_KEY_PREFIX


def test_vector_store_uses_the_own_collection():
    from app.services.rag import vector_store

    assert vector_store.COLLECTION == settings.QDRANT_COLLECTION
    assert vector_store.COLLECTION != LMS_QDRANT_COLLECTION


async def test_every_redis_key_is_prefixed(client_factory, test_db):
    from tests.test_security_fixes import _seed_and_login_user

    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="Iso")
    assert (await learner.get("/me")).status_code == 200      # account-state cache
    await revoke_user_tokens("abc123", 60)                     # revocation key
    await set_job("iso-job:", "j1", {"status": "queued"})      # job state
    await learner.post("/logout")                              # rate-limit + revoke keys

    keys = [k async for k in get_redis().scan_iter(match="*")]
    assert keys, "expected the app to have written Redis keys"
    stray = [k for k in keys if not k.startswith(settings.REDIS_KEY_PREFIX)]
    assert not stray, f"Redis keys without the app prefix: {stray}"


def test_prefixed_helper():
    assert prefixed("x") == f"{settings.REDIS_KEY_PREFIX}x"
