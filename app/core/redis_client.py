import json
import logging
import time

import redis.asyncio as redis_asyncio

from app.core.config import settings

_client: redis_asyncio.Redis | None = None


async def connect_to_redis() -> None:
    """Create the client and verify the server answers.

    If it doesn't: in production keep the real client and log loudly (rate
    limits, job tracking and logout revocation will fail until Redis is back).
    In development fall back to an in-memory fake so the app is runnable on a
    laptop that can't reach the shared Redis (state is lost on restart).
    """
    global _client
    client = redis_asyncio.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=3,  # fail fast instead of hanging on an unreachable host
        socket_timeout=5,
    )
    try:
        await client.ping()
        _client = client
        logging.info("Redis connection verified")
        return
    except Exception as exc:  # noqa: BLE001 - any connection failure is handled the same way
        if settings.is_production:
            logging.error("Redis unreachable (%s): rate limiting, job tracking and session revocation will fail", exc)
            _client = client
            return
        await client.aclose()
        try:
            from fakeredis import FakeAsyncRedis
        except ImportError:
            logging.error("Redis unreachable (%s) and fakeredis is not installed (pip install -r requirements-dev.txt)", exc)
            _client = redis_asyncio.from_url(settings.REDIS_URL, decode_responses=True)
            return
        logging.warning("Redis unreachable (%s): using an in-memory fake. Development only; state is lost on restart.", exc)
        _client = FakeAsyncRedis(decode_responses=True)


async def close_redis_connection() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


def prefixed(key: str) -> str:
    """The only way a Redis key should be built — applies REDIS_KEY_PREFIX."""
    return f"{settings.REDIS_KEY_PREFIX}{key}"


def get_redis() -> redis_asyncio.Redis:
    if _client is None:
        raise RuntimeError("Redis client not initialized — call connect_to_redis() first")
    return _client


# ============================================================
# TOKEN REVOCATION
# The JWT is a stateless cookie, so logout (or a password change) can't
# invalidate an already-issued token on its own. A per-user cutoff timestamp
# is recorded here; app.api.deps.get_current_identity rejects any token whose
# `iat` predates it. Fails open on Redis errors — a Redis outage must not
# lock every user out (the DB is_active check still applies).
# ============================================================

_REVOKED_KEY = "revoked_before:{user_id}"


async def revoke_user_tokens(user_id: str, ttl_seconds: int) -> None:
    """Invalidate every token issued for user_id up to now. The TTL matches the
    max token lifetime, so the key self-expires once no pre-cutoff token could
    still be valid anyway."""
    try:
        await get_redis().setex(prefixed(_REVOKED_KEY.format(user_id=user_id)), ttl_seconds, repr(time.time()))
    except Exception:
        logging.warning("revoke_user_tokens: Redis unavailable, user_id=%s not revoked", user_id)


async def tokens_revoked_after(user_id: str) -> float | None:
    """Unix-seconds cutoff for user_id, or None when nothing is recorded or
    Redis is unreachable (fail-open)."""
    try:
        raw = await get_redis().get(prefixed(_REVOKED_KEY.format(user_id=user_id)))
        return float(raw) if raw else None
    except Exception:
        return None


# ============================================================
# ACCOUNT-STATE CACHE
# get_current_identity needs to know, per request, whether the account is
# still active/undeleted. Caching that tiny state for a short window keeps
# Mongo off the hot path. Fails OPEN on any Redis/Mongo error.
# ============================================================

_ACCOUNT_STATE_KEY = "acctstate:{user_id}"
ACCOUNT_STATE_TTL = 30  # seconds — deactivation / delete takes effect within this window


async def get_cached_account_state(user_id: str):
    try:
        raw = await get_redis().get(prefixed(_ACCOUNT_STATE_KEY.format(user_id=user_id)))
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def set_cached_account_state(user_id: str, state: dict) -> None:
    try:
        await get_redis().setex(prefixed(_ACCOUNT_STATE_KEY.format(user_id=user_id)), ACCOUNT_STATE_TTL, json.dumps(state))
    except Exception:
        pass


async def bust_account_state(user_id: str) -> None:
    """Drop the cached state now (e.g. after a reactivation)."""
    try:
        await get_redis().delete(prefixed(_ACCOUNT_STATE_KEY.format(user_id=user_id)))
    except Exception:
        pass
