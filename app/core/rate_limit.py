# ============================================================
# RATE LIMITING
# Fixed-window counters in Redis (same client as app/services/job_store.py),
# so tests get this for free through the existing autouse fakeredis fixture
# in tests/conftest.py — no separate test wiring needed.
#
# Two application points:
#   - RateLimitByIP / RateLimitByUser: FastAPI dependencies for specific
#     routes (auth endpoints and the AI-cost endpoints respectively).
#   - GlobalRateLimitMiddleware: a loose per-IP safety net applied to every
#     request, registered in app/main.py.
# ============================================================

import logging

from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.api.deps import get_current_identity
from app.core.config import settings
from app.core.redis_client import get_redis

WINDOW_SECONDS = 60

_RATE_LIMIT_MESSAGE = "Too many requests. Please slow down and try again shortly."


async def _increment(key: str) -> int:
    """Fixed-window counter: first hit in a window sets the expiry, every
    hit after that just increments. Not perfectly atomic against the rare
    race between INCR and EXPIRE on the very first request in a window, but
    that only risks a slightly-longer-lived key, never an under-count — an
    acceptable tradeoff for the same reason the existing token counters in
    app/utils/token_usage.py accept eventual-consistency-shaped edges."""
    redis = get_redis()
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, WINDOW_SECONDS)
    return count


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RateLimitByIP:
    """Per-IP limiter for unauthenticated endpoints (login, contact)."""

    def __init__(self, limit: int, name: str):
        self.limit = limit
        self.name = name

    async def __call__(self, request: Request) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        key = f"ratelimit:{self.name}:ip:{_client_ip(request)}"
        count = await _increment(key)
        if count > self.limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=_RATE_LIMIT_MESSAGE)


class RateLimitByUser:
    """Per-user limiter for authenticated, cost-bearing AI endpoints."""

    def __init__(self, limit: int, name: str):
        self.limit = limit
        self.name = name

    async def __call__(self, identity: dict = Depends(get_current_identity)) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        key = f"ratelimit:{self.name}:user:{identity['user_id']}"
        count = await _increment(key)
        if count > self.limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=_RATE_LIMIT_MESSAGE)


class GlobalRateLimitMiddleware(BaseHTTPMiddleware):
    """Loose per-IP safety net applied to every request, so no single client
    can flood the server regardless of which endpoint it targets. The tight,
    endpoint-specific limits (RateLimitByIP/RateLimitByUser above) are what
    actually protects the billed AI endpoints — this is just a floor under
    everything else."""

    async def dispatch(self, request: Request, call_next):
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        try:
            key = f"ratelimit:global:ip:{_client_ip(request)}"
            count = await _increment(key)
        except Exception:
            # Redis hiccup must never take the whole app down — fail open.
            logging.exception("GlobalRateLimitMiddleware: Redis check failed, allowing request")
            return await call_next(request)

        if count > settings.RATE_LIMIT_GLOBAL_PER_MINUTE:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": _RATE_LIMIT_MESSAGE, "detail": _RATE_LIMIT_MESSAGE},
            )

        return await call_next(request)


# Ready-made instances — import these directly into routers.
login_rate_limit = RateLimitByIP(settings.RATE_LIMIT_AUTH_PER_MINUTE, "auth")
contact_rate_limit = RateLimitByIP(settings.RATE_LIMIT_AUTH_PER_MINUTE, "contact")
# Own name/key so a burst of MySkillGuru signups never throttles login
# attempts (or vice versa) from the same IP.
myskillguru_register_rate_limit = RateLimitByIP(settings.RATE_LIMIT_AUTH_PER_MINUTE, "myskillguru-register")
ai_rate_limit = RateLimitByUser(settings.RATE_LIMIT_AI_PER_MINUTE, "ai")
bulk_grading_rate_limit = RateLimitByUser(settings.RATE_LIMIT_BULK_GRADING_PER_MINUTE, "bulk-grading")
