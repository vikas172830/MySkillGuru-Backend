import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import (
    ai_tutor,
    auth,
    mock_tests,
    myskillguru_auth,
    profile,
    roadmap,
    self_learner_analytics,
    self_learner_course_material,
)
from app.core.config import settings
from app.core.rate_limit import GlobalRateLimitMiddleware
from app.core.redis_client import close_redis_connection, connect_to_redis
from app.db.mongodb import close_mongo_connection, connect_to_mongo, get_database, ping_mongo
from app.services.ai_usage import ensure_ai_usage_indexes

logging.basicConfig(level=logging.INFO)
logging.getLogger("pymongo").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn forces WindowsSelectorEventLoopPolicy on Windows (its default
    # asyncio loop backend, since uvloop isn't available there), which has no
    # subprocess transport — any code that spawns a subprocess raises
    # NotImplementedError. Playwright (used for PDF rendering) needs one to
    # launch its browser. Overriding the policy back to Proactor here — after
    # uvicorn's own startup has already set Selector and started its main
    # loop — doesn't touch that already-running loop, but does mean any NEW
    # loop created later (which is exactly what Playwright's sync API spins
    # up internally, per-call, in its own thread) picks up Proactor instead.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    connect_to_mongo()
    await ping_mongo()
    await ensure_ai_usage_indexes(get_database())
    connect_to_redis()
    yield
    await close_redis_connection()
    close_mongo_connection()


app = FastAPI(title="LMS Evaluation API", lifespan=lifespan)

# Registered before CORSMiddleware so CORS ends up as the outer layer (FastAPI
# wraps middleware in reverse registration order) — otherwise a 429 response
# short-circuited by GlobalRateLimitMiddleware would go out with no CORS
# headers, and the browser would block it before the frontend ever saw it.
app.add_middleware(GlobalRateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FastAPI's HTTPException(detail=...) natively produces {"detail": ...}, but
# the Flask backend (and most of the frontend's error-reading code, e.g.
# `err?.response?.data?.error`) expects {"error": ...}. Rather than touching
# every one of the ~340 raise HTTPException(...) call sites across the
# routers, add both keys here in one place so existing frontend error
# handling keeps showing the real backend message instead of falling back
# to generic hardcoded text. Endpoints that already build their own
# JSONResponse with an "error" key (e.g. ai_tutor.py) bypass this handler
# entirely and are unaffected.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        body = dict(exc.detail)
        # Preserve the real payload even when it uses none of the recognized
        # keys, instead of discarding it behind a hardcoded generic message.
        body.setdefault(
            "error",
            body.get("error") or body.get("detail") or body.get("message") or json.dumps(exc.detail),
        )
        body.setdefault("detail", exc.detail)
    else:
        body = {"detail": exc.detail, "error": exc.detail}

    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

# All routers are mounted at root (no /api prefix) to match the Flask
# backend's URL layout, so the existing Next.js frontend could point at
# this backend later without changes.
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(ai_tutor.router)
app.include_router(mock_tests.router)
app.include_router(myskillguru_auth.router)
app.include_router(roadmap.router)
app.include_router(self_learner_analytics.router)
app.include_router(self_learner_course_material.router)


@app.get("/")
async def index():
    return "MySkillGuru API is running"


@app.get("/health")
async def health():
    try:
        db = get_database()
        await db.command("ping")
        database_status = "connected"
    except Exception:
        database_status = "disconnected"

    return {
        "status": "healthy",
        "database": database_status,
        "imagekit": "configured" if settings.IMAGEKIT_PUBLIC_KEY and settings.IMAGEKIT_PRIVATE_KEY else "not configured",
        "ai_models": {"gemini": "gemini-2.5-flash", "claude": "claude-sonnet-4-6"},
    }
