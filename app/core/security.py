from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Response

from app.core.config import settings


# ============================================================
# PASSWORD HASHING (mirrors Flask-Bcrypt)
# ============================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, AttributeError):
        return False


# ============================================================
# JWT (mirrors Flask-JWT-Extended: identity + additional claims)
# ============================================================

def create_access_token(identity: str, additional_claims: dict | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": identity,
        "iat": now,
        "exp": now + timedelta(days=settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS),
        **(additional_claims or {}),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])


# ============================================================
# COOKIE HELPERS (mirrors set_access_cookies / unset_jwt_cookies)
# ============================================================

def set_access_cookie(response: Response, token: str) -> None:
    # samesite="strict" (not Flask's CSRF-protection-disabled original, and
    # stricter than a plain "lax"): safe with zero frontend changes. Most
    # frontend calls go through next.config.mjs's rewrites() proxy, so
    # they're same-origin. A handful of pages (Pomodoro, AI Tutor) call the
    # backend directly via a hardcoded host — but SameSite is scoped to the
    # registrable domain/IP, not the port, so same-host-different-port calls
    # are still "same-site" and unaffected by lax->strict.
    response.set_cookie(
        key=settings.JWT_COOKIE_NAME,
        value=token,
        max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
        httponly=True,
        samesite="strict",
        secure=settings.is_production,
    )


def unset_access_cookie(response: Response) -> None:
    response.delete_cookie(key=settings.JWT_COOKIE_NAME, path="/")
