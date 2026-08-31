import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, Request, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import settings
from app.core.security import decode_access_token
from app.db.mongodb import get_database
from app.models.user import SELF_LEARNER

# ============================================================
# IDENTITY (equivalent to @jwt_required() + get_jwt_identity()/get_jwt())
# ============================================================

def get_current_identity(request: Request) -> dict:
    token = request.cookies.get(settings.JWT_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return {"user_id": user_id, "role": payload.get("role")}


def require_role(*allowed_roles: int):
    """Dependency factory — equivalent to Flask's inline `if role not in [...]: 403`."""

    def _dependency(identity: dict = Depends(get_current_identity)) -> dict:
        if identity.get("role") not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")
        return identity

    return _dependency


# ============================================================
# FULL USER DOCUMENT (equivalent to /me's user_collection.find_one)
# ============================================================

async def get_current_user(
    identity: dict = Depends(get_current_identity),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> dict:
    if not ObjectId.is_valid(identity["user_id"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user id")

    user = await db["users"].find_one({"_id": ObjectId(identity["user_id"]), "is_deleted": False})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


# ============================================================
# MYSKILLGURU ACCESS GATE
# MySkillGuru has no institute/faculty concept — every account it creates
# is a SELF_LEARNER (role 7), so the gate is a plain role check. (The
# original LMS variant also gated INSTITUTE_STUDENT callers behind an
# institute/school opt-in flag; that branch doesn't apply here since
# institute accounts can't exist in this scoped-down copy.)
# ============================================================

async def can_use_myskillguru(identity: dict) -> bool:
    return identity.get("role") == SELF_LEARNER


async def require_myskillguru_access(identity: dict = Depends(get_current_identity)) -> dict:
    if not await can_use_myskillguru(identity):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="MySkillGuru access required")
    return identity
