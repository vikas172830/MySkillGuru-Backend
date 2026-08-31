# ============================================================
# MYSKILLGURU PUBLIC REGISTRATION
#
# A separate, unauthenticated entry point for individual students signing
# up directly at the MySkillGuru landing page — distinct from /register
# (auth.py), whose router-independent `Depends(get_current_identity)` on
# every _register_* handler means it requires an already-authenticated
# caller for every role, including self_learner. That's correct for
# self-learners created *by* an already-logged-in caller, but wrong for a
# stranger landing on a public signup page with no session at all.
#
# Delegates to the exact same _register_self_learner() an authenticated
# /register call would use, so both entry points produce identical
# accounts: role 7, is_active=False, pending superadmin approval — see
# CLAUDE.md/the MySkillGuru rollout plan for why approval-gating (not
# auto-activation + email verification) was chosen: no email/SMTP
# infrastructure exists anywhere in this codebase yet, and reusing the
# already-working approval flow avoids standing up net-new infra for this
# feature alone.
# ============================================================
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.routers.auth import _register_self_learner
from app.core.rate_limit import myskillguru_register_rate_limit
from app.core.security import hash_password
from app.db.mongodb import get_database
from app.schemas.auth import RegisterSelfLearner

router = APIRouter(prefix="/myskillguru", tags=["myskillguru-auth"])


@router.post("/register", dependencies=[Depends(myskillguru_register_rate_limit)])
async def myskillguru_register(
    payload: RegisterSelfLearner,
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    email = payload.email.lower()
    if await db["users"].find_one({"email": email}):
        raise HTTPException(status_code=400, detail="A user with this email already exists")

    data = payload.model_dump(exclude_unset=True)
    password_hash = hash_password(payload.password)

    body, code = await _register_self_learner(db, data, password_hash)
    if code >= 400:
        raise HTTPException(status_code=code, detail=body.get("error"))
    return body
