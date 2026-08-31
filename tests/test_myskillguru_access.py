# ============================================================
# Coverage for the MySkillGuru access gate:
#   - app.api.deps.can_use_myskillguru — SELF_LEARNER-only in this
#     scoped-down copy (no institute/school opt-in flag, since institute
#     accounts don't exist here)
#   - app.api.deps.require_myskillguru_access — the router-level
#     dependency wired onto every MySkillGuru router
# ============================================================
from bson import ObjectId

from app.api.deps import can_use_myskillguru
from tests.test_security_fixes import _seed_and_login_user

ROADMAP_LIST_PATH = "/api/self-learner/roadmap"


async def test_self_learner_always_allowed():
    assert await can_use_myskillguru({"user_id": str(ObjectId()), "role": 7}) is True


async def test_other_roles_not_allowed():
    assert await can_use_myskillguru({"user_id": str(ObjectId()), "role": 1}) is False


async def test_self_learner_reaches_gated_router(client_factory, test_db):
    learner = await _seed_and_login_user(test_db, client_factory, role=7, name="MCG Self Learner")
    resp = await learner.get(ROADMAP_LIST_PATH)
    assert resp.status_code == 200
