"""
Backfills courseMaterials.owner_user_ids, which did not exist before access
to uploaded course material was scoped to its owner.

Why it is needed: every read of courseMaterials is now owner-scoped — the
by-id lookup that gates roadmap grounding (rag/mongo_store.py's
find_document_by_id). A document carrying no owner_user_ids matches nobody,
so without this backfill every roadmap whose grounded_doc_id points at a
pre-existing document would silently lose its grounding and start
generating ungrounded notes.

Where ownership is recovered from: selfLearnerRoadmaps.grounded_doc_id.
A roadmap records both the document it was grounded in and the user it
belongs to, which is exactly the (document, owner) pair that was never
stored on the document itself. Nothing else in the database references a
courseMaterials id, so this recovers every document that any roadmap still
depends on — precisely the set whose loss would be user-visible.

What is deliberately NOT recovered: documents uploaded through
/api/self-learner/course-material that no roadmap ever used. The
uploader's identity was only ever held in a Redis job record with a
one-hour TTL, so it is genuinely unrecoverable. Those are left ownerless.
MySkillGuru has no institute-facing course-material listing endpoint (that
router doesn't exist in this scoped-down product — see
self_learner_course_material.py's header comment), so unlike the sibling
LMS product there is no listing view whose leak this backfill is closing;
it exists purely to keep existing roadmaps grounded. Re-uploading the same
file re-establishes ownership for free: dedup is by content hash, so it
costs no reprocessing and simply adds the uploader to owner_user_ids.

Safety: only ever $addToSet — never removes an owner, never deletes a
document. Idempotent, so re-running it changes nothing the first run
already did.

Dry run by default — prints exactly what would change, writes nothing.
Pass --apply to actually write.

Usage:
    python scripts/backfill_course_material_owners.py            # dry run
    python scripts/backfill_course_material_owners.py --apply     # writes
"""
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.core.config import settings  # noqa: E402


async def main(apply: bool) -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.DB_NAME]

    print(f"database: {settings.DB_NAME}   mode: {'APPLY' if apply else 'DRY RUN'}\n")

    # doc_id -> set of user_ids that should own it, per the roadmaps
    # currently grounded in it.
    wanted: dict[str, set[str]] = defaultdict(set)
    cursor = db.selfLearnerRoadmaps.find(
        {"grounded_doc_id": {"$nin": [None, ""]}},
        {"grounded_doc_id": 1, "user_id": 1},
    )
    async for roadmap in cursor:
        user_id = roadmap.get("user_id")
        if user_id:
            wanted[roadmap["grounded_doc_id"]].add(str(user_id))

    print(f"roadmaps reference {len(wanted)} distinct course-material document(s)")

    updated = 0
    already = 0
    missing = 0

    for doc_id, user_ids in sorted(wanted.items()):
        doc = await db.courseMaterials.find_one({"_id": doc_id}, {"owner_user_ids": 1, "filename": 1})
        if doc is None:
            # Roadmap outlived the document it was grounded in. Already
            # handled at runtime: find_document_by_id returns None and
            # generation falls back to ungrounded.
            print(f"  MISSING  {doc_id}  (referenced by {len(user_ids)} roadmap owner(s), no such document)")
            missing += 1
            continue

        existing = set(doc.get("owner_user_ids") or [])
        to_add = user_ids - existing
        if not to_add:
            already += 1
            continue

        print(f"  + {doc_id}  {doc.get('filename')!r}  add {len(to_add)} owner(s): {sorted(to_add)}")
        if apply:
            await db.courseMaterials.update_one(
                {"_id": doc_id}, {"$addToSet": {"owner_user_ids": {"$each": sorted(to_add)}}}
            )
        updated += 1

    total = await db.courseMaterials.count_documents({})
    ownerless = await db.courseMaterials.count_documents(
        {"$or": [{"owner_user_ids": {"$exists": False}}, {"owner_user_ids": []}]}
    )

    print(
        f"\n{'updated' if apply else 'would update'}: {updated}   "
        f"already correct: {already}   dangling references: {missing}"
    )
    print(f"courseMaterials total: {total}")
    print(
        f"still ownerless after this: {ownerless if apply else 'run with --apply to see the final count'}"
        + ("" if apply else f"  (currently {ownerless})")
    )
    if not apply:
        print("\nDRY RUN — nothing was written. Re-run with --apply to commit.")

    client.close()


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
