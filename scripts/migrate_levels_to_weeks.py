"""
Migrates legacy selfLearnerRoadmaps documents from the old
levels[] -> topics[] -> subtopics[] schema into the flat weeks[] schema
every part of this backend now assumes (roadmap.py, mock_tests.py's
roadmap-mode Test Engine, self_learner_analytics.py). Mirrors the reference
Flask project's own scripts/migrate_levels_to_weeks.py — both projects
share the same MongoDB database, and these documents were created/used
through that project (the "<DominantStyle>-<Difficulty>" notes cache keys
match its Phase 2 VARK work exactly) before its own Phase 1 migration ran,
or independently of it.

Safety: only ADDS weeks/unlockedWeeks/schemaVersion — the old levels/
unlockedLevels fields are deliberately left in place as an inert backup
rather than removed. Nothing in this codebase reads them once weeks[]
exists, so keeping them costs nothing and gives a free rollback path.

Every document in the wild data this was checked against has exactly one
topic per level, so level -> week is a direct 1:1 mapping and existing
progress.completedSubtopics keys ("<levelNum>-<subtopicIdx>-<title>") stay
valid unchanged (subtopic order/index is preserved by concatenating each
level's topics' subtopics in encounter order — a no-op when there's only
one topic, which is every case observed).

Dry run by default — prints exactly what would change, writes nothing.
Pass --apply to actually write.

Usage:
    python scripts/migrate_levels_to_weeks.py            # dry run
    python scripts/migrate_levels_to_weeks.py --apply     # writes
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.core.config import settings  # noqa: E402


def _build_week(level: dict) -> dict:
    subtopics = []
    for topic in level.get("topics", []):
        subtopics.extend(topic.get("subtopics", []))

    intro = level.get("introDescription") or level.get("description") or ""
    if not intro and level.get("topics"):
        intro = level["topics"][0].get("description", "")

    return {
        "week": level.get("level"),
        "title": level.get("title", ""),
        "introDescription": intro,
        "subtopics": subtopics,
        "practiceQuestions": level.get("practiceQuestions", []),
    }


async def main(apply: bool) -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.DB_NAME]

    cursor = db["selfLearnerRoadmaps"].find({"weeks": {"$exists": False}})
    count = 0
    async for doc in cursor:
        count += 1
        levels = doc.get("levels", [])
        weeks = [_build_week(lvl) for lvl in levels]
        unlocked_weeks = doc.get("unlockedLevels", [1])

        print(f"[{doc['_id']}] subject={doc.get('subject')!r} active={doc.get('active')} "
              f"-> {len(weeks)} weeks, unlockedWeeks={unlocked_weeks}")
        for wk in weeks:
            print(f"    week {wk['week']}: {wk['title']!r} ({len(wk['subtopics'])} subtopics)")

        if apply:
            result = await db["selfLearnerRoadmaps"].update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "weeks": weeks,
                    "unlockedWeeks": unlocked_weeks,
                    "schemaVersion": 2,
                }},
            )
            print(f"    -> written (matched={result.matched_count}, modified={result.modified_count})")

    if count == 0:
        print("No legacy roadmaps found (nothing to migrate).")
    else:
        print(f"\n{count} legacy roadmap(s) {'migrated' if apply else 'found — re-run with --apply to write'}.")

    client.close()


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
