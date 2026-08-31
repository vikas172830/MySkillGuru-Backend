"""
Follow-up to migrate_levels_to_weeks.py. That script correctly builds
weeks[] from the old levels[] -> topics[] -> subtopics[] structure, but
progress.completedSubtopics keys created by the reference Flask project's
OLD (pre-its-own-Phase-1) code used "<level>-<topicIdx>-<title>" — and
since every level had exactly one topic, topicIdx was always 0, no matter
which subtopic it actually was. This backend's key convention is
"<week>-<subtopicIdxWithinWeek>-<title>", where the index varies per
subtopic — so every completedSubtopics key except each week's first
subtopic silently stopped matching after migration (real progress the
student had already made would show as incomplete).

Fix: for each completedSubtopics key, look up the subtopic by title within
its week's now-correct subtopics[] list and rewrite the key with that
subtopic's real index. Idempotent — a key that's already correct maps to
itself unchanged, so this is safe to re-run or run against roadmaps that
were never on the old schema at all.

Dry run by default; pass --apply to write. Also recomputes
progress.overallProgress after the fix, since it depends on
completedSubtopics's length matching real distinct subtopics.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.core.config import settings  # noqa: E402


def _fix_keys(doc: dict) -> tuple[list, bool]:
    weeks_by_num = {w["week"]: w for w in doc.get("weeks", [])}
    old_keys = doc.get("progress", {}).get("completedSubtopics", [])
    new_keys = []
    changed = False

    for key in old_keys:
        parts = key.split("-", 2)
        if len(parts) != 3:
            new_keys.append(key)  # unrecognized shape, leave as-is
            continue
        week_str, idx_str, title = parts
        try:
            week_num = int(week_str)
        except ValueError:
            new_keys.append(key)
            continue

        week = weeks_by_num.get(week_num)
        if not week:
            new_keys.append(key)  # week no longer exists, leave as-is
            continue

        real_idx = next((i for i, s in enumerate(week.get("subtopics", [])) if s.get("title") == title), None)
        if real_idx is None:
            new_keys.append(key)  # title not found, leave as-is rather than guess
            continue

        correct_key = f"{week_num}-{real_idx}-{title}"
        new_keys.append(correct_key)
        if correct_key != key:
            changed = True

    # de-dupe while preserving order, in case the fix collapses any accidental repeats
    seen = set()
    deduped = []
    for k in new_keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    return deduped, changed


def _recalculate_progress(doc: dict, completed_subtopics: list) -> int:
    weeks = doc.get("weeks", [])
    passed_quizzes = doc.get("progress", {}).get("passedQuizzes", {})
    total_sub = sum(len(wk.get("subtopics", [])) for wk in weeks)
    total_actions = total_sub + len(weeks)
    completed_actions = len(completed_subtopics) + len(passed_quizzes)
    return min(100, round((completed_actions / total_actions * 100))) if total_actions > 0 else 0


async def main(apply: bool) -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=8000)
    db = client[settings.DB_NAME]

    cursor = db["selfLearnerRoadmaps"].find({"weeks": {"$exists": True}})
    fixed_count = 0
    async for doc in cursor:
        new_keys, changed = _fix_keys(doc)
        if not changed:
            continue

        fixed_count += 1
        old_keys = doc.get("progress", {}).get("completedSubtopics", [])
        new_progress = round(_recalculate_progress(doc, new_keys))
        old_progress = doc.get("progress", {}).get("overallProgress")

        print(f"[{doc['_id']}] subject={doc.get('subject')!r}")
        print(f"    before: {old_keys}")
        print(f"    after:  {new_keys}")
        print(f"    overallProgress: {old_progress} -> {new_progress}")

        if apply:
            result = await db["selfLearnerRoadmaps"].update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "progress.completedSubtopics": new_keys,
                    "progress.overallProgress": new_progress,
                }},
            )
            print(f"    -> written (matched={result.matched_count}, modified={result.modified_count})")

    if fixed_count == 0:
        print("No roadmaps needed a completedSubtopics key fix.")
    else:
        print(f"\n{fixed_count} roadmap(s) {'fixed' if apply else 'need fixing — re-run with --apply to write'}.")

    client.close()


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
