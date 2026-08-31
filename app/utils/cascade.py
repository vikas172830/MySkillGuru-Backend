import logging

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


async def cascade_institute_status(db: AsyncIOMotorDatabase, institute_user_id, is_active: bool) -> None:
    """
    When an institute account is activated/deactivated, cascade the same
    is_active value to all faculty and institute_students under it.
    """
    institute = await db["instituteDetails"].find_one({"user_id": ObjectId(institute_user_id)})
    if not institute:
        logging.warning("No institute found for cascade (user_id=%s)", institute_user_id)
        return

    institute_id = institute["_id"]

    faculty_user_ids = [
        f["user_id"] async for f in db["facultyDetails"].find({"institute_id": institute_id})
    ]
    student_user_ids = [
        s["user_id"]
        async for s in db["studentDetails"].find({"institute_id": institute_id, "role": 4})
    ]

    all_ids = faculty_user_ids + student_user_ids
    if all_ids:
        result = await db["users"].update_many(
            {"_id": {"$in": [ObjectId(uid) for uid in all_ids]}},
            {"$set": {"is_active": is_active}},
        )
        action = "activated" if is_active else "deactivated"
        logging.info(
            "Cascade %s %d users (faculty + institute_students) under institute %s",
            action, result.modified_count, institute_id,
        )


async def cascade_tutor_status(db: AsyncIOMotorDatabase, tutor_user_id, is_active: bool) -> None:
    """When a tutor account is activated/deactivated, cascade to all their tutor_students."""
    student_user_ids = [
        s["user_id"]
        async for s in db["studentDetails"].find({"tutor_id": ObjectId(tutor_user_id), "role": 6})
    ]

    if student_user_ids:
        result = await db["users"].update_many(
            {"_id": {"$in": [ObjectId(uid) for uid in student_user_ids]}},
            {"$set": {"is_active": is_active}},
        )
        action = "activated" if is_active else "deactivated"
        logging.info(
            "Cascade %s %d tutor_students under tutor %s",
            action, result.modified_count, tutor_user_id,
        )


async def cascade_institute_access(
    db: AsyncIOMotorDatabase,
    institute_id,
    user_object_id,
    co_access: bool | None = None,
    qpg_access: bool | None = None,
    color: str | None = None,
) -> None:
    """Propagates hasCOAccess / hasQPGAccess / color from institute -> faculty users."""
    faculty_user_ids = [
        f["user_id"] async for f in db["facultyDetails"].find({"institute_id": institute_id})
    ]

    update_payload = {}
    if co_access is not None:
        update_payload["hasCOAccess"] = co_access
    if qpg_access is not None:
        update_payload["hasQPGAccess"] = qpg_access
    if color is not None:
        update_payload["color"] = color

    if not update_payload:
        return

    await db["users"].update_one({"_id": user_object_id}, {"$set": update_payload})

    if faculty_user_ids:
        await db["users"].update_many(
            {"_id": {"$in": faculty_user_ids}}, {"$set": update_payload}
        )
