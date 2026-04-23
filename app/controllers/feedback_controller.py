from datetime import datetime, timezone
from bson import ObjectId
from pymongo import DESCENDING

from app.core.database import get_database
from app.models.feedback_model import FeedbackCreate, FeedbackUpdate


def _serialize_feedback(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "user_id": doc.get("user_id", ""),
        "username": doc.get("username", "Unknown"),
        "email": doc.get("email", ""),
        "rating": doc.get("rating", 0),
        "message": doc.get("message"),
        "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
        "updated_at": doc["updated_at"].isoformat() if doc.get("updated_at") else None,
    }


# ------------------------------------------------------------------
# CREATE
# ------------------------------------------------------------------
def create_feedback(payload: FeedbackCreate) -> dict:
    if not (1 <= payload.rating <= 5):
        return {"error": "Rating must be between 1 and 5"}

    db = get_database()
    col = db["feedbacks"]

    now = datetime.now(timezone.utc)
    doc = {
        "user_id": payload.user_id,
        "username": payload.username,
        "email": payload.email,
        "rating": payload.rating,
        "message": payload.message,
        "created_at": now,
        "updated_at": now,
    }

    result = col.insert_one(doc)
    created = col.find_one({"_id": result.inserted_id})
    return _serialize_feedback(created)


# ------------------------------------------------------------------
# READ – all (admin)
# ------------------------------------------------------------------
def list_feedbacks(rating: int | None = None, search: str | None = None) -> list:
    db = get_database()
    col = db["feedbacks"]

    query: dict = {}
    if rating:
        query["rating"] = rating
    if search:
        query["$or"] = [
            {"username": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
            {"message": {"$regex": search, "$options": "i"}},
        ]

    docs = list(col.find(query).sort("created_at", DESCENDING))
    return [_serialize_feedback(d) for d in docs]


# ------------------------------------------------------------------
# READ – by user_id (for the user's own feedback history)
# ------------------------------------------------------------------
def get_feedbacks_by_user(user_id: str) -> list:
    db = get_database()
    col = db["feedbacks"]
    docs = list(col.find({"user_id": user_id}).sort("created_at", DESCENDING))
    return [_serialize_feedback(d) for d in docs]


# ------------------------------------------------------------------
# READ – single
# ------------------------------------------------------------------
def get_feedback_by_id(feedback_id: str) -> dict | None:
    if not ObjectId.is_valid(feedback_id):
        return None
    db = get_database()
    doc = db["feedbacks"].find_one({"_id": ObjectId(feedback_id)})
    return _serialize_feedback(doc) if doc else None


# ------------------------------------------------------------------
# UPDATE
# ------------------------------------------------------------------
def update_feedback(feedback_id: str, payload: FeedbackUpdate) -> dict:
    if not ObjectId.is_valid(feedback_id):
        return {"error": "Invalid feedback id"}

    db = get_database()
    col = db["feedbacks"]

    existing = col.find_one({"_id": ObjectId(feedback_id)})
    if not existing:
        return {"error": "Feedback not found"}

    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if payload.rating is not None:
        if not (1 <= payload.rating <= 5):
            return {"error": "Rating must be between 1 and 5"}
        updates["rating"] = payload.rating
    if payload.message is not None:
        updates["message"] = payload.message

    col.update_one({"_id": ObjectId(feedback_id)}, {"$set": updates})
    updated = col.find_one({"_id": ObjectId(feedback_id)})
    return _serialize_feedback(updated)


# ------------------------------------------------------------------
# DELETE
# ------------------------------------------------------------------
def delete_feedback(feedback_id: str) -> dict:
    if not ObjectId.is_valid(feedback_id):
        return {"error": "Invalid feedback id"}

    db = get_database()
    result = db["feedbacks"].delete_one({"_id": ObjectId(feedback_id)})
    if result.deleted_count == 0:
        return {"error": "Feedback not found"}
    return {"message": "Feedback deleted successfully"}


# ------------------------------------------------------------------
# STATS – average rating + count (for admin dashboard)
# ------------------------------------------------------------------
def get_feedback_stats() -> dict:
    db = get_database()
    col = db["feedbacks"]

    total = col.count_documents({})
    pipeline = [{"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}]
    agg = list(col.aggregate(pipeline))
    avg_rating = round(agg[0]["avg_rating"], 2) if agg else 0.0

    # Distribution: count per star
    dist_pipeline = [{"$group": {"_id": "$rating", "count": {"$sum": 1}}}]
    dist_docs = list(col.aggregate(dist_pipeline))
    distribution = {str(i): 0 for i in range(1, 6)}
    for d in dist_docs:
        distribution[str(d["_id"])] = d["count"]

    return {
        "total_feedbacks": total,
        "average_rating": avg_rating,
        "distribution": distribution,
    }
