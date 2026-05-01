from datetime import datetime, timezone, timedelta
from collections import defaultdict
from app.core.database import get_database
from app.models.user_model import User, UserLogin, AdminUserCreate, AdminUserUpdate
from passlib.context import CryptContext
from bson import ObjectId  # <--- CRITICAL IMPORT
from pymongo import DESCENDING

from app.services.r2_storage import upload_profile_image

# Password Hashing Setup
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_user_mongo(user: User):
    db = get_database()
    users_collection = db["users"]

    # Check for existing email
    if users_collection.find_one({"email": user.email}):
        return {"msg": "Email already registered"}

    # If appwrite_id provided, also check it's not already linked
    if user.appwrite_id and users_collection.find_one({"appwrite_id": user.appwrite_id}):
        return {"msg": "Email already registered"}

    hashed_pw = hash_password(user.password)

    user_doc = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_pw,
        "appwrite_id": user.appwrite_id,  # Store Appwrite user.$id (may be None for legacy)
        "created_at": datetime.now(timezone.utc),
        "last_active": datetime.now(timezone.utc),
        "role": "User",
        "status": "Active",
        "is_admin": False,
        "profile_picture": None  # Initialize as None
    }

    result = users_collection.insert_one(user_doc)

    return {
        "msg": "User created successfully",
        "user_id": str(result.inserted_id)
    }


def login_user_mongo(credentials: UserLogin):
    db = get_database()
    users_collection = db["users"]

    user_doc = users_collection.find_one({"email": credentials.email})
    if not user_doc:
        return {"msg": "User not found"}

    if not pwd_context.verify(credentials.password, user_doc["hashed_password"]):
        return {"msg": "Incorrect password"}

    users_collection.update_one(
        {"_id": user_doc["_id"]},
        {"$set": {"last_active": datetime.now(timezone.utc)}}
    )

    return {
        "msg": "Login successful",
        "user_id": str(user_doc["_id"]),
        "username": user_doc["username"],
        "email": user_doc["email"],
        "profile_picture": user_doc.get("profile_picture")  # Send image if exists
    }


def _to_iso(dt_value):
    if not dt_value:
        return None
    try:
        return dt_value.isoformat()
    except Exception:
        return None


def _serialize_user(user_doc):
    username = (
        user_doc.get("username")
        or user_doc.get("full_name")
        or user_doc.get("name")
        or "Unknown"
    )
    created_at = user_doc.get("created_at")
    if not created_at and user_doc.get("_id"):
        try:
            created_at = user_doc["_id"].generation_time
        except Exception:
            created_at = None

    normalized_role = _normalize_role(user_doc.get("role"), bool(user_doc.get("is_admin", False)))

    return {
        "id": str(user_doc["_id"]),
        "username": username,
        "email": user_doc.get("email", ""),
        "role": normalized_role,
        "status": user_doc.get("status", "Active"),
        "is_admin": normalized_role == "Admin",
        "profile_picture": user_doc.get("profile_picture"),
        "created_at": _to_iso(created_at),
        "last_active": _to_iso(user_doc.get("last_active")),
    }


def _is_admin_role(role: str | None) -> bool:
    return (role or "").strip().lower() in {"admin", "superadmin", "moderator", "developer"}


def _normalize_role(role: str | None, is_admin: bool | None = None) -> str:
    if is_admin is True:
        return "Admin"
    if _is_admin_role(role):
        return "Admin"
    return "User"


def get_admin_dashboard_stats():
    db = get_database()
    users_collection = db["users"]

    total_users = users_collection.count_documents({})
    active_users = users_collection.count_documents(
        {
            "$or": [
                {"status": "Active"},
                {"status": {"$exists": False}},
            ]
        }
    )
    total_admins = users_collection.count_documents({
        "$or": [
            {"is_admin": True},
            {"role": {"$in": ["Admin", "Superadmin", "Moderator", "Developer"]}},
        ]
    })
    since = datetime.now(timezone.utc) - timedelta(days=30)
    new_users_30d = users_collection.count_documents({"created_at": {"$gte": since}})

    recent = list(
        users_collection.find(
            {},
            {
                "username": 1,
                "full_name": 1,
                "name": 1,
                "email": 1,
                "created_at": 1,
                "profile_picture": 1,
            },
        )
        .sort([("created_at", DESCENDING), ("_id", DESCENDING)])
        .limit(5)
    )

    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_admins": total_admins,
        "new_users_30d": new_users_30d,
        "recent_users": [
            {
                "id": str(u["_id"]),
                "username": (u.get("username") or u.get("full_name") or u.get("name") or "Unknown"),
                "email": u.get("email", ""),
                "profile_picture": u.get("profile_picture"),
                "created_at": _to_iso(u.get("created_at")),
            }
            for u in recent
        ],
    }


def list_users_admin(search: str | None = None, role: str | None = None, status: str | None = None):
    db = get_database()
    users_collection = db["users"]

    query = {}
    if search:
        query["$or"] = [
            {"username": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}},
        ]
    if role:
        query["role"] = role
    if status:
        query["status"] = status

    users = list(users_collection.find(query).sort("created_at", DESCENDING))
    return [_serialize_user(user) for user in users]


def create_user_admin(payload: AdminUserCreate):
    db = get_database()
    users_collection = db["users"]

    if users_collection.find_one({"email": payload.email}):
        return {"error": "Email already registered"}

    role = _normalize_role(payload.role, payload.is_admin)
    is_admin = role == "Admin"

    doc = {
        "username": payload.username,
        "email": payload.email,
        "hashed_password": hash_password(payload.password),
        "role": role,
        "status": payload.status or "Active",
        "is_admin": is_admin,
        "profile_picture": None,
        "created_at": datetime.now(timezone.utc),
        "last_active": datetime.now(timezone.utc),
    }

    result = users_collection.insert_one(doc)
    created = users_collection.find_one({"_id": result.inserted_id})
    return _serialize_user(created)


def update_user_admin(user_id: str, payload: AdminUserUpdate):
    db = get_database()
    users_collection = db["users"]

    if not ObjectId.is_valid(user_id):
        return {"error": "Invalid user id"}

    existing = users_collection.find_one({"_id": ObjectId(user_id)})
    if not existing:
        return {"error": "User not found"}

    updates = {}
    if payload.username is not None:
        updates["username"] = payload.username

    if payload.email is not None:
        duplicate = users_collection.find_one({"email": payload.email, "_id": {"$ne": ObjectId(user_id)}})
        if duplicate:
            return {"error": "Email already registered"}
        updates["email"] = payload.email

    if payload.password:
        updates["hashed_password"] = hash_password(payload.password)

    role_for_update = None
    if payload.role is not None or payload.is_admin is not None:
        role_for_update = _normalize_role(payload.role, payload.is_admin)
        updates["role"] = role_for_update

    if payload.status is not None:
        updates["status"] = payload.status

    if role_for_update is not None:
        updates["is_admin"] = role_for_update == "Admin"

    if updates:
        users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": updates})

    updated = users_collection.find_one({"_id": ObjectId(user_id)})
    return _serialize_user(updated)


def delete_user_admin(user_id: str):
    db = get_database()
    users_collection = db["users"]

    if not ObjectId.is_valid(user_id):
        return {"error": "Invalid user id"}

    result = users_collection.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        return {"error": "User not found"}

    return {"message": "User deleted successfully"}


def list_admins(search: str | None = None):
    db = get_database()
    users_collection = db["users"]

    query = {
        "$or": [
            {"is_admin": True},
            {"role": {"$in": ["Admin", "Superadmin", "Moderator", "Developer"]}},
        ]
    }

    if search:
        query["$and"] = [{
            "$or": [
                {"username": {"$regex": search, "$options": "i"}},
                {"email": {"$regex": search, "$options": "i"}},
            ]
        }]

    admins = list(users_collection.find(query).sort("created_at", DESCENDING))
    return [_serialize_user(admin) for admin in admins]


# --- THE UPLOAD FUNCTION ---
def get_analytics_stats():
    """Return rich analytics data for the admin Analytics dashboard."""
    db = get_database()
    users_col = db["users"]
    history_col = db["history"]
    dictionary_col = db["dictionary"]

    now = datetime.now(timezone.utc)

    # ── 1. Translations per day (last 30 days) ───────────────────────────────
    thirty_days_ago = now - timedelta(days=30)
    pipeline_daily = [
        {"$match": {"timestamp": {"$gte": thirty_days_ago}}},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    daily_raw = list(history_col.aggregate(pipeline_daily))
    # Fill gaps so every day in the range appears
    daily_map = {d["_id"]: d["count"] for d in daily_raw}
    translations_per_day = []
    for i in range(30):
        day = (thirty_days_ago + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        translations_per_day.append({"date": day, "count": daily_map.get(day, 0)})

    # ── 2. User registrations per month (last 12 months) ────────────────────
    # Use $addFields to coalesce created_at with the ObjectId generation time,
    # so users without a created_at field are still counted.
    pipeline_monthly = [
        {
            "$addFields": {
                "reg_date": {
                    "$ifNull": ["$created_at", {"$toDate": "$_id"}]
                }
            }
        },
        {"$match": {"reg_date": {"$gte": now - timedelta(days=365)}}},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m", "date": "$reg_date"}
                },
                "new_users": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    monthly_raw = list(users_col.aggregate(pipeline_monthly))
    monthly_map = {m["_id"]: m["new_users"] for m in monthly_raw}

    # Count ALL users registered before the 12-month window as starting baseline
    existing_users_count = users_col.count_documents({})
    for v in monthly_map.values():
        existing_users_count -= v
    running_total = max(0, existing_users_count)

    # Build a proper calendar-month sequence (no timedelta drift)
    registrations_per_month = []
    cursor_year = now.year
    cursor_month = now.month
    months_ordered = []
    for _ in range(12):
        months_ordered.append((cursor_year, cursor_month))
        cursor_month -= 1
        if cursor_month == 0:
            cursor_month = 12
            cursor_year -= 1
    months_ordered.reverse()

    for (yr, mo) in months_ordered:
        month_key = f"{yr:04d}-{mo:02d}"
        month_label = datetime(yr, mo, 1).strftime("%b %Y")
        new = monthly_map.get(month_key, 0)
        running_total += new
        registrations_per_month.append({
            "month": month_label,
            "new_users": new,
            "total_users": running_total,
        })

    # ── 3. Language usage distribution ───────────────────────────────────────
    pipeline_lang = [
        {"$group": {"_id": "$target_language", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    lang_raw = list(history_col.aggregate(pipeline_lang))
    language_distribution = [
        {"language": item["_id"] or "Unknown", "count": item["count"]}
        for item in lang_raw
    ]

    # ── 4. Top translators (users with most history entries) ─────────────────
    pipeline_top_users = [
        {"$group": {"_id": "$user_id", "translations": {"$sum": 1}}},
        {"$sort": {"translations": -1}},
        {"$limit": 8},
    ]
    top_raw = list(history_col.aggregate(pipeline_top_users))
    top_translators = []
    for item in top_raw:
        uid = item["_id"]
        user_doc = None
        if uid:
            if ObjectId.is_valid(uid):
                user_doc = users_col.find_one({"_id": ObjectId(uid)}, {"username": 1, "email": 1})
            if not user_doc:
                user_doc = users_col.find_one({"appwrite_id": uid}, {"username": 1, "email": 1})
        username = (
            user_doc.get("username") if user_doc else None
        ) or (uid[:8] + "..." if uid else "Unknown")
        top_translators.append({"username": username, "translations": item["translations"]})

    # ── 5. Dictionary entries by category ────────────────────────────────────
    pipeline_dict = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    dict_raw = list(dictionary_col.aggregate(pipeline_dict))
    dictionary_by_category = [
        {"category": item["_id"] or "Uncategorized", "count": item["count"]}
        for item in dict_raw
    ]
    total_dictionary_entries = sum(d["count"] for d in dictionary_by_category)

    # ── 6. Total translations overall ────────────────────────────────────────
    total_translations = history_col.count_documents({})

    # ── 7. Translations this month vs last month ──────────────────────────────
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_of_last_month = (start_of_month - timedelta(days=1)).replace(day=1)
    translations_this_month = history_col.count_documents({"timestamp": {"$gte": start_of_month}})
    translations_last_month = history_col.count_documents({
        "timestamp": {"$gte": start_of_last_month, "$lt": start_of_month}
    })

    return {
        "translations_per_day": translations_per_day,
        "registrations_per_month": registrations_per_month,
        "language_distribution": language_distribution,
        "top_translators": top_translators,
        "dictionary_by_category": dictionary_by_category,
        "total_translations": total_translations,
        "total_dictionary_entries": total_dictionary_entries,
        "translations_this_month": translations_this_month,
        "translations_last_month": translations_last_month,
    }


def update_profile_picture(
    user_id: str,
    file_data: bytes,
    content_type: str,
    file_name: str | None = None,
    user_email: str | None = None,
):
    print(f"🖼️ Processing image for User ID: {user_id}")

    try:
        db = get_database()
        users_collection = db["users"]

        user_doc = None
        if ObjectId.is_valid(user_id):
            user_doc = users_collection.find_one({"_id": ObjectId(user_id)})
        # Also try Appwrite ID — the frontend passes user.$id as user_id
        if not user_doc:
            user_doc = users_collection.find_one({"appwrite_id": user_id})
        if not user_doc and user_email:
            user_doc = users_collection.find_one({"email": user_email})

        if not user_doc:
            print("❌ User not found in DB")
            return {"error": "User not found"}

        storage_key = str(user_doc.get("_id") or user_id)
        image_url = upload_profile_image(
            user_identifier=storage_key,
            file_data=file_data,
            content_type=content_type,
            original_filename=file_name,
        )

        result = users_collection.update_one(
            {"_id": user_doc["_id"]},
            {
                "$set": {
                    "profile_picture": image_url,
                    "profile_picture_updated_at": datetime.now(timezone.utc),
                }
            },
        )

        if result.matched_count == 0:
            print("❌ User not found in DB")
            return {"error": "User not found"}

        print("✅ Image uploaded to R2 and URL saved to MongoDB")
        return {"msg": "Profile picture updated", "url": image_url}

    except Exception as e:
        if "cloudflare" in str(e).lower() or "s3" in str(e).lower() or "r2" in str(e).lower():
            print(f"❌ R2 upload failed: {e}")
            return {"error": "Failed to upload image to storage"}
        print(f"❌ Error in update_profile_picture: {e}")
        return {"error": str(e)}

