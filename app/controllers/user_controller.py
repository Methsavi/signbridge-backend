from datetime import datetime, timezone, timedelta
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

    if users_collection.find_one({"email": user.email}):
        return {"msg": "Email already registered"}

    hashed_pw = hash_password(user.password)

    user_doc = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_pw,
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

