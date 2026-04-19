from datetime import datetime, timezone
from app.core.database import get_database
from app.models.user_model import User, UserLogin
from passlib.context import CryptContext
from bson import ObjectId  # <--- CRITICAL IMPORT

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

    return {
        "msg": "Login successful",
        "user_id": str(user_doc["_id"]),
        "username": user_doc["username"],
        "email": user_doc["email"],
        "profile_picture": user_doc.get("profile_picture")  # Send image if exists
    }


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

