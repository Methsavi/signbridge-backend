from datetime import datetime, timezone
from app.core.database import get_database
from app.models.user_model import User, UserLogin
from passlib.context import CryptContext
import base64  # <--- CRITICAL IMPORT
from bson import ObjectId  # <--- CRITICAL IMPORT

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
def update_profile_picture(user_id: str, file_data: bytes, content_type: str):
    print(f"🖼️ Processing image for User ID: {user_id}")  # Debug print

    try:
        db = get_database()
        users_collection = db["users"]

        # 1. Convert bytes to Base64 String
        base64_str = base64.b64encode(file_data).decode('utf-8')
        final_string = f"data:{content_type};base64,{base64_str}"

        # 2. Update Database
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"profile_picture": final_string}}
        )

        if result.matched_count == 0:
            print("❌ User not found in DB")
            return {"error": "User not found"}

        print("✅ Image saved to MongoDB")
        return {"msg": "Profile picture updated", "url": final_string}

    except Exception as e:
        print(f"❌ Error in update_profile_picture: {e}")
        # Return the specific error so we can see it in the frontend
        return {"error": str(e)}