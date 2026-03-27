from fastapi import APIRouter, status, Body, HTTPException, UploadFile, File
from app.controllers.user_controller import create_user_mongo, login_user_mongo, update_profile_picture
from app.models.user_model import User, UserLogin
from bson import ObjectId
from app.core.database import get_database

router = APIRouter(prefix="/users", tags=["users"])


# --- REGISTER ---
@router.post("/register", status_code=status.HTTP_201_CREATED)
def create_user(user: User = Body(...)):
    try:
        response = create_user_mongo(user)
        if response.get("msg") == "Email already registered":
            raise HTTPException(status_code=400, detail="Email already registered")

        return {"message": "User created successfully", "user": response}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- LOGIN ---
@router.post("/login", status_code=status.HTTP_200_OK)
def login(credentials: UserLogin = Body(...)):
    try:
        login_response = login_user_mongo(credentials)

        if login_response.get("msg") == "Login successful":
            return {
                "message": "Login successful",
                "user_id": login_response.get("user_id"),
                "username": login_response.get("username"),
                "email": login_response.get("email"),
                "profile_picture": login_response.get("profile_picture")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=login_response.get("msg", "Invalid credentials")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- UPLOAD PROFILE PICTURE ---
@router.patch("/{user_id}/profile-picture")
async def upload_profile_pic(user_id: str, file: UploadFile = File(...)):
    try:
        # 1. Read file
        file_content = await file.read()

        # 2. Check size (5MB limit)
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (Max 5MB)")

        # 3. Process
        result = update_profile_picture(user_id, file_content, file.content_type)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- GET USER PROFILE (NEW) ---
# This allows the frontend to fetch fresh data on reload
@router.get("/{user_id}")
def get_user_profile(user_id: str):
    try:
        db = get_database()
        user = db["users"].find_one({"_id": ObjectId(user_id)})

        if user:
            return {
                "user_id": str(user["_id"]),
                "username": user["username"],
                "email": user["email"],
                "profile_picture": user.get("profile_picture")
            }
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))