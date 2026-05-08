from fastapi import APIRouter, status, Body, HTTPException, UploadFile, File, Query
from app.controllers.user_controller import (
    create_user_mongo,
    login_user_mongo,
    update_profile_picture,
    get_admin_dashboard_stats,
    get_analytics_stats,
    list_users_admin,
    create_user_admin,
    update_user_admin,
    delete_user_admin,
    list_admins,
)
from app.models.user_model import User, UserLogin, AdminUserCreate, AdminUserUpdate
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
async def upload_profile_pic(
    user_id: str,
    file: UploadFile = File(...),
    email: str | None = Query(default=None),
):
    try:
        # 1. Read file
        file_content = await file.read()

        # 2. Check size (5MB limit)
        if len(file_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (Max 5MB)")

        # 3. Process
        result = update_profile_picture(
            user_id=user_id,
            file_data=file_content,
            content_type=file.content_type,
            file_name=file.filename,
            user_email=email,
        )

        if "error" in result:
            status_code = 404 if result["error"] == "User not found" else 400
            raise HTTPException(status_code=status_code, detail=result["error"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN DASHBOARD STATS ---
@router.get("/admin/dashboard-stats", status_code=status.HTTP_200_OK)
def get_dashboard_stats():
    try:
        return get_admin_dashboard_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN ANALYTICS ---
@router.get("/admin/analytics", status_code=status.HTTP_200_OK)
def get_analytics():
    try:
        return get_analytics_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN USERS CRUD ---
@router.get("/admin/users", status_code=status.HTTP_200_OK)
def get_admin_users(
    search: str | None = Query(default=None),
    role: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
):
    try:
        users = list_users_admin(search=search, role=role, status=status_filter)
        return {"items": users, "count": len(users)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/users", status_code=status.HTTP_201_CREATED)
def create_admin_user(payload: AdminUserCreate = Body(...)):
    try:
        created = create_user_admin(payload)
        if created.get("error") == "Email already registered":
            raise HTTPException(status_code=400, detail="Email already registered")
        return created
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/admin/users/{user_id}", status_code=status.HTTP_200_OK)
def update_admin_user(user_id: str, payload: AdminUserUpdate = Body(...)):
    try:
        updated = update_user_admin(user_id, payload)
        if "error" in updated:
            if updated["error"] in ["User not found", "Invalid user id"]:
                raise HTTPException(status_code=404, detail=updated["error"])
            raise HTTPException(status_code=400, detail=updated["error"])
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/admin/users/{user_id}", status_code=status.HTTP_200_OK)
def delete_admin_user(user_id: str):
    try:
        result = delete_user_admin(user_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ADMIN MANAGEMENT CRUD (admins only subset) ---
@router.get("/admin/admins", status_code=status.HTTP_200_OK)
def get_admin_accounts(search: str | None = Query(default=None)):
    try:
        admins = list_admins(search=search)
        return {"items": admins, "count": len(admins)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/admins", status_code=status.HTTP_201_CREATED)
def create_admin_account(payload: AdminUserCreate = Body(...)):
    try:
        payload.is_admin = True
        if not payload.role or payload.role.strip().lower() == "user":
            payload.role = "Admin"

        created = create_user_admin(payload)
        if created.get("error") == "Email already registered":
            raise HTTPException(status_code=400, detail="Email already registered")
        return created
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/admin/admins/{user_id}", status_code=status.HTTP_200_OK)
def update_admin_account(user_id: str, payload: AdminUserUpdate = Body(...)):
    try:
        payload.is_admin = True if payload.is_admin is None else payload.is_admin
        updated = update_user_admin(user_id, payload)
        if "error" in updated:
            if updated["error"] in ["User not found", "Invalid user id"]:
                raise HTTPException(status_code=404, detail=updated["error"])
            raise HTTPException(status_code=400, detail=updated["error"])
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/admin/admins/{user_id}", status_code=status.HTTP_200_OK)
def delete_admin_account(user_id: str):
    try:
        result = delete_user_admin(user_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- GET USER PROFILE ---
# Keep this dynamic route after fixed /admin/* routes to avoid path shadowing.
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
    #     raise HTTPException(status_code=500, detail=str(e))e")
    #         }
    #     raise HTTPException(status_code=404, detail="User not found")
    # except HTTPException:
    #     raise
    # except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))