from pydantic import BaseModel, EmailStr
from typing import Optional

class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    appwrite_id: Optional[str] = None  # Appwrite user.$id, stored for cross-referencing

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AdminUserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "User"
    status: str = "Active"
    is_admin: bool = False


class AdminUserUpdate(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    password: str | None = None
    role: str | None = None
    status: str | None = None
    is_admin: bool | None = None