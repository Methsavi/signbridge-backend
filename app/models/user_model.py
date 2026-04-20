from pydantic import BaseModel, EmailStr

class User(BaseModel):
    username: str
    email: EmailStr
    password: str

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