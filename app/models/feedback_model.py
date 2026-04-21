from pydantic import BaseModel
from typing import Optional


class FeedbackCreate(BaseModel):
    user_id: str
    username: str
    email: str
    rating: int          # 1-5
    message: Optional[str] = None


class FeedbackUpdate(BaseModel):
    rating: Optional[int] = None
    message: Optional[str] = None
