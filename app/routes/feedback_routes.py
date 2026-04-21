from fastapi import APIRouter, status, Body, HTTPException, Query

from app.controllers.feedback_controller import (
    create_feedback,
    list_feedbacks,
    get_feedbacks_by_user,
    get_feedback_by_id,
    update_feedback,
    delete_feedback,
    get_feedback_stats,
)
from app.models.feedback_model import FeedbackCreate, FeedbackUpdate

router = APIRouter(prefix="/feedbacks", tags=["feedbacks"])


# ── CREATE ──────────────────────────────────────────────────────────
@router.post("/", status_code=status.HTTP_201_CREATED)
def submit_feedback(payload: FeedbackCreate = Body(...)):
    result = create_feedback(payload)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── LIST ALL (admin) ────────────────────────────────────────────────
@router.get("/admin/all", status_code=status.HTTP_200_OK)
def admin_list_feedbacks(
    rating: int | None = Query(default=None),
    search: str | None = Query(default=None),
):
    items = list_feedbacks(rating=rating, search=search)
    return {"items": items, "count": len(items)}


# ── ADMIN STATS ──────────────────────────────────────────────────────
@router.get("/admin/stats", status_code=status.HTTP_200_OK)
def admin_feedback_stats():
    return get_feedback_stats()


# ── GET BY USER ─────────────────────────────────────────────────────
@router.get("/user/{user_id}", status_code=status.HTTP_200_OK)
def user_feedbacks(user_id: str):
    return get_feedbacks_by_user(user_id)


# ── GET SINGLE ──────────────────────────────────────────────────────
@router.get("/{feedback_id}", status_code=status.HTTP_200_OK)
def get_single_feedback(feedback_id: str):
    item = get_feedback_by_id(feedback_id)
    if not item:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return item


# ── UPDATE ──────────────────────────────────────────────────────────
@router.put("/{feedback_id}", status_code=status.HTTP_200_OK)
def edit_feedback(feedback_id: str, payload: FeedbackUpdate = Body(...)):
    result = update_feedback(feedback_id, payload)
    if "error" in result:
        code = 404 if "not found" in result["error"].lower() else 400
        raise HTTPException(status_code=code, detail=result["error"])
    return result


# ── DELETE ──────────────────────────────────────────────────────────
@router.delete("/{feedback_id}", status_code=status.HTTP_200_OK)
def remove_feedback(feedback_id: str):
    result = delete_feedback(feedback_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
