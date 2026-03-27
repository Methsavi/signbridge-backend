from fastapi import APIRouter, Body, HTTPException
from app.controllers.feature_controller import translate_text, save_translation_history, get_user_history, \
    delete_history_item
from app.models.history_model import TranslationRequest, HistoryItem

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/translate")
def translate(request: TranslationRequest = Body(...)):
    # Pass the source_lang from request
    result = translate_text(request.text, request.target_lang, request.source_lang)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/history")
def save_history(item: HistoryItem = Body(...)):
    return save_translation_history(item)


@router.get("/history/{user_id}")
def get_history(user_id: str):
    return get_user_history(user_id)


@router.delete("/history/{item_id}")
def delete_history(item_id: str, user_id: str):
    result = delete_history_item(item_id, user_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result