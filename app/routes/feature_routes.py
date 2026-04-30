from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from app.controllers.feature_controller import (
    translate_text,
    save_translation_history,
    get_user_history,
    delete_history_item,
    synthesize_speech,
)
from app.models.history_model import TranslationRequest, HistoryItem, TTSRequest

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/translate")
def translate(request: TranslationRequest = Body(...)):
    result = translate_text(request.text, request.target_lang, request.source_lang)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/tts")
def text_to_speech(request: TTSRequest = Body(...)):
    """
    Synthesizes speech using Google Cloud TTS and returns raw MP3 audio.
    Returns 503 when the API key is not configured.
    Returns 422 when the language is not supported by Google TTS
    (frontend should fall back to browser speechSynthesis in that case).
    """
    try:
        audio_bytes = synthesize_speech(request.text, request.language_code)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except ValueError as e:
        # Missing API key → 503  |  unsupported language → 422
        msg = str(e)
        if "not configured" in msg:
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=422, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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