from deep_translator import GoogleTranslator
from app.core.database import get_database
from app.models.history_model import HistoryItem
from datetime import datetime
from bson import ObjectId
from gtts import gTTS
import io


# --- TRANSLATION LOGIC ---
def translate_text(text: str, target_lang: str, source_lang: str = 'auto'):
    try:
        # Use the provided source_lang (or 'auto')
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return {
            "original": text,
            "translated": translated,
            "source": source_lang,
            "target": target_lang
        }
    except Exception as e:
        return {"error": str(e)}


# --- HISTORY LOGIC ---
def save_translation_history(item: HistoryItem):
    db = get_database()
    history_collection = db["history"]

    doc = item.dict()
    result = history_collection.insert_one(doc)

    return {"msg": "Saved to history", "id": str(result.inserted_id)}


def get_user_history(user_id: str):
    db = get_database()
    history_collection = db["history"]

    cursor = history_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(100)

    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)

    return results


def delete_history_item(item_id: str, user_id: str):
    db = get_database()
    history_collection = db["history"]

    try:
        result = history_collection.delete_one({
            "_id": ObjectId(item_id),
            "user_id": user_id
        })

        if result.deleted_count == 1:
            return {"msg": "Deleted successfully"}
        else:
            return {"error": "Item not found or permission denied"}
    except Exception as e:
        return {"error": str(e)}


# --- TEXT-TO-SPEECH LOGIC ---
def synthesize_speech(text: str, language_code: str) -> bytes:
    """
    Synthesizes speech using gTTS (Google Translate TTS — free, no API key).
    Returns raw MP3 bytes.
    Raises ValueError for unsupported language codes (caller returns 422).
    """
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except ValueError as e:
        raise ValueError(f"Language '{language_code}' is not supported: {e}")