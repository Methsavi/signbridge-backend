from deep_translator import GoogleTranslator
from app.core.database import get_database
from app.models.history_model import HistoryItem
from datetime import datetime
from bson import ObjectId
from gtts import gTTS
import requests
import os
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


# --- ASL GLOSS CONVERSION ---
def convert_to_asl_gloss(text: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    prompt = (
        "Convert the following English sentence to ASL gloss notation.\n"
        "Rules:\n"
        "- Remove articles (a, an, the)\n"
        "- Remove helping verbs (is, are, am, was, were, be, been, being, do, does, did)\n"
        "- Remove conjunctions when possible (and, but, or)\n"
        "- Use Topic-Comment word order when appropriate (e.g. 'I am hungry' → 'HUNGRY ME')\n"
        "- Output ONLY uppercase gloss words separated by spaces — no punctuation, no explanation\n\n"
        f"English: {text}\n"
        "ASL Gloss:"
    )

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 100,
        },
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Groq API {resp.status_code}: {resp.text}")

    raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
    gloss = " ".join(w.strip(".,!?;:\"'") for w in raw.split() if w.strip(".,!?;:\"'"))

    return {
        "original": text,
        "gloss": gloss,
        "words": gloss.split(),
    }


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