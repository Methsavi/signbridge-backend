from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class HistoryItem(BaseModel):
    user_id: str
    original_text: str
    translated_text: str
    target_language: str
    timestamp: datetime = Field(default_factory=datetime.now)

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "auto"

class TTSRequest(BaseModel):
    text: str
    language_code: str = "en"