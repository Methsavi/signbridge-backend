from pydantic import BaseModel
from typing import Optional, Literal


class DictionaryEntryCreate(BaseModel):
    label: str                                     # e.g. "A", "Hello", "1"
    category: Literal["letter", "number", "word", "sentence"]
    media_type: Literal["image", "video"]
    media_url: str                                 # R2 public URL after upload


class DictionaryEntryUpdate(BaseModel):
    label: Optional[str] = None
    category: Optional[Literal["letter", "number", "word", "sentence"]] = None
    media_type: Optional[Literal["image", "video"]] = None
    media_url: Optional[str] = None
