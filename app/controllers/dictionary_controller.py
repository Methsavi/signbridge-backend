from bson import ObjectId
from app.core.database import get_database
from app.models.dictionary_model import DictionaryEntryCreate, DictionaryEntryUpdate
from datetime import datetime, timezone


def _serialize(entry: dict) -> dict:
    """Convert MongoDB document to JSON-serialisable dict."""
    entry["id"] = str(entry.pop("_id"))
    return entry


# ── CREATE ────────────────────────────────────────────────────────────────────
def create_entry(payload: DictionaryEntryCreate) -> dict:
    db = get_database()
    doc = {
        "label": payload.label.strip(),
        "category": payload.category,
        "media_type": payload.media_type,
        "media_url": payload.media_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    result = db["dictionary"].insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


# ── READ ALL (with optional filters) ─────────────────────────────────────────
def list_entries(category: str | None = None, search: str | None = None) -> list[dict]:
    db = get_database()
    query: dict = {}

    if category:
        query["category"] = category

    if search:
        query["label"] = {"$regex": search, "$options": "i"}

    entries = list(db["dictionary"].find(query).sort("label", 1))
    return [_serialize(e) for e in entries]


# ── READ ONE ──────────────────────────────────────────────────────────────────
def get_entry(entry_id: str) -> dict | None:
    db = get_database()
    try:
        entry = db["dictionary"].find_one({"_id": ObjectId(entry_id)})
    except Exception:
        return None
    if entry:
        return _serialize(entry)
    return None


# ── UPDATE ────────────────────────────────────────────────────────────────────
def update_entry(entry_id: str, payload: DictionaryEntryUpdate) -> dict:
    db = get_database()
    try:
        oid = ObjectId(entry_id)
    except Exception:
        return {"error": "Invalid entry id"}

    update_fields = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not update_fields:
        return {"error": "No fields to update"}

    update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = db["dictionary"].find_one_and_update(
        {"_id": oid},
        {"$set": update_fields},
        return_document=True,
    )
    if not result:
        return {"error": "Entry not found"}
    return _serialize(result)


# ── DELETE ────────────────────────────────────────────────────────────────────
def delete_entry(entry_id: str) -> dict:
    db = get_database()
    try:
        oid = ObjectId(entry_id)
    except Exception:
        return {"error": "Invalid entry id"}

    result = db["dictionary"].delete_one({"_id": oid})
    if result.deleted_count == 0:
        return {"error": "Entry not found"}
    return {"message": "Entry deleted successfully"}
