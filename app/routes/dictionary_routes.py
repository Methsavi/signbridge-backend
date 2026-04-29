import mimetypes
import os
from uuid import uuid4

import boto3
from botocore.config import Config

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile, status

from app.controllers.dictionary_controller import (
    create_entry,
    delete_entry,
    get_entry,
    list_entries,
    update_entry,
)
from app.models.dictionary_model import DictionaryEntryCreate, DictionaryEntryUpdate

router = APIRouter(prefix="/dictionary", tags=["dictionary"])


# ── helpers ────────────────────────────────────────────────────────────────────
def _upload_media(file: UploadFile, file_data: bytes) -> str:
    """Upload sign media to Cloudflare R2 and return the public URL."""
    endpoint_url = os.getenv("R2_S3_CLIENT_URL") or os.getenv("R2_S3_CLIENTS")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KET_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    public_base_url = (os.getenv("R2_PUBLIC_BASE_URL") or "").rstrip("/")

    if not all([endpoint_url, access_key_id, secret_access_key, bucket_name, public_base_url]):
        raise HTTPException(status_code=500, detail="R2 storage is not configured")

    # Determine extension
    ext = ""
    if file.filename and "." in file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    else:
        guessed = mimetypes.guess_extension(file.content_type or "")
        ext = guessed or ".bin"

    key = f"dictionary/{uuid4().hex}{ext}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=file_data,
        ContentType=file.content_type or "application/octet-stream",
    )
    return f"{public_base_url}/{key}"


# ── UPLOAD MEDIA (standalone endpoint) ────────────────────────────────────────
@router.post("/upload-media", status_code=status.HTTP_200_OK)
async def upload_media(file: UploadFile = File(...)):
    """Upload an image or video to R2 and return the public URL."""
    try:
        file_data = await file.read()
        if len(file_data) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 50 MB)")
        url = _upload_media(file, file_data)
        return {"url": url, "media_type": "video" if (file.content_type or "").startswith("video") else "image"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET ALL ────────────────────────────────────────────────────────────────────
@router.get("/", status_code=status.HTTP_200_OK)
def get_entries(
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
):
    try:
        entries = list_entries(category=category, search=search)
        return {"items": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET ONE ────────────────────────────────────────────────────────────────────
@router.get("/{entry_id}", status_code=status.HTTP_200_OK)
def get_one_entry(entry_id: str):
    try:
        entry = get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── CREATE ─────────────────────────────────────────────────────────────────────
@router.post("/", status_code=status.HTTP_201_CREATED)
def create_dictionary_entry(payload: DictionaryEntryCreate = Body(...)):
    try:
        return create_entry(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── UPDATE ─────────────────────────────────────────────────────────────────────
@router.put("/{entry_id}", status_code=status.HTTP_200_OK)
def update_dictionary_entry(entry_id: str, payload: DictionaryEntryUpdate = Body(...)):
    try:
        result = update_entry(entry_id, payload)
        if "error" in result:
            code = 404 if result["error"] in ["Entry not found", "Invalid entry id"] else 400
            raise HTTPException(status_code=code, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DELETE ─────────────────────────────────────────────────────────────────────
@router.delete("/{entry_id}", status_code=status.HTTP_200_OK)
def delete_dictionary_entry(entry_id: str):
    try:
        result = delete_entry(entry_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
