import mimetypes
import os
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class R2ConfigError(RuntimeError):
    pass


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _safe_ext(filename: str | None, content_type: str | None) -> str:
    if filename and "." in filename:
        return f".{filename.rsplit('.', 1)[-1].lower()}"
    guessed = mimetypes.guess_extension(content_type or "")
    return guessed or ".jpg"


def _safe_segment(raw: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw).strip("-")
    return cleaned[:64] or "user"


def _build_client_and_config():
    endpoint_url = _first_env("R2_S3_CLIENT_URL", "R2_S3_CLIENTS")
    access_key_id = _first_env("R2_ACCESS_KEY_ID", "R2_ACCESS_KET_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    public_base_url = os.getenv("R2_PUBLIC_BASE_URL")
    profile_prefix = os.getenv("R2_PROFILE_IMAGE_PREFIX", "profile-images").strip("/")

    missing = []
    if not endpoint_url:
        missing.append("R2_S3_CLIENT_URL (or R2_S3_CLIENTS)")
    if not access_key_id:
        missing.append("R2_ACCESS_KEY_ID (or R2_ACCESS_KET_ID)")
    if not secret_access_key:
        missing.append("R2_SECRET_ACCESS_KEY")
    if not bucket_name:
        missing.append("R2_BUCKET_NAME")
    if not public_base_url:
        missing.append("R2_PUBLIC_BASE_URL")

    if missing:
        raise R2ConfigError(f"Missing Cloudflare R2 environment variables: {', '.join(missing)}")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    return client, bucket_name, public_base_url.rstrip("/"), profile_prefix


def ensure_profile_image_directory() -> None:
    client, bucket_name, _, profile_prefix = _build_client_and_config()
    keep_key = f"{profile_prefix}/.keep"

    try:
        client.head_object(Bucket=bucket_name, Key=keep_key)
        return
    except ClientError as exc:
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code != 404:
            raise

    client.put_object(Bucket=bucket_name, Key=keep_key, Body=b"")


def upload_profile_image(
    user_identifier: str,
    file_data: bytes,
    content_type: str | None,
    original_filename: str | None = None,
) -> str:
    client, bucket_name, public_base_url, profile_prefix = _build_client_and_config()

    extension = _safe_ext(original_filename, content_type)
    key = f"{profile_prefix}/{_safe_segment(user_identifier)}/{uuid4().hex}{extension}"

    client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=file_data,
        ContentType=content_type or "application/octet-stream",
    )

    return f"{public_base_url}/{key}"
