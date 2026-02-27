"""
MinIO artifact storage client.
"""
import io
import json
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.settings import get_settings
from app.logging_config import get_logger

logger = get_logger("minio")

_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
    return _client


def ensure_bucket():
    settings = get_settings()
    client = get_minio_client()
    bucket = settings.minio_bucket
    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("created_minio_bucket", bucket=bucket)
    except S3Error as e:
        logger.error("minio_bucket_error", error=str(e))


def upload_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    settings = get_settings()
    client = get_minio_client()
    client.put_object(
        settings.minio_bucket,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return f"minio://{settings.minio_bucket}/{object_name}"


def upload_file(object_name: str, file_path: str, content_type: str = "application/octet-stream") -> str:
    settings = get_settings()
    client = get_minio_client()
    client.fput_object(settings.minio_bucket, object_name, file_path, content_type=content_type)
    return f"minio://{settings.minio_bucket}/{object_name}"


def upload_json(object_name: str, data: dict) -> str:
    raw = json.dumps(data, indent=2, default=str).encode()
    return upload_bytes(object_name, raw, content_type="application/json")


def get_presigned_url(object_name: str, expires_seconds: int = 3600) -> str:
    from datetime import timedelta
    settings = get_settings()
    client = get_minio_client()
    try:
        url = client.presigned_get_object(
            settings.minio_bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )
        return url
    except S3Error:
        return f"minio://{settings.minio_bucket}/{object_name}"


def download_bytes(object_name: str) -> bytes:
    settings = get_settings()
    client = get_minio_client()
    response = client.get_object(settings.minio_bucket, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
