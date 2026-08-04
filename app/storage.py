"""
storage.py — Módulo de integración con Cloudflare R2.
"""

import os
import base64
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "rag-bucket")
_PUBLIC_URL_BASE = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

_R2_ENDPOINT = f"https://{_ACCOUNT_ID}.r2.cloudflarestorage.com"

_client = None

def _get_client():
    """Retorna el cliente boto3 apuntando a R2 (singleton lazy)."""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=_R2_ENDPOINT,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto", 
        )
    return _client

def upload_pdf(pdf_bytes: bytes, project_id: int, filename: str) -> tuple[str, str]:
    client = _get_client()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = filename.replace(" ", "_")
    key = f"projects/{project_id}/{timestamp}_{safe_filename}"

    try:
        client.put_object(
            Bucket=_BUCKET_NAME,
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Error al subir archivo a R2: {e}") from e

    r2_uri = f"r2://{_BUCKET_NAME}/{key}"
    public_url = f"{_PUBLIC_URL_BASE}/{key}" if _PUBLIC_URL_BASE else r2_uri

    return r2_uri, public_url

def upload_pdf_from_base64(
    pdf_base64: str, project_id: int, filename: str
) -> tuple[str, str]:
    pdf_bytes = base64.b64decode(pdf_base64)
    return upload_pdf(pdf_bytes, project_id, filename)
