"""
s3_utils.py — Helpers S3 partagés (Tigris via boto3).
"""
from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from loguru import logger

_TRANSFER_CFG = TransferConfig(multipart_threshold=200 * 1024 * 1024, use_threads=True)

BUCKET = "burkimbia"
ENDPOINT_URL = "https://fly.storage.tigris.dev"
RAW_PREFIX = "augmented-data-for-tts/raw/"
CHUNKS_PREFIX = "augmented-data-for-tts/chunks/"
AUDIO_EXTS = {".webm", ".wav", ".mp3", ".opus", ".ogg", ".m4a", ".flac"}


def make_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3", ENDPOINT_URL),
    )


def key_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def upload_file(s3, local_path: Path | str, key: str) -> bool:
    try:
        s3.upload_file(str(local_path), BUCKET, key, Config=_TRANSFER_CFG)
        return True
    except Exception as e:
        logger.error(f"Upload S3 échoué {key}: {e}")
        return False


def upload_dir(s3, local_dir: Path, s3_prefix: str, max_workers: int = 8) -> int:
    files = [f for f in local_dir.iterdir() if f.is_file()]
    if not files:
        return 0
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(upload_file, s3, f, f"{s3_prefix}/{f.name}"): f for f in files}
        for fut in as_completed(futures):
            if fut.result():
                ok += 1
    return ok


def download_file(s3, key: str, local_path: Path | str) -> bool:
    try:
        s3.download_file(BUCKET, key, str(local_path))
        return True
    except Exception as e:
        logger.error(f"Download S3 échoué {key}: {e}")
        return False


def download_dir(s3, keys: list[str], local_dir: Path, vid_id: str, max_workers: int = 16) -> dict[str, str]:
    result: dict[str, str] = {}

    def _dl(key: str):
        local = local_dir / f"{vid_id}__{Path(key).name}"
        if download_file(s3, key, local):
            return str(local)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for local_path in ex.map(_dl, keys):
            if local_path:
                result[local_path] = vid_id
    return result


def delete_object(s3, key: str) -> bool:
    try:
        s3.delete_object(Bucket=BUCKET, Key=key)
        return True
    except Exception as e:
        logger.error(f"Suppression S3 échouée {key}: {e}")
        return False


def list_raw_audio(s3) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=RAW_PREFIX):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() in AUDIO_EXTS:
                keys.append(obj["Key"])
    return keys


# ── Fonctions projet-aware ────────────────────────────────────────────────────

def chunks_prefix(project: str) -> str:
    return f"{project}/chunks/"


def transcripts_prefix(project: str) -> str:
    return f"{project}/transcripts/"


def list_chunk_video_ids(s3, project: str) -> list[str]:
    video_ids = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=chunks_prefix(project), Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            vid_id = p["Prefix"].rstrip("/").split("/")[-1]
            if vid_id:
                video_ids.append(vid_id)
    return video_ids


def list_chunks_for_video(s3, project: str, vid_id: str) -> list[str]:
    prefix = f"{chunks_prefix(project)}{vid_id}/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() == ".flac":
                keys.append(obj["Key"])
    return keys
