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

ENDPOINT_URL = "https://fly.storage.tigris.dev"
AUDIO_EXTS = {".webm", ".wav", ".mp3", ".opus", ".ogg", ".m4a", ".flac"}


def make_client():
    from botocore.config import Config
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3", ENDPOINT_URL),
        config=Config(
            read_timeout=120,
            connect_timeout=30,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def upload_file(s3, bucket: str, local_path: Path | str, key: str) -> bool:
    try:
        s3.upload_file(str(local_path), bucket, key, Config=_TRANSFER_CFG)
        return True
    except Exception as e:
        logger.error(f"Upload S3 échoué {key}: {e}")
        return False


def upload_dir(s3, bucket: str, local_dir: Path, s3_prefix: str, max_workers: int = 8) -> int:
    files = [f for f in local_dir.iterdir() if f.is_file()]
    if not files:
        return 0
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(upload_file, s3, bucket, f, f"{s3_prefix}/{f.name}"): f for f in files}
        for fut in as_completed(futures):
            if fut.result():
                ok += 1
    return ok


def download_file(s3, bucket: str, key: str, local_path: Path | str) -> bool:
    try:
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception as e:
        logger.error(f"Download S3 échoué {key}: {e}")
        return False


def download_dir(s3, bucket: str, keys: list[str], local_dir: Path, vid_id: str, max_workers: int = 16) -> dict[str, str]:
    result: dict[str, str] = {}

    def _dl(key: str):
        local = local_dir / f"{vid_id}__{Path(key).name}"
        if download_file(s3, bucket, key, local):
            return str(local)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for local_path in ex.map(_dl, keys):
            if local_path:
                result[local_path] = vid_id
    return result


def delete_object(s3, bucket: str, key: str) -> bool:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        logger.error(f"Suppression S3 échouée {key}: {e}")
        return False


def list_raw_audio(s3, bucket: str, raw_root: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=raw_root.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() in AUDIO_EXTS:
                keys.append(obj["Key"])
    return keys


# ── Fonctions projet-aware ────────────────────────────────────────────────────

def chunks_prefix(project: str, datasets_root: str) -> str:
    return f"{datasets_root.rstrip('/')}/{project}/chunks/"


def transcripts_prefix(project: str, datasets_root: str) -> str:
    return f"{datasets_root.rstrip('/')}/{project}/transcripts/"


def list_chunk_video_ids(s3, bucket: str, project: str, datasets_root: str) -> list[str]:
    video_ids = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=chunks_prefix(project, datasets_root), Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            vid_id = p["Prefix"].rstrip("/").split("/")[-1]
            if vid_id:
                video_ids.append(vid_id)
    return video_ids


def list_chunks_for_video(s3, bucket: str, project: str, datasets_root: str, vid_id: str) -> list[str]:
    prefix = f"{chunks_prefix(project, datasets_root)}{vid_id}/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if Path(obj["Key"]).suffix.lower() == ".flac":
                keys.append(obj["Key"])
    return keys
