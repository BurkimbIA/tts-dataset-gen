"""
create_dataset.py — S3 transcripts/ + chunks/ -> HuggingFace Dataset

Usage:
  uv run python create_dataset.py projects/sidbi-ziri/config.yaml
"""
from __future__ import annotations

import json
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from tts_dataset_gen.s3_utils import chunks_prefix, transcripts_prefix, download_file, make_client

load_dotenv()

CACHE_DIR = Path("data/flac_cache")


def _load_all_transcripts(s3, bucket: str, project: str, datasets_root: str) -> list[dict]:
    tp = transcripts_prefix(project, datasets_root)
    records = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=tp):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            vid_id = Path(key).stem
            response = s3.get_object(Bucket=bucket, Key=key)
            for line in response["Body"].read().decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Ligne JSONL corrompue ignoree dans {key}")
                        continue
                    rec["video_id"] = vid_id
                    records.append(rec)
    logger.info(f"[{project}] {len(records)} segments charges")
    return records


def _download_one(args):
    s3, bucket, key, local_path = args
    if local_path.exists():
        return local_path  # cache hit
    if download_file(s3, bucket, key, local_path):
        return local_path
    return None


def run(config_path: str, max_workers: int = 64):
    cfg = yaml.safe_load(open(config_path))
    project = cfg["project"]["name"]
    ds_cfg = cfg["dataset"]
    language = cfg["project"].get("language", "unknown")
    bucket = cfg["bucket"]["name"]
    datasets_root = cfg["bucket"]["paths"]["datasets"]

    Path(cfg.get("logging", {}).get("log_file", "logs/pipeline.log")).parent.mkdir(parents=True, exist_ok=True)
    logger.add(cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), rotation="50 MB", encoding="utf-8")

    cache_dir = CACHE_DIR / project
    cache_dir.mkdir(parents=True, exist_ok=True)

    s3 = make_client()

    records = _load_all_transcripts(s3, bucket, project, datasets_root)
    if not records:
        print(f"[{project}] Aucun transcript — lance d'abord transcribe_pipeline.py")
        return

    # Preparer les taches
    tasks = []
    for rec in records:
        s3_path = rec["path"]
        key = s3_path.replace(f"s3://{bucket}/", "")
        vid_id = rec["video_id"]
        local = cache_dir / f"{vid_id}__{Path(key).name}"
        tasks.append((s3, bucket, key, local, rec))

    cached = sum(1 for _, _, _, local, _ in tasks if local.exists())
    print(f"[{project}] {len(records)} segments — {cached} en cache, {len(records)-cached} a telecharger ({max_workers} workers)...")

    import datasets

    rec_by_local = {str(local): rec for _, _, _, local, rec in tasks}
    download_args = [(s3, bucket, key, local) for s3, bucket, key, local, _ in tasks]

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, arg): arg[2] for arg in download_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="FLAC"):
            local_path = fut.result()
            if local_path is None:
                continue
            if not local_path.exists():
                continue
            rec = rec_by_local[str(local_path)]
            rows.append({
                "audio": str(local_path),
                "text": rec["text"],
                "duration": rec.get("duration", 0.0),
                "language": language,
                "video_id": rec["video_id"],
            })

    if not rows:
        print(f"[{project}] Aucun FLAC disponible.")
        return

    print(f"[{project}] {len(rows)} segments -> creation dataset HF...")
    ds = datasets.Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=False))

    if ds_cfg.get("push_to_hub", True):
        hf_repo = ds_cfg["hf_repo"]
        ds.push_to_hub(hf_repo, private=ds_cfg.get("hf_private", False))
        print(f"\n[{project}] Dataset pushed -> https://huggingface.co/datasets/{hf_repo}")
    else:
        out = Path(f"data/{project}_dataset")
        ds.save_to_disk(str(out))
        print(f"\n[{project}] Dataset sauvegarde -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_dataset.py projects/<name>/config.yaml")
        sys.exit(1)
    run(sys.argv[1])
