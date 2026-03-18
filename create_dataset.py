"""
create_dataset.py — S3 transcripts/ + chunks/ → HuggingFace Dataset

Usage:
  uv run python create_dataset.py projects/sidbi-ziri/config.yaml
"""
from __future__ import annotations

import json
import sys
import tempfile
import yaml
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from s3_utils import BUCKET, chunks_prefix, transcripts_prefix, download_file, make_client

load_dotenv()


def _load_all_transcripts(s3, project: str) -> list[dict]:
    tp = transcripts_prefix(project)
    records = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=tp):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue
            vid_id = Path(key).stem
            response = s3.get_object(Bucket=BUCKET, Key=key)
            for line in response["Body"].read().decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["video_id"] = vid_id
                    records.append(rec)
    logger.info(f"[{project}] {len(records)} segments chargés")
    return records


def run(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    project = cfg["project"]["name"]
    ds_cfg = cfg["dataset"]
    language = cfg["project"].get("language", "unknown")

    Path(cfg.get("logging", {}).get("log_file", "logs/pipeline.log")).parent.mkdir(parents=True, exist_ok=True)
    logger.add(cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), rotation="50 MB", encoding="utf-8")

    s3 = make_client()
    cp = chunks_prefix(project)

    records = _load_all_transcripts(s3, project)
    if not records:
        print(f"[{project}] Aucun transcript — lance d'abord transcribe_pipeline.py")
        return

    print(f"[{project}] {len(records)} segments — téléchargement FLAC...")

    import datasets
    import soundfile as sf

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for rec in tqdm(records, desc="Download FLAC"):
            s3_path = rec["path"]
            key = s3_path.replace(f"s3://{BUCKET}/", "")
            vid_id = rec["video_id"]
            local = tmp_dir / f"{vid_id}__{Path(key).name}"

            if not download_file(s3, key, local):
                continue
            try:
                audio_array, sr = sf.read(str(local), dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
            except Exception as e:
                logger.warning(f"Lecture FLAC {local}: {e}")
                local.unlink(missing_ok=True)
                continue

            rows.append({
                "audio": {"array": audio_array, "sampling_rate": sr, "path": str(local)},
                "text": rec["text"],
                "duration": rec.get("duration", len(audio_array) / sr),
                "language": language,
                "video_id": vid_id,
            })
            local.unlink(missing_ok=True)

        if not rows:
            print(f"[{project}] Aucun FLAC téléchargeable.")
            return

        print(f"[{project}] {len(rows)} segments → création dataset HF...")
        ds = datasets.Dataset.from_list(rows).cast_column("audio", datasets.Audio(sampling_rate=16000))

        if ds_cfg.get("push_to_hub", True):
            hf_repo = ds_cfg["hf_repo"]
            ds.push_to_hub(hf_repo, private=ds_cfg.get("hf_private", False))
            print(f"\n[{project}] Dataset → https://huggingface.co/datasets/{hf_repo}")
        else:
            out = Path(f"data/{project}_dataset")
            ds.save_to_disk(str(out))
            print(f"\n[{project}] Dataset sauvegardé → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_dataset.py projects/<name>/config.yaml")
        sys.exit(1)
    run(sys.argv[1])
