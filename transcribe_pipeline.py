"""
transcribe_pipeline.py — Transcription par vidéo depuis S3 {project}/chunks/

Usage:
  uv run python transcribe_pipeline.py projects/sidbi-ziri/config.yaml
  uv run python transcribe_pipeline.py projects/sidbi-ziri/config.yaml 50
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

from tts_dataset_gen.s3_utils import (
    chunks_prefix, transcripts_prefix,
    download_dir, key_exists,
    list_chunk_video_ids, list_chunks_for_video,
    make_client, upload_file,
)
from tts_dataset_gen.transcriber import Transcriber

load_dotenv()


def run(config_path: str, videos_per_window: int = 20):
    cfg = yaml.safe_load(open(config_path))
    project = cfg["project"]["name"]
    tr_cfg = cfg["transcription"]
    flt_cfg = cfg.get("filters", {})

    Path(cfg.get("logging", {}).get("log_file", "logs/pipeline.log")).parent.mkdir(parents=True, exist_ok=True)
    logger.add(cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), rotation="50 MB", encoding="utf-8")

    s3 = make_client()
    tp = transcripts_prefix(project)
    cp = chunks_prefix(project)

    all_video_ids = list_chunk_video_ids(s3, project)
    pending = [v for v in all_video_ids if not key_exists(s3, f"{tp}{v}.jsonl")]
    logger.info(f"[{project}] {len(pending)} à transcrire / {len(all_video_ids)} chunked")

    if not pending:
        print(f"[{project}] Rien à transcrire.")
        return

    transcriber = Transcriber(
        model_id=tr_cfg["model_id"],
        local_dir=tr_cfg.get("local_dir"),
        device=tr_cfg.get("device", "auto"),
    )

    total_segments = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for vid_idx, vid_id in enumerate(tqdm(pending, desc=f"[{project}] Transcription")):
            keys = list_chunks_for_video(s3, project, vid_id)
            if not keys:
                continue
            local_paths = list(download_dir(s3, keys, tmp_dir, vid_id).keys())
            if not local_paths:
                continue

            records = transcriber.transcribe_batch(
                audio_paths=local_paths,
                batch_size=tr_cfg.get("batch_size", 8),
                language=cfg["project"].get("language", "hausa"),
                filters=flt_cfg,
            )

            for r in records:
                fname = Path(r["path"]).name.split("__", 1)[-1]
                r["path"] = f"s3://burkimbia/{cp}{vid_id}/{fname}"

            jsonl_local = tmp_dir / f"{vid_id}.jsonl"
            with open(jsonl_local, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            upload_file(s3, jsonl_local, f"{tp}{vid_id}.jsonl")
            jsonl_local.unlink(missing_ok=True)

            for p in local_paths:
                Path(p).unlink(missing_ok=True)

            total_segments += len(records)
            logger.info(f"  [{vid_idx+1}/{len(pending)}] {vid_id} → {len(records)} segments")

    logger.info(f"[{project}] Terminé — {total_segments} segments")
    print(f"\n[{project}] {total_segments} segments pour {len(pending)} vidéos")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_pipeline.py projects/<name>/config.yaml [videos_per_window]")
        sys.exit(1)
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
