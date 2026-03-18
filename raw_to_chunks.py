"""
raw_to_chunks.py — Migration legacy: S3 raw/ → chunks/ (one-shot)

Pour chaque fichier dans S3 raw/:
  1. Download → local tmp
  2. VAD → FLAC
  3. Upload → S3 chunks/{vid_id}/
  4. Marker chunks/{vid_id}.done
  5. Supprime raw/{vid_id} de S3
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from s3_utils import (
    BUCKET, CHUNKS_PREFIX, RAW_PREFIX,
    delete_object, download_file, key_exists,
    list_raw_audio, make_client, upload_dir, upload_file,
)
from segmenter import segment_all

load_dotenv()


def _process_raw(raw_key: str, tmp_dir: Path, seg_cfg: dict, s3) -> dict:
    vid_id = Path(raw_key).stem
    marker_key = f"{CHUNKS_PREFIX}{vid_id}.done"

    if key_exists(s3, marker_key):
        return {"id": vid_id, "status": "cached"}

    local_audio = tmp_dir / Path(raw_key).name
    if not download_file(s3, raw_key, local_audio):
        return {"id": vid_id, "status": "error", "error": "download S3 échoué"}

    vid_chunks_dir = tmp_dir / f"chunks_{vid_id}"
    segs = segment_all(
        audio_paths=[str(local_audio)],
        output_dir=str(vid_chunks_dir),
        min_duration=seg_cfg.get("min_duration", 5.0),
        max_duration=seg_cfg.get("max_duration", 10.0),
        max_workers=1,
    )
    local_audio.unlink(missing_ok=True)

    if not segs:
        shutil.rmtree(vid_chunks_dir, ignore_errors=True)
        return {"id": vid_id, "status": "error", "error": "aucun segment VAD"}

    n_uploaded = upload_dir(s3, vid_chunks_dir, f"{CHUNKS_PREFIX}{vid_id}")
    shutil.rmtree(vid_chunks_dir, ignore_errors=True)

    if n_uploaded == 0:
        return {"id": vid_id, "status": "error", "error": "upload S3 échoué"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".done") as f:
        tmp_marker = f.name
    upload_file(s3, tmp_marker, marker_key)
    os.unlink(tmp_marker)
    delete_object(s3, raw_key)

    return {"id": vid_id, "n_chunks": n_uploaded, "status": "chunked"}


def run(config_path: str = "config.yaml", max_workers: int = 3):
    cfg = yaml.safe_load(open(config_path))
    seg_cfg = cfg["segmentation"]

    log_file = cfg.get("logging", {}).get("log_file", "logs/pipeline.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="50 MB", encoding="utf-8")

    s3 = make_client()
    raw_keys = list_raw_audio(s3)
    logger.info(f"{len(raw_keys)} fichiers dans raw/ à traiter")

    if not raw_keys:
        print("raw/ vide — rien à faire.")
        return

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_raw, k, tmp_dir, seg_cfg, s3): k for k in raw_keys}
            with tqdm(total=len(futures), desc="raw→chunks→S3") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    if res["status"] == "error":
                        logger.warning(f"Erreur {res['id']}: {res.get('error', '')}")
                    pbar.update(1)
                    pbar.set_postfix({"ok": sum(1 for r in results if r["status"] in ("chunked", "cached"))})

    chunked = sum(1 for r in results if r["status"] == "chunked")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\n{chunked} chunked, {cached} cached, {errors} erreurs")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
