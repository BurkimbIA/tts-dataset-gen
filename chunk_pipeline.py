"""
chunk_pipeline.py — YouTube → VAD segments → S3 {project}/chunks/

Usage:
  uv run python chunk_pipeline.py projects/sidbi-ziri/config.yaml
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from s3_utils import chunks_prefix, key_exists, make_client, upload_dir, upload_file
from segmenter import segment_all

load_dotenv()


def _get_playlist_videos(playlist_url: str) -> list[dict]:
    logger.info(f"Récupération playlist: {playlist_url}")
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    videos = [{"id": e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}"} for e in info["entries"]]
    logger.info(f"{len(videos)} vidéos — {info.get('title', '')}")
    return videos


def _process_one(video: dict, tmp_dir: Path, seg_cfg: dict, project: str, s3) -> dict:
    vid_id = video["id"]
    marker_key = f"{chunks_prefix(project)}{vid_id}.done"

    if key_exists(s3, marker_key):
        return {"id": vid_id, "status": "cached"}

    cookies_file = Path(__file__).parent / "cookies.txt"
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(tmp_dir / f"{vid_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        **({"cookiefile": str(cookies_file)} if cookies_file.exists() else {}),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video["url"], download=True)
            audio_path = Path(ydl.prepare_filename(info))
            if not audio_path.exists():
                matches = [m for m in tmp_dir.glob(f"{vid_id}.*") if m.suffix != ".done"]
                if not matches:
                    return {"id": vid_id, "status": "error", "error": "fichier introuvable"}
                audio_path = matches[0]
    except Exception as e:
        return {"id": vid_id, "status": "error", "error": str(e)[:200]}

    vid_chunks_dir = tmp_dir / f"chunks_{vid_id}"
    segs = segment_all(
        audio_paths=[str(audio_path)],
        output_dir=str(vid_chunks_dir),
        min_duration=seg_cfg.get("min_duration", 5.0),
        max_duration=seg_cfg.get("max_duration", 10.0),
        max_workers=1,
    )
    audio_path.unlink(missing_ok=True)

    if not segs:
        shutil.rmtree(vid_chunks_dir, ignore_errors=True)
        return {"id": vid_id, "status": "error", "error": "aucun segment VAD"}

    n_uploaded = upload_dir(s3, vid_chunks_dir, f"{chunks_prefix(project)}{vid_id}")
    shutil.rmtree(vid_chunks_dir, ignore_errors=True)

    if n_uploaded == 0:
        return {"id": vid_id, "status": "error", "error": "upload S3 échoué"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".done") as f:
        tmp_marker = f.name
    upload_file(s3, tmp_marker, marker_key)
    os.unlink(tmp_marker)

    return {"id": vid_id, "n_chunks": n_uploaded, "status": "chunked"}


def run(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    project = cfg["project"]["name"]
    yt_cfg = cfg["youtube"]
    seg_cfg = cfg["segmentation"]

    Path(cfg.get("logging", {}).get("log_file", "logs/pipeline.log")).parent.mkdir(parents=True, exist_ok=True)
    logger.add(cfg.get("logging", {}).get("log_file", "logs/pipeline.log"), rotation="50 MB", encoding="utf-8")

    s3 = make_client()
    videos = _get_playlist_videos(yt_cfg["playlist_url"])

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=yt_cfg.get("max_workers", 3)) as executor:
            futures = {executor.submit(_process_one, v, tmp_dir, seg_cfg, project, s3): v for v in videos}
            with tqdm(total=len(futures), desc=f"[{project}] Chunk→S3") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    results.append(res)
                    if res["status"] == "error":
                        logger.warning(f"Erreur {res['id']}: {res.get('error', '')}")
                    pbar.update(1)
                    pbar.set_postfix({"ok": sum(1 for r in results if r["status"] in ("chunked", "cached")), "status": res["status"]})

    chunked = sum(1 for r in results if r["status"] == "chunked")
    cached = sum(1 for r in results if r["status"] == "cached")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(f"[{project}] {chunked} chunked, {cached} cached, {errors} erreurs")
    print(f"\n[{project}] {chunked} chunked, {cached} cached, {errors} erreurs")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunk_pipeline.py projects/<name>/config.yaml")
        sys.exit(1)
    run(sys.argv[1])
