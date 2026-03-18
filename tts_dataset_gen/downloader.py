"""
downloader.py — Téléchargement parallèle via yt-dlp + upload Tigris S3.
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from s3_utils import BUCKET, RAW_PREFIX, make_client, upload_file

load_dotenv()


def _get_playlist_videos(playlist_url: str) -> list[dict]:
    logger.info(f"Récupération playlist: {playlist_url}")
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    videos = [
        {"id": e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}"}
        for e in info["entries"]
    ]
    logger.info(f"{len(videos)} vidéos — {info.get('title', '')}")
    return videos


def _download_and_upload(video: dict, tmp_dir: Path, s3) -> dict:
    vid_id = video["id"]
    marker = tmp_dir / f"{vid_id}.done"
    if marker.exists():
        return {"id": vid_id, "status": "cached"}

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(tmp_dir / f"{vid_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video["url"], download=True)
            out_path = Path(ydl.prepare_filename(info))
            if not out_path.exists():
                matches = [m for m in tmp_dir.glob(f"{vid_id}.*") if m.suffix != ".done"]
                if not matches:
                    return {"id": vid_id, "status": "error", "error": "fichier introuvable"}
                out_path = matches[0]
    except Exception as e:
        return {"id": vid_id, "status": "error", "error": str(e)[:200]}

    s3_key = f"{RAW_PREFIX}{out_path.name}"
    if upload_file(s3, out_path, s3_key):
        out_path.unlink()
        marker.touch()
        return {"id": vid_id, "path": f"s3://{BUCKET}/{s3_key}", "status": "uploaded"}
    return {"id": vid_id, "path": str(out_path), "status": "downloaded_only"}


def download_playlist(playlist_url: str, output_dir: str, max_workers: int = 4, s3=None) -> list[dict]:
    tmp = Path(output_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    videos = _get_playlist_videos(playlist_url)
    if not videos:
        logger.error("Aucune vidéo trouvée.")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_and_upload, v, tmp, s3): v for v in videos}
        with tqdm(total=len(futures), desc="Download→S3") as pbar:
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                if res["status"] == "error":
                    logger.warning(f"Erreur {res['id']}: {res.get('error', '')}")
                pbar.update(1)
                pbar.set_postfix({"ok": sum(1 for r in results if r["status"] != "error"), "status": res["status"]})

    ok = sum(1 for r in results if r["status"] in ("uploaded", "downloaded_only", "cached"))
    logger.info(f"Terminé: {ok}/{len(videos)}")
    return results


if __name__ == "__main__":
    import yaml, sys
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
    yt = cfg["youtube"]
    s3 = make_client()
    download_playlist(yt["playlist_url"], yt["download_dir"], yt.get("max_workers", 4), s3=s3)
