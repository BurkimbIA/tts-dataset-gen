"""
segmenter.py — Segmentation audio parallèle via Silero VAD.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger


def _load_silero_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps, *_ = utils
    return model, get_speech_timestamps


def _to_wav16k(src: Path, dst: Path) -> bool:
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst), "-loglevel", "error"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return True
    except Exception:
        return False


def _segment_one(args: tuple) -> list[dict]:
    audio_path, output_dir, min_dur, max_dur = args
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        model, get_speech_timestamps = _load_silero_vad()

        if audio_path.suffix.lower() != ".wav":
            tmp_wav = output_dir / f"_tmp_{audio_path.stem}.wav"
            if not _to_wav16k(audio_path, tmp_wav):
                logger.error(f"Conversion ffmpeg échouée: {audio_path.name}")
                return []
        else:
            tmp_wav = audio_path

        audio_np, sr = sf.read(str(tmp_wav), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        wav = torch.from_numpy(audio_np)

        if tmp_wav != audio_path:
            tmp_wav.unlink(missing_ok=True)

        timestamps = get_speech_timestamps(
            wav, model,
            sampling_rate=16000,
            min_silence_duration_ms=300,
            min_speech_duration_ms=int(min_dur * 1000),
        )

        segments = []

        def _flush(start: int, end: int):
            chunk_samples = int(max_dur * 16000)
            pos = start
            while pos < end:
                chunk_end = min(pos + chunk_samples, end)
                dur = (chunk_end - pos) / 16000
                if dur >= min_dur:
                    seg_id = f"{audio_path.stem}_{pos:08d}"
                    seg_path = output_dir / f"{seg_id}.flac"
                    sf.write(str(seg_path), wav[pos:chunk_end].numpy(), 16000, format="flac")
                    segments.append({"path": str(seg_path), "duration": dur})
                pos = chunk_end

        current_start = current_end = None
        for ts in timestamps:
            if current_start is None:
                current_start, current_end = ts["start"], ts["end"]
            else:
                if (ts["end"] - current_start) / 16000 <= max_dur:
                    current_end = ts["end"]
                else:
                    _flush(current_start, current_end)
                    current_start, current_end = ts["start"], ts["end"]

        if current_start is not None:
            _flush(current_start, current_end)

        return segments

    except Exception as e:
        logger.error(f"Erreur segmentation {audio_path.name}: {e}")
        return []


def segment_all(
    audio_paths: list[str],
    output_dir: str,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    max_workers: int = 4,
) -> list[dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args_list = [(p, output_dir, min_duration, max_duration) for p in audio_paths]
    all_segments = []

    logger.info(f"Segmentation de {len(audio_paths)} fichiers ({max_workers} workers)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_segment_one, args): args[0] for args in args_list}
        with tqdm(total=len(futures), desc="Segmentation") as pbar:
            for future in as_completed(futures):
                all_segments.extend(future.result())
                pbar.update(1)
                pbar.set_postfix({"segments": len(all_segments)})

    logger.info(f"Segmentation terminée: {len(all_segments)} segments")
    return all_segments
