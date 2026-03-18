"""
transcriber.py — Batch transcription using BIA-WHISPER (Whisper fine-tuné Mooré).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

SAMPLE_RATE = 16_000


class Transcriber:

    def __init__(
        self,
        model_id: str = "burkimbia/BIA-WHISPER-LARGE-SACHI_V2",
        local_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.model_path = local_dir if (local_dir and Path(local_dir).exists()) else model_id

        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else (device if device != "auto" else "cpu")
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Chargement Whisper: {model_id} sur {self.device}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_path)

        self._model = model
        self._processor = processor
        self._pipelines: dict = {}

    def _get_pipeline(self, is_long_audio: bool = False, chunk_length_s: int = 30, batch_size: int = 8):
        key = f"{is_long_audio}_{chunk_length_s}_{batch_size}"
        if key not in self._pipelines:
            kwargs = dict(
                task="automatic-speech-recognition",
                model=self._model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            if is_long_audio:
                kwargs["chunk_length_s"] = chunk_length_s
                kwargs["batch_size"] = batch_size
            self._pipelines[key] = pipeline(**kwargs)
        return self._pipelines[key]

    def transcribe_batch(
        self,
        audio_paths: list[str],
        batch_size: int = 8,
        language: str = "hausa",
        filters: dict | None = None,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
    ) -> list[dict]:
        if language in ("mos", "moor", "moore"):
            language = "hausa"

        min_chars: int = (filters or {}).get("min_chars", 3)
        arrays, metas = [], []
        for p in audio_paths:
            try:
                audio, sr = sf.read(str(p), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                duration = len(audio) / SAMPLE_RATE
                if not (min_duration <= duration <= max_duration):
                    continue
                arrays.append({"array": audio, "sampling_rate": SAMPLE_RATE})
                metas.append({"path": str(p), "duration": duration})
            except Exception as e:
                logger.warning(f"Lecture {p} échouée: {e}")

        if not arrays:
            return []

        pipe = self._get_pipeline(is_long_audio=False, batch_size=batch_size)
        logger.info(f"Transcription de {len(arrays)} segments (batch_size={batch_size})...")

        CHUNK = 128  # évite l'OOM cross-vidéos
        all_results = []
        for i in range(0, len(arrays), CHUNK):
            sub = arrays[i: i + CHUNK]
            with torch.no_grad():
                all_results.extend(pipe(sub, batch_size=batch_size, generate_kwargs={"language": language, "do_sample": False, "num_beams": 1, "max_new_tokens": 444}))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records = []
        for meta, res in zip(metas, all_results):
            text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
            if len(text) < min_chars:
                continue
            records.append({**meta, "text": text, "language": language})

        logger.info(f"Transcription terminée: {len(records)}/{len(arrays)} segments gardés")
        return records
