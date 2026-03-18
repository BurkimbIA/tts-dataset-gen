# tts-dataset-gen

**Automatisation de la collecte et transcription de playlists YouTube en Mooré pour construire des datasets TTS.**

Ce projet télécharge automatiquement des émissions YouTube en langue Mooré, segmente l'audio, transcrit avec BIA-Whisper et publie un dataset HuggingFace prêt pour le fine-tuning TTS.

![Strategy](docs/strategy.png)

![Architecture](docs/architecture.png)

---

## Objectif

Le Mooré est une langue sans ressources numériques suffisantes pour l'IA. Ce pipeline permet de **transformer n'importe quelle playlist YouTube en Mooré en dataset TTS structuré**, de façon automatique et reproductible.

```
Playlist YouTube (Mooré) → VAD → BIA-Whisper → HuggingFace Dataset → Fine-tuning TTS
```

---

## Multi-projets

Chaque **projet** correspond à une ou plusieurs playlists YouTube. Il suffit d'ajouter un `config.yaml` dans `projects/` :

```
projects/
├── sidbi-ziri/         # Émission SID BI ZÉRÉ — SAVANE TV (247 vidéos)
│   └── config.yaml
├── kibare/             # Émission KIBARE — (à venir)
│   └── config.yaml
└── mon-projet/         # Toute autre playlist Mooré
    └── config.yaml
```

Le même pipeline s'exécute pour chaque projet :
```bash
uv run --no-sync python chunk_pipeline.py      projects/kibare/config.yaml
uv run --no-sync python transcribe_pipeline.py projects/kibare/config.yaml
uv run --no-sync python create_dataset.py      projects/kibare/config.yaml
```

---

## Structure

```
tts-dataset-gen/
├── chunk_pipeline.py        # Étape 1 : Download YouTube + Segmentation VAD → S3
├── transcribe_pipeline.py   # Étape 2 : Transcription Whisper → S3
├── create_dataset.py        # Étape 3 : S3 → HuggingFace Dataset
│
├── tts_dataset_gen/         # Package core
│   ├── s3_utils.py          # Tigris S3 helpers
│   ├── downloader.py        # yt-dlp parallel download
│   ├── segmenter.py         # Silero VAD (FLAC, 5–10s)
│   └── transcriber.py       # Whisper batch transcription (CUDA)
│
├── projects/
│   └── sidbi-ziri/
│       └── config.yaml
│
├── docs/
│   ├── strategy.png         # Vision stratégique
│   └── architecture.png     # Diagramme technique
│
└── pyproject.toml
```

---

## Créer un nouveau projet

### 1. Créer le config

```yaml
# projects/kibare/config.yaml
project:
  name: "kibare"
  language: "moore"

youtube:
  playlist_url: "https://www.youtube.com/playlist?list=..."
  max_workers: 3

segmentation:
  min_duration: 5.0
  max_duration: 10.0

transcription:
  model_id: "burkimbia/BIA-WHISPER-LARGE-SACHI_V2"
  batch_size: 8
  device: "auto"

dataset:
  hf_repo: "sawadogosalif/kibare-dataset"

bucket:
  name: "burkimbia"
  endpoint: "https://fly.storage.tigris.dev"
```

### 2. Lancer le pipeline

```bash
# Téléchargement + segmentation VAD
uv run --no-sync python chunk_pipeline.py projects/kibare/config.yaml

# Transcription (GPU recommandé)
uv run --no-sync python transcribe_pipeline.py projects/kibare/config.yaml

# Push HuggingFace
uv run --no-sync python create_dataset.py projects/kibare/config.yaml
```

---

## Modèle de transcription

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `burkimbia/BIA-WHISPER-LARGE-SACHI_V2` |
| Base | Whisper Large fine-tuné Mooré |
| Language tag | `hausa` (proxy pour Mooré) |
| Decoding | Greedy (`num_beams=1`) |
| batch_size | 8 |
| max_new_tokens | 444 |

---

## S3 (Tigris)

```
burkimbia/
└── {project_name}/
    ├── chunks/{video_id}/*.flac        # Segments audio VAD
    └── transcripts/{video_id}.jsonl    # Transcriptions Whisper
```

---

## Format JSONL

```json
{"path": "s3://burkimbia/sidbi-ziri/chunks/VIDEO_ID/seg_001.flac", "duration": 7.4, "text": "yãmb yibar...", "language": "hausa"}
```

---

## Projets actifs

| Projet | Source | Vidéos | Statut |
|--------|--------|--------|--------|
| `sidbi-ziri` | SID BI ZÉRÉ — SAVANE TV | 247 | 🔄 Transcription en cours |
| `kibare` | KIBARE | — | ⏳ À lancer |
