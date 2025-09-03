### Project Overview

- Purpose: Convert Tamil audio to actionable insights (intent/sentiment/emotion) with per-stop and per-day aggregates.
- Main entrypoint: `src/main.py`
- Quick start:
  - Place MP3s in `audio/`
  - Ensure FFmpeg installed; create venv and `pip install -r requirements.txt`
  - Run: `python src/main.py --config config.yaml`
- Configuration: see `config.yaml` (paths, provider, thresholds)
- Outputs: JSON files in `data/outputs/`
- Providers:
  - Default Sarvam STT (`SARVAM_API_KEY` required)
  - Whisper fallback (offline; slower)
- Tests: `pytest -q`


