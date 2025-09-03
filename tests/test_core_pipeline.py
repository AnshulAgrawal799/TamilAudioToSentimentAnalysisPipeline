import json
from pathlib import Path
import pytest
import sys


# Ensure src on path for Windows runners
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from main import main as pipeline_main  # type: ignore


def test_minimal_pipeline_runs(tmp_path, monkeypatch):
    cfg_path = Path('config.yaml').resolve()
    assert cfg_path.exists()

    # Prepare temp output and temp_dir
    out_dir = tmp_path / 'outputs'
    tmp_dir = tmp_path / 'tmp'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Reuse sample WAVs if present; else skip to avoid long runs
    sample_wavs = list(Path('data/tmp').glob('*.wav'))
    if not sample_wavs:
        pytest.skip('No sample WAVs available to run pipeline quickly')

    # Monkeypatch argv
    import sys as _sys
    _sys.argv = [
        'main.py',
        '--config', str(cfg_path),
        '--skip-audio-conversion'
    ]

    pipeline_main()

    segments = Path('data/outputs/segments.json')
    assert segments.exists(), 'segments.json not created'
    data = json.loads(segments.read_text(encoding='utf-8'))
    assert isinstance(data, list) and len(data) >= 1


