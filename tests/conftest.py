"""
Test fixtures for AutoMoM route tests.

Heavy optional dependencies are stubbed before the Flask app is imported so
tests can run without Whisper, NVIDIA API access, GPU, or SMTP.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Stub heavy / optional third-party modules before application imports.
_STUB_MODULES = [
    "whisper",
    "yagmail",
    "openai",
    "dotenv",
    "torch",
    "pyannote",
    "pyannote.audio",
]

for _name in _STUB_MODULES:
    sys.modules.setdefault(_name, MagicMock())

# Provide a minimal OpenAI client surface used at import time.
sys.modules["openai"].OpenAI = MagicMock()
sys.modules["dotenv"].load_dotenv = MagicMock()


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import config
    import routes.upload as upload_mod
    from main import create_app

    uploads = tmp_path / "uploads"
    processed = tmp_path / "processed"
    gists = tmp_path / "gists"
    mom_output = tmp_path / "mom_output"
    data_dir = tmp_path
    for path in (uploads, processed, gists, mom_output):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "GISTS_DIR", gists)
    monkeypatch.setattr(config, "MOM_OUTPUT_DIR", mom_output)
    monkeypatch.setattr(config, "MAX_CONTENT_LENGTH", 1024)

    monkeypatch.setattr(upload_mod, "DATA_DIR", data_dir)
    monkeypatch.setattr(upload_mod, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(upload_mod, "PROCESSED_DIR", processed)
    monkeypatch.setattr(upload_mod, "GISTS_DIR", gists)
    monkeypatch.setattr(upload_mod, "MOM_OUTPUT_DIR", mom_output)
    # Keep diarization off by default in route tests (opt-in per test).
    monkeypatch.setattr(config, "ENABLE_DIARIZATION", False)
    monkeypatch.setattr(upload_mod, "ENABLE_DIARIZATION", False)

    application = create_app()
    application.config["TESTING"] = True
    application.config["MAX_CONTENT_LENGTH"] = 1024

    application.test_paths = {
        "uploads": uploads,
        "processed": processed,
        "gists": gists,
        "mom_output": mom_output,
        "data_dir": data_dir,
    }
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
