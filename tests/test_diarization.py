"""Unit tests for speaker diarization helpers (no model downloads / CUDA)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pipelines.diarization as diarization_mod
from pipelines.diarization import (
    assign_speakers_to_transcript,
    diarize,
    format_transcript_segments,
)


def test_diarize_returns_normalized_segments_from_pipeline():
    turn_a = SimpleNamespace(start=0.0, end=1.5)
    turn_b = SimpleNamespace(start=1.5, end=3.0)

    mock_result = MagicMock()
    mock_result.itertracks.return_value = [
        (turn_a, None, "SPEAKER_00"),
        (turn_b, None, "SPEAKER_01"),
    ]

    mock_pipeline = MagicMock(return_value=mock_result)

    with patch.object(
        diarization_mod,
        "load_diarization_pipeline",
        return_value=mock_pipeline,
    ):
        segments = diarize("unused-audio.wav")

    mock_pipeline.assert_called_once_with("unused-audio.wav")
    mock_result.itertracks.assert_called_once_with(yield_label=True)
    assert segments == [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
    ]


def test_diarize_returns_empty_list_when_pipeline_has_no_tracks():
    mock_result = MagicMock()
    mock_result.itertracks.return_value = []
    mock_pipeline = MagicMock(return_value=mock_result)

    with patch.object(
        diarization_mod,
        "load_diarization_pipeline",
        return_value=mock_pipeline,
    ):
        assert diarize("empty.wav") == []


def test_assign_speakers_uses_max_overlap_and_leaves_gaps_unlabelled():
    transcript = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
        {"start": 5.0, "end": 6.0, "text": "alone"},
    ]
    diarization = [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 1.2, "end": 2.5, "speaker": "SPEAKER_01"},
    ]

    labelled = assign_speakers_to_transcript(transcript, diarization)
    assert labelled[0]["speaker"] == "SPEAKER_00"
    assert labelled[1]["speaker"] == "SPEAKER_01"
    assert labelled[2]["speaker"] is None
    assert labelled[2]["text"] == "alone"


def test_assign_speakers_empty_diarization_keeps_speakers_none():
    transcript = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    labelled = assign_speakers_to_transcript(transcript, [])
    assert labelled == [
        {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None},
    ]


def test_format_transcript_segments_with_and_without_speakers():
    plain = format_transcript_segments(
        [{"start": 0.0, "end": 1.0, "text": "hello"}]
    )
    assert plain == "[0.00 - 1.00] hello"

    labelled = format_transcript_segments(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "speaker": "SPEAKER_00",
            },
            {"start": 1.0, "end": 2.0, "text": "gap", "speaker": None},
        ]
    )
    assert labelled == (
        "[0.00 - 1.00] SPEAKER_00: hello\n"
        "[1.00 - 2.00] gap"
    )


def test_upload_with_diarization_writes_speaker_labelled_transcript(client, app):
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"
    captured = {}

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **_kwargs):
        captured["transcript"] = transcript_path.read_text(encoding="utf-8")
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        return mom_path

    with patch("routes.upload.ENABLE_DIARIZATION", True), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "there"},
        ],
    ), patch(
        "pipelines.diarization.diarize",
        return_value=[
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ],
    ) as mock_diarize, patch(
        "routes.upload.run_llm_on_transcript", side_effect=fake_llm
    ):
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"x" * 32), "meeting.wav")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    mock_diarize.assert_called_once()
    assert "SPEAKER_00: hello" in captured["transcript"]
    assert "SPEAKER_01: there" in captured["transcript"]


def test_upload_diarization_failure_falls_back_to_unlabeled_transcript(client, app):
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"
    captured = {}

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **_kwargs):
        captured["transcript"] = transcript_path.read_text(encoding="utf-8")
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        return mom_path

    with patch("routes.upload.ENABLE_DIARIZATION", True), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch(
        "pipelines.diarization.diarize",
        side_effect=RuntimeError("HF gated model token=sekrit"),
    ), patch("routes.upload.run_llm_on_transcript", side_effect=fake_llm):
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"x" * 32), "meeting.wav")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert "redirect" in response.get_json()
    assert captured["transcript"] == "[0.00 - 1.00] hello"
    assert "SPEAKER_" not in captured["transcript"]
    assert "sekrit" not in captured["transcript"]


def test_upload_empty_diarization_keeps_unlabeled_transcript(client, app):
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"
    captured = {}

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **_kwargs):
        captured["transcript"] = transcript_path.read_text(encoding="utf-8")
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        return mom_path

    with patch("routes.upload.ENABLE_DIARIZATION", True), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch("pipelines.diarization.diarize", return_value=[]), patch(
        "routes.upload.run_llm_on_transcript", side_effect=fake_llm
    ):
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"x" * 32), "meeting.wav")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert captured["transcript"] == "[0.00 - 1.00] hello"


def test_upload_skips_diarization_when_disabled(client, app):
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    with patch("routes.upload.ENABLE_DIARIZATION", False), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch("pipelines.diarization.diarize") as mock_diarize, patch(
        "routes.upload.run_llm_on_transcript", return_value=mom_path
    ):
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"x" * 32), "meeting.wav")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    mock_diarize.assert_not_called()


def test_upload_missing_diarization_deps_falls_back_when_enabled(client, app):
    """ImportError for torch/pyannote must not fail the upload when diarization is on."""
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"
    captured = {}

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **_kwargs):
        captured["transcript"] = transcript_path.read_text(encoding="utf-8")
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        return mom_path

    with patch("routes.upload.ENABLE_DIARIZATION", True), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch(
        "pipelines.diarization.diarize",
        side_effect=ModuleNotFoundError("No module named 'torch'"),
    ), patch("routes.upload.run_llm_on_transcript", side_effect=fake_llm):
        response = client.post(
            "/upload",
            data={"file": (BytesIO(b"x" * 32), "meeting.wav")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert captured["transcript"] == "[0.00 - 1.00] hello"


def test_diarization_module_imports_without_torch_or_pyannote():
    """
    Regression: helpers must load without torch/pyannote.

    Uses an isolated subprocess so tests/conftest.py stubs cannot hide an
    eager import of the heavy ML packages.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = r"""
import sys
from unittest.mock import MagicMock

# Stub only non-ML app deps; deliberately do NOT stub torch/pyannote.
for name in ("yagmail", "openai", "dotenv", "whisper"):
    sys.modules[name] = MagicMock()
sys.modules["openai"].OpenAI = MagicMock()
sys.modules["dotenv"].load_dotenv = MagicMock()

class _BlockHeavy:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in {"torch", "pyannote"}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, _BlockHeavy())

for key in list(sys.modules):
    if key == "torch" or key.startswith("torch.") or key == "pyannote" or key.startswith("pyannote."):
        del sys.modules[key]
    if key == "pipelines.diarization" or key.startswith("pipelines.diarization."):
        del sys.modules[key]
    if key == "routes.upload" or key.startswith("routes.upload"):
        del sys.modules[key]

from pipelines.diarization import format_transcript_segments, assign_speakers_to_transcript

assert format_transcript_segments([{"start": 0.0, "end": 1.0, "text": "hi"}]) == "[0.00 - 1.00] hi"
assert assign_speakers_to_transcript(
    [{"start": 0.0, "end": 1.0, "text": "hi"}],
    [],
)[0]["speaker"] is None

# Upload route import must succeed with diarization disabled and ML deps absent.
import config
assert config.ENABLE_DIARIZATION is False
import routes.upload  # noqa: F401
print("LAZY_IMPORT_OK")
"""
    env = os.environ.copy()
    env["ENABLE_DIARIZATION"] = "0"
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "LAZY_IMPORT_OK" in result.stdout
