"""Unit tests for speaker diarization helpers (no model downloads / CUDA)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pipelines.diarization as diarization_mod
from pipelines.diarization import diarize


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
