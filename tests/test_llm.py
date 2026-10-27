"""Unit tests for MoM LLM helpers (no NVIDIA API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pipelines.llm_inference import build_user_prompt, run_llm_on_transcript


def test_build_user_prompt_includes_transcript_and_required_sections():
    transcript = "Alice: We will ship next week."
    prompt = build_user_prompt(transcript)

    assert "Alice: We will ship next week." in prompt
    assert "## Attendees" in prompt
    assert "## Agenda Summary" in prompt
    assert "## Key Discussions" in prompt
    assert "## Decisions Made" in prompt
    assert "## Action Items" in prompt
    assert "## Next Steps" in prompt


def test_run_llm_on_transcript_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        run_llm_on_transcript(missing)


def test_run_llm_on_transcript_writes_mom_using_mocked_generator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    transcript = tmp_path / "sample_meeting.txt"
    transcript.write_text("Bob: Action item assigned to Carol.", encoding="utf-8")

    fake_mom = "# Sample Meeting\n\n## Attendees\n- Bob\n"

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value=fake_mom,
    ) as mock_generate:
        output_file = run_llm_on_transcript(transcript)

    mock_generate.assert_called_once_with("Bob: Action item assigned to Carol.")
    expected = tmp_path / "data" / "mom_output" / "sample_meeting_mom.md"
    assert output_file.resolve() == expected.resolve()
    assert expected.is_file()
    assert expected.read_text(encoding="utf-8") == fake_mom
