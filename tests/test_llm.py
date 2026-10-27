"""Unit tests for MoM LLM helpers (no NVIDIA API calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import pipelines.llm_inference as llm_mod
from pipelines.llm_inference import (
    build_user_prompt,
    mom_output_path,
    mom_temp_path,
    run_llm_on_transcript,
)


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
    mom_dir = tmp_path / "mom_output"
    mom_dir.mkdir()
    monkeypatch.setattr(llm_mod, "MOM_OUTPUT_DIR", mom_dir)

    transcript = tmp_path / "sample_meeting_reqabc.txt"
    transcript.write_text("Bob: Action item assigned to Carol.", encoding="utf-8")

    fake_mom = "# Sample Meeting\n\n## Attendees\n- Bob\n"
    request_id = "reqabc"

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value=fake_mom,
    ) as mock_generate:
        output_file = run_llm_on_transcript(
            transcript,
            output_stem="sample_meeting",
            request_id=request_id,
        )

    mock_generate.assert_called_once_with("Bob: Action item assigned to Carol.")
    expected = mom_output_path("sample_meeting", mom_dir)
    assert output_file.resolve() == expected.resolve()
    assert expected.is_file()
    assert expected.read_text(encoding="utf-8") == fake_mom
    assert not mom_temp_path("sample_meeting", request_id, mom_dir).exists()
    assert list(mom_dir.glob("*.tmp")) == []


def test_run_llm_on_transcript_rejects_empty_output(tmp_path, monkeypatch):
    mom_dir = tmp_path / "mom_output"
    mom_dir.mkdir()
    monkeypatch.setattr(llm_mod, "MOM_OUTPUT_DIR", mom_dir)

    transcript = tmp_path / "empty_result.txt"
    transcript.write_text("unused", encoding="utf-8")

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value="",
    ):
        with pytest.raises(ValueError, match="empty MoM"):
            run_llm_on_transcript(transcript, output_stem="empty_result")

    assert list(mom_dir.iterdir()) == []


def test_run_llm_on_transcript_rejects_whitespace_only_output(tmp_path, monkeypatch):
    mom_dir = tmp_path / "mom_output"
    mom_dir.mkdir()
    monkeypatch.setattr(llm_mod, "MOM_OUTPUT_DIR", mom_dir)

    prior = mom_output_path("meeting", mom_dir)
    prior_bytes = b"# Existing MoM\n"
    prior.write_bytes(prior_bytes)

    transcript = tmp_path / "meeting_ws.txt"
    transcript.write_text("unused", encoding="utf-8")

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value="  \n\t  ",
    ):
        with pytest.raises(ValueError, match="empty MoM"):
            run_llm_on_transcript(
                transcript,
                output_stem="meeting",
                request_id="wsrid",
            )

    assert prior.read_bytes() == prior_bytes
    assert not mom_temp_path("meeting", "wsrid", mom_dir).exists()
    assert list(mom_dir.glob("*.tmp")) == []


def test_run_llm_on_transcript_removes_temp_on_write_failure(tmp_path, monkeypatch):
    mom_dir = tmp_path / "mom_output"
    mom_dir.mkdir()
    monkeypatch.setattr(llm_mod, "MOM_OUTPUT_DIR", mom_dir)

    transcript = tmp_path / "meeting.txt"
    transcript.write_text("hello", encoding="utf-8")
    request_id = "writefail"
    temp_path = mom_temp_path("meeting", request_id, mom_dir)

    real_write_text = Path.write_text

    def flaky_write(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            real_write_text(self, *args, **kwargs)
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value="# MoM\n",
    ), patch.object(Path, "write_text", flaky_write):
        with pytest.raises(OSError, match="disk full"):
            run_llm_on_transcript(
                transcript,
                output_stem="meeting",
                request_id=request_id,
                temp_file=temp_path,
            )

    assert not mom_output_path("meeting", mom_dir).exists()
    assert not temp_path.exists()


def test_run_llm_on_transcript_replace_failure_preserves_existing_mom(
    tmp_path, monkeypatch
):
    mom_dir = tmp_path / "mom_output"
    mom_dir.mkdir()
    monkeypatch.setattr(llm_mod, "MOM_OUTPUT_DIR", mom_dir)

    output_file = mom_output_path("meeting", mom_dir)
    prior_bytes = b"# Prior successful MoM\n## Attendees\n- Alice\n"
    output_file.write_bytes(prior_bytes)

    request_id = "replacefail"
    temp_file = mom_temp_path("meeting", request_id, mom_dir)

    transcript = tmp_path / "meeting_replacefail.txt"
    transcript.write_text("hello", encoding="utf-8")

    def boom_replace(self, target):
        raise OSError("replace failed")

    with patch(
        "pipelines.llm_inference.generate_mom_for_transcript",
        return_value="# New MoM that must not land\n",
    ), patch.object(Path, "replace", boom_replace):
        with pytest.raises(OSError, match="replace failed"):
            run_llm_on_transcript(
                transcript,
                output_stem="meeting",
                request_id=request_id,
                output_file=output_file,
                temp_file=temp_file,
            )

    assert output_file.read_bytes() == prior_bytes
    assert not temp_file.exists()
