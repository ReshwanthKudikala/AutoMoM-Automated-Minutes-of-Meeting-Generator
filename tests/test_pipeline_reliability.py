"""Pipeline reliability: stage errors, cleanup, and request isolation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipelines.llm_inference import mom_temp_path
from routes.upload import (
    _remove_file_if_exists,
    request_gist_path,
    request_processed_path,
    request_transcript_path,
    request_upload_path,
)


def _audio_file(filename: str = "meeting.wav", payload: bytes = b"x" * 32):
    return {"file": (BytesIO(payload), filename)}


REQUEST_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
REQUEST_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_request_artifact_paths_are_unique_per_request_id(tmp_path):
    uploads = tmp_path / "uploads"
    processed = tmp_path / "processed"
    transcripts = tmp_path / "transcript"
    moms = tmp_path / "mom_output"
    gists = tmp_path / "gists"
    for d in (uploads, processed, transcripts, moms, gists):
        d.mkdir()

    a_up = request_upload_path("meeting.wav", REQUEST_A, uploads)
    b_up = request_upload_path("meeting.wav", REQUEST_B, uploads)
    a_proc = request_processed_path("meeting", REQUEST_A, processed)
    b_proc = request_processed_path("meeting", REQUEST_B, processed)
    a_tr = request_transcript_path("meeting", REQUEST_A, transcripts)
    b_tr = request_transcript_path("meeting", REQUEST_B, transcripts)
    a_tmp = mom_temp_path("meeting", REQUEST_A, moms)
    b_tmp = mom_temp_path("meeting", REQUEST_B, moms)
    a_gist = request_gist_path("meeting", REQUEST_A, gists)
    b_gist = request_gist_path("meeting", REQUEST_B, gists)

    assert a_up != b_up
    assert a_proc != b_proc
    assert a_tr != b_tr
    assert a_tmp != b_tmp
    assert a_gist != b_gist
    assert a_up.name == f"meeting_{REQUEST_A}.wav"
    assert a_proc.name == f"meeting_{REQUEST_A}_processed.wav"
    assert a_tr.name == f"meeting_{REQUEST_A}.txt"
    assert a_tmp.name == f"meeting_{REQUEST_A}_mom.md.tmp"
    assert a_gist.name == f"meeting_{REQUEST_A}.gist.txt"
    assert "meeting_mom.md" not in {a_tmp.name, b_tmp.name}


def test_cleanup_only_removes_owned_request_artifacts(tmp_path):
    uploads = tmp_path / "uploads"
    processed = tmp_path / "processed"
    transcripts = tmp_path / "transcript"
    moms = tmp_path / "mom_output"
    gists = tmp_path / "gists"
    for d in (uploads, processed, transcripts, moms, gists):
        d.mkdir()

    a_up = request_upload_path("meeting.wav", REQUEST_A, uploads)
    b_up = request_upload_path("meeting.wav", REQUEST_B, uploads)
    a_proc = request_processed_path("meeting", REQUEST_A, processed)
    b_proc = request_processed_path("meeting", REQUEST_B, processed)
    a_tr = request_transcript_path("meeting", REQUEST_A, transcripts)
    b_tr = request_transcript_path("meeting", REQUEST_B, transcripts)
    a_tmp = mom_temp_path("meeting", REQUEST_A, moms)
    b_tmp = mom_temp_path("meeting", REQUEST_B, moms)
    a_gist = request_gist_path("meeting", REQUEST_A, gists)
    b_gist = request_gist_path("meeting", REQUEST_B, gists)
    final_mom = moms / "meeting_mom.md"

    for path, payload in (
        (a_up, b"a-src"),
        (b_up, b"b-src"),
        (a_proc, b"a-wav"),
        (b_proc, b"b-wav"),
        (a_tr, "a-transcript"),
        (b_tr, "b-transcript"),
        (a_tmp, "a-tmp"),
        (b_tmp, "b-tmp"),
        (a_gist, "a-gist"),
        (b_gist, "b-gist"),
        (final_mom, "# Keep me\n"),
    ):
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload, encoding="utf-8")

    # Simulate request-A failure cleanup (owned intermediates only; never uploads/gists).
    _remove_file_if_exists(a_proc)
    _remove_file_if_exists(a_tr)
    _remove_file_if_exists(a_tmp)

    assert not a_proc.exists()
    assert not a_tr.exists()
    assert not a_tmp.exists()
    assert a_up.read_bytes() == b"a-src"
    assert b_up.read_bytes() == b"b-src"
    assert a_gist.read_text(encoding="utf-8") == "a-gist"
    assert b_gist.read_text(encoding="utf-8") == "b-gist"
    assert b_proc.read_bytes() == b"b-wav"
    assert b_tr.read_text(encoding="utf-8") == "b-transcript"
    assert b_tmp.read_text(encoding="utf-8") == "b-tmp"
    assert final_mom.read_text(encoding="utf-8") == "# Keep me\n"


def test_preprocess_failure_returns_stage_error_and_cleans_owned_processed(client, app):
    other = request_processed_path(
        "meeting", REQUEST_B, app.test_paths["processed"]
    )
    other.write_bytes(b"other-request")
    captured = {}

    def failing_preprocess(_inp, outp):
        captured["processed"] = Path(outp)
        Path(outp).write_bytes(b"partial")
        raise RuntimeError("ffmpeg exploded: C:\\secret\\path")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=failing_preprocess
    ), patch("routes.upload.load_whisper") as mock_load, patch(
        "routes.upload.transcribe"
    ) as mock_stt, patch("routes.upload.run_llm_on_transcript") as mock_llm:
        response = client.post(
            "/upload",
            data=_audio_file(),
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "Audio preprocessing failed."
    assert "ffmpeg" not in body["error"].lower()
    assert "secret" not in body["error"].lower()
    assert "C:\\" not in body["error"]
    assert captured["processed"] == request_processed_path(
        "meeting", REQUEST_A, app.test_paths["processed"]
    )
    assert not captured["processed"].exists()
    assert other.read_bytes() == b"other-request"
    owned_upload = request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    )
    assert owned_upload.exists()
    mock_load.assert_not_called()
    mock_stt.assert_not_called()
    mock_llm.assert_not_called()


def test_stt_failure_returns_stage_error_and_cleans_owned_processed(client, app):
    transcripts_dir = app.test_paths["data_dir"] / "transcript"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    other_proc = request_processed_path(
        "meeting", REQUEST_B, app.test_paths["processed"]
    )
    other_tr = request_transcript_path("meeting", REQUEST_B, transcripts_dir)
    other_proc.write_bytes(b"keep")
    other_tr.write_text("keep", encoding="utf-8")
    captured = {}

    def fake_preprocess(_inp, outp):
        captured["processed"] = Path(outp)
        Path(outp).write_bytes(b"wav")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        side_effect=RuntimeError("whisper CUDA OOM at /models/secret"),
    ), patch("routes.upload.run_llm_on_transcript") as mock_llm:
        response = client.post(
            "/upload",
            data=_audio_file(),
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "Speech-to-text transcription failed."
    assert "CUDA" not in body["error"]
    assert "secret" not in body["error"]
    assert not captured["processed"].exists()
    owned_tr = request_transcript_path(
        "meeting", REQUEST_A, app.test_paths["data_dir"] / "transcript"
    )
    assert not owned_tr.exists()
    assert other_proc.read_bytes() == b"keep"
    assert other_tr.read_text(encoding="utf-8") == "keep"
    assert request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    ).exists()
    mock_llm.assert_not_called()


def test_transcript_save_failure_is_not_reported_as_stt_failure(client, app):
    captured = {}

    def fake_preprocess(_inp, outp):
        captured["processed"] = Path(outp)
        Path(outp).write_bytes(b"wav")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch.object(
        Path,
        "write_text",
        side_effect=OSError("disk full at /data/secret"),
    ), patch("routes.upload.run_llm_on_transcript") as mock_llm:
        response = client.post(
            "/upload",
            data=_audio_file(),
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "Failed to save transcript."
    assert "disk" not in body["error"].lower()
    assert "secret" not in body["error"]
    assert "Speech-to-text" not in body["error"]
    assert not captured["processed"].exists()
    owned_tr = request_transcript_path(
        "meeting", REQUEST_A, app.test_paths["data_dir"] / "transcript"
    )
    assert not owned_tr.exists()
    mock_llm.assert_not_called()


def test_llm_failure_returns_stage_error_keeps_owned_transcript_cleans_owned_temps(
    client, app
):
    transcripts_dir = app.test_paths["data_dir"] / "transcript"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    other_proc = request_processed_path(
        "meeting", REQUEST_B, app.test_paths["processed"]
    )
    other_tr = request_transcript_path("meeting", REQUEST_B, transcripts_dir)
    other_tmp = mom_temp_path("meeting", REQUEST_B, app.test_paths["mom_output"])
    other_proc.write_bytes(b"keep-proc")
    other_tr.write_text("keep-tr", encoding="utf-8")
    other_tmp.write_text("keep-tmp", encoding="utf-8")

    prior_mom = app.test_paths["mom_output"] / "meeting_mom.md"
    prior_mom.write_text("# Prior successful MoM\n", encoding="utf-8")
    owned_tmp = mom_temp_path("meeting", REQUEST_A, app.test_paths["mom_output"])
    owned_tmp.write_text("incomplete", encoding="utf-8")

    captured = {}

    def fake_preprocess(_inp, outp):
        captured["processed"] = Path(outp)
        Path(outp).write_bytes(b"wav")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch(
        "routes.upload.run_llm_on_transcript",
        side_effect=RuntimeError("NVIDIA 401 unauthorized token=sekrit"),
    ):
        response = client.post(
            "/upload",
            data=_audio_file(),
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    body = response.get_json()
    assert body["error"] == "Minutes of Meeting generation failed."
    assert "401" not in body["error"]
    assert "sekrit" not in body["error"]
    assert "NVIDIA" not in body["error"]
    assert not captured["processed"].exists()

    owned_tr = request_transcript_path("meeting", REQUEST_A, transcripts_dir)
    assert owned_tr.exists()
    assert owned_tr.read_text(encoding="utf-8")
    assert not owned_tmp.exists()

    assert other_proc.read_bytes() == b"keep-proc"
    assert other_tr.read_text(encoding="utf-8") == "keep-tr"
    assert other_tmp.read_text(encoding="utf-8") == "keep-tmp"
    assert prior_mom.read_text(encoding="utf-8") == "# Prior successful MoM\n"
    assert request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    ).exists()


def test_successful_upload_cleans_owned_processed_keeps_upload_transcript_mom(
    client, app
):
    mom_path = app.test_paths["mom_output"] / "meeting_mom.md"
    other_proc = request_processed_path(
        "meeting", REQUEST_B, app.test_paths["processed"]
    )
    other_proc.write_bytes(b"other")
    captured = {}

    def fake_preprocess(inp, outp):
        captured["source"] = Path(inp)
        captured["processed"] = Path(outp)
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **kwargs):
        captured["transcript"] = transcript_path
        assert transcript_path.exists()
        assert kwargs.get("output_stem") == "meeting"
        assert kwargs.get("request_id") == REQUEST_A
        mom_path.write_text("# Meeting\n", encoding="utf-8")
        return mom_path

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch("routes.upload.run_llm_on_transcript", side_effect=fake_llm):
        response = client.post(
            "/upload",
            data=_audio_file(payload=b"owned-bytes"),
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert "redirect" in response.get_json()
    assert not captured["processed"].exists()
    assert other_proc.read_bytes() == b"other"
    owned_upload = request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    )
    assert owned_upload.exists()
    assert owned_upload.read_bytes() == b"owned-bytes"
    assert captured["source"].resolve() == owned_upload.resolve()
    assert captured["source"].name != "meeting.wav"
    assert not (app.test_paths["uploads"] / "meeting.wav").exists()
    assert captured["transcript"].exists()
    assert captured["transcript"] == request_transcript_path(
        "meeting", REQUEST_A, app.test_paths["data_dir"] / "transcript"
    )
    assert mom_path.exists()


def test_upload_isolation_cleanup_does_not_delete_peer_intermediates(client, app):
    """Full upload failure for request A must leave request B intermediates intact."""
    transcripts_dir = app.test_paths["data_dir"] / "transcript"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    b_up = request_upload_path("meeting.wav", REQUEST_B, app.test_paths["uploads"])
    b_proc = request_processed_path("meeting", REQUEST_B, app.test_paths["processed"])
    b_tr = request_transcript_path("meeting", REQUEST_B, transcripts_dir)
    b_tmp = mom_temp_path("meeting", REQUEST_B, app.test_paths["mom_output"])
    b_gist = request_gist_path("meeting", REQUEST_B, app.test_paths["gists"])
    b_up.write_bytes(b"peer-src")
    b_proc.write_bytes(b"peer-wav")
    b_tr.write_text("peer-transcript", encoding="utf-8")
    b_tmp.write_text("peer-tmp", encoding="utf-8")
    b_gist.write_text("peer-gist", encoding="utf-8")

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"a-wav")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch(
        "routes.upload.run_llm_on_transcript",
        side_effect=RuntimeError("boom"),
    ):
        response = client.post(
            "/upload",
            data=_audio_file(),
            content_type="multipart/form-data",
        )

    assert response.status_code == 500
    assert b_up.read_bytes() == b"peer-src"
    assert b_proc.read_bytes() == b"peer-wav"
    assert b_tr.read_text(encoding="utf-8") == "peer-transcript"
    assert b_tmp.read_text(encoding="utf-8") == "peer-tmp"
    assert b_gist.read_text(encoding="utf-8") == "peer-gist"
    assert request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    ).exists()
    assert not request_processed_path(
        "meeting", REQUEST_A, app.test_paths["processed"]
    ).exists()
    assert not mom_temp_path(
        "meeting", REQUEST_A, app.test_paths["mom_output"]
    ).exists()
    # Request A transcript is retained after LLM failure.
    assert request_transcript_path(
        "meeting", REQUEST_A, transcripts_dir
    ).exists()


def test_same_original_filename_persists_distinct_source_paths(client, app):
    """Two uploads named meeting.wav must not share a single uploads path."""
    captured_sources = []
    ids = iter([REQUEST_A, REQUEST_B])

    def fake_preprocess(inp, outp):
        captured_sources.append(Path(inp))
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **kwargs):
        stem = kwargs["output_stem"]
        mom = app.test_paths["mom_output"] / f"{stem}_mom.md"
        mom.write_text("# Meeting\n", encoding="utf-8")
        return mom

    with patch(
        "routes.upload._new_request_id", side_effect=lambda: next(ids)
    ), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch("routes.upload.run_llm_on_transcript", side_effect=fake_llm):
        first = client.post(
            "/upload",
            data=_audio_file(payload=b"first-audio"),
            content_type="multipart/form-data",
        )
        second = client.post(
            "/upload",
            data=_audio_file(payload=b"second-audio"),
            content_type="multipart/form-data",
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(captured_sources) == 2
    assert captured_sources[0] != captured_sources[1]
    assert captured_sources[0].name == f"meeting_{REQUEST_A}.wav"
    assert captured_sources[1].name == f"meeting_{REQUEST_B}.wav"
    assert captured_sources[0].resolve() != (
        app.test_paths["uploads"] / "meeting.wav"
    ).resolve()

    a_path = request_upload_path("meeting.wav", REQUEST_A, app.test_paths["uploads"])
    b_path = request_upload_path("meeting.wav", REQUEST_B, app.test_paths["uploads"])
    assert a_path.read_bytes() == b"first-audio"
    assert b_path.read_bytes() == b"second-audio"
    assert not (app.test_paths["uploads"] / "meeting.wav").exists()


def test_preprocess_receives_request_specific_saved_source(client, app):
    captured = {}

    def fake_preprocess(inp, outp):
        captured["inp"] = Path(inp)
        Path(outp).write_bytes(b"wav")

    with patch("routes.upload._new_request_id", return_value=REQUEST_A), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch(
        "routes.upload.run_llm_on_transcript",
        return_value=app.test_paths["mom_output"] / "meeting_mom.md",
    ):
        (app.test_paths["mom_output"] / "meeting_mom.md").write_text(
            "# MoM\n", encoding="utf-8"
        )
        response = client.post(
            "/upload",
            data=_audio_file(payload=b"solo-bytes"),
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    expected = request_upload_path(
        "meeting.wav", REQUEST_A, app.test_paths["uploads"]
    )
    assert captured["inp"].resolve() == expected.resolve()
    assert captured["inp"].name == f"meeting_{REQUEST_A}.wav"
    assert expected.read_bytes() == b"solo-bytes"


def test_gist_paths_are_request_specific_and_isolated(client, app):
    ids = iter([REQUEST_A, REQUEST_B])
    peer_gist = request_gist_path("meeting", REQUEST_B, app.test_paths["gists"])
    peer_gist.write_text("peer-should-remain", encoding="utf-8")

    def fake_preprocess(_inp, outp):
        Path(outp).write_bytes(b"wav")

    def fake_llm(transcript_path: Path, **kwargs):
        mom = app.test_paths["mom_output"] / "meeting_mom.md"
        mom.write_text("# Meeting\n", encoding="utf-8")
        return mom

    with patch(
        "routes.upload._new_request_id", side_effect=lambda: next(ids)
    ), patch(
        "routes.upload.preprocess_audio", side_effect=fake_preprocess
    ), patch("routes.upload.load_whisper"), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
    ), patch("routes.upload.run_llm_on_transcript", side_effect=fake_llm):
        ok = client.post(
            "/upload",
            data={**_audio_file(), "gist": "notes-for-A"},
            content_type="multipart/form-data",
        )
        # Second request fails after save; must not touch peer gist or A's gist.
        with patch(
            "routes.upload.run_llm_on_transcript",
            side_effect=RuntimeError("llm down"),
        ):
            failed = client.post(
                "/upload",
                data={**_audio_file(), "gist": "notes-for-B-unused"},
                content_type="multipart/form-data",
            )

    assert ok.status_code == 200
    assert failed.status_code == 500

    a_gist = request_gist_path("meeting", REQUEST_A, app.test_paths["gists"])
    b_gist = request_gist_path("meeting", REQUEST_B, app.test_paths["gists"])
    assert a_gist != b_gist
    assert a_gist.read_text(encoding="utf-8") == "notes-for-A"
    # Failure path never writes gist; pre-seeded peer content remains.
    assert b_gist.read_text(encoding="utf-8") == "peer-should-remain"
    assert not (app.test_paths["gists"] / "meeting.wav.gist.txt").exists()
    assert not (app.test_paths["gists"] / "meeting.gist.txt").exists()


@patch("routes.upload.yagmail.SMTP")
def test_email_delivery_failure_does_not_look_like_pipeline_failure(
    mock_smtp_cls, client, app
):
    mom = app.test_paths["mom_output"] / "demo_mom.md"
    mom.write_text("# Demo\n", encoding="utf-8")

    mock_smtp = MagicMock()
    mock_smtp.send.side_effect = RuntimeError("SMTP auth failed password=sekrit")
    mock_smtp_cls.return_value = mock_smtp

    response = client.post(
        "/moms",
        data={"filename": "demo_mom.md", "emails": "a@example.com"},
    )
    assert response.status_code == 200
    assert b"Email delivery failed" in response.data
    assert b"Minutes of Meeting file was not changed" in response.data
    assert b"sekrit" not in response.data
    assert b"SMTP" not in response.data
    assert b"shared successfully" not in response.data
    assert mom.exists()
    assert mom.read_text(encoding="utf-8") == "# Demo\n"
