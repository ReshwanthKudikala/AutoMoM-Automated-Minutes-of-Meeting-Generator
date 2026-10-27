"""Route-level tests for secure uploads and MoM path confinement."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


def _audio_file(filename: str, payload: bytes = b"RIFF....WAVEfmt "):
    return {
        "file": (BytesIO(payload), filename),
    }


@patch("routes.upload.run_llm_on_transcript")
@patch("routes.upload.transcribe")
@patch("routes.upload.load_whisper")
@patch("routes.upload.preprocess_audio")
def test_valid_upload_sanitizes_filename_and_processes(
    mock_preprocess,
    mock_load_whisper,
    mock_transcribe,
    mock_llm,
    client,
    app,
):
    mock_transcribe.return_value = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    mom_path = app.test_paths["mom_output"] / "meeting_audio_mom.md"
    mom_path.write_text("# Meeting\n", encoding="utf-8")
    mock_llm.return_value = mom_path

    response = client.post(
        "/upload",
        data=_audio_file("My Meeting Audio.wav", b"x" * 32),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert "redirect" in body
    assert "meeting_audio_mom.md" in body["redirect"] or "highlight=" in body["redirect"]

    saved = list(app.test_paths["uploads"].iterdir())
    assert len(saved) == 1
    assert saved[0].name == "My_Meeting_Audio.wav"
    mock_preprocess.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_llm.assert_called_once()


def test_upload_missing_file(client):
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file provided."


def test_upload_empty_filename(client):
    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"abc"), "")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file selected."


def test_upload_unsupported_extension(client, app):
    response = client.post(
        "/upload",
        data=_audio_file("notes.txt", b"not audio"),
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.get_json()["error"]
    assert list(app.test_paths["uploads"].iterdir()) == []


def test_upload_traversal_filename_is_confined(client, app):
    """Traversal components are stripped; file must remain under uploads."""
    with patch("routes.upload.preprocess_audio"), patch(
        "routes.upload.load_whisper"
    ), patch(
        "routes.upload.transcribe",
        return_value=[{"start": 0.0, "end": 1.0, "text": "hi"}],
    ), patch(
        "routes.upload.run_llm_on_transcript"
    ) as mock_llm:
        mom_path = app.test_paths["mom_output"] / "evil_mom.md"
        mom_path.write_text("# MoM\n", encoding="utf-8")
        mock_llm.return_value = mom_path

        response = client.post(
            "/upload",
            data=_audio_file("../../evil.wav", b"x" * 16),
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    saved = list(app.test_paths["uploads"].iterdir())
    assert len(saved) == 1
    assert saved[0].name == "evil.wav"
    assert saved[0].resolve().parent == app.test_paths["uploads"].resolve()
    # Must not create a file outside the uploads directory.
    assert not (app.test_paths["data_dir"].parent / "evil.wav").exists()


def test_upload_rejects_extensionless_traversal_name(client, app):
    response = client.post(
        "/upload",
        data=_audio_file("../../../etc/passwd", b"x"),
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    # secure_filename strips path components; extensionless result is rejected.
    assert "Unsupported audio format" in error or error == "Invalid filename."
    assert list(app.test_paths["uploads"].iterdir()) == []


def test_upload_oversized_file(client, app):
    oversized = b"a" * (app.config["MAX_CONTENT_LENGTH"] + 100)
    response = client.post(
        "/upload",
        data=_audio_file("big.wav", oversized),
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == "Uploaded file is too large."


def test_show_mom_valid(client, app):
    mom = app.test_paths["mom_output"] / "demo_mom.md"
    mom.write_text("# Demo Meeting\n\n## Attendees\n- A\n", encoding="utf-8")

    response = client.get("/mom/demo_mom.md")
    assert response.status_code == 200
    assert b"Demo Meeting" in response.data or b"demo_mom.md" in response.data


def test_show_mom_nonexistent(client):
    response = client.get("/mom/missing_mom.md")
    assert response.status_code == 404


@pytest.mark.parametrize(
    "evil_name",
    [
        "../secrets.txt",
        "..%2Fsecrets.txt",
        "../../config.py",
        r"..\..\config.py",
    ],
)
def test_show_mom_rejects_traversal(client, app, evil_name):
    # Plant a file outside mom_output that must not be readable via traversal.
    outside = app.test_paths["data_dir"] / "secret.txt"
    outside.write_text("SECRET", encoding="utf-8")

    response = client.get(f"/mom/{evil_name}")
    assert response.status_code == 404
    assert b"SECRET" not in response.data


def test_download_mom_valid(client, app):
    mom = app.test_paths["mom_output"] / "demo_mom.md"
    mom.write_bytes(b"# Demo\n")

    response = client.get("/mom/demo_mom.md/download")
    assert response.status_code == 200
    assert response.data == b"# Demo\n"


def test_download_mom_rejects_traversal(client, app):
    outside = app.test_paths["data_dir"] / "secret.txt"
    outside.write_text("SECRET", encoding="utf-8")

    response = client.get("/mom/../secret.txt/download")
    assert response.status_code == 404
    assert b"SECRET" not in response.data


def test_download_mom_nonexistent(client):
    response = client.get("/mom/nope_mom.md/download")
    assert response.status_code == 404


@patch("routes.upload.yagmail.SMTP")
def test_email_share_valid_mom(mock_smtp_cls, client, app):
    mom = app.test_paths["mom_output"] / "demo_mom.md"
    mom.write_text("# Demo\n", encoding="utf-8")
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value = mock_smtp

    response = client.post(
        "/moms",
        data={"filename": "demo_mom.md", "emails": "a@example.com"},
    )
    assert response.status_code == 200
    assert b"shared successfully" in response.data
    mock_smtp.send.assert_called_once()


@patch("routes.upload.yagmail.SMTP")
def test_email_share_rejects_traversal(mock_smtp_cls, client, app):
    outside = app.test_paths["data_dir"] / "secret.txt"
    outside.write_text("SECRET", encoding="utf-8")

    response = client.post(
        "/moms",
        data={"filename": "../secret.txt", "emails": "a@example.com"},
    )
    assert response.status_code == 200
    assert b"File not found." in response.data
    mock_smtp_cls.assert_not_called()
    assert b"SECRET" not in response.data


@patch("routes.upload.yagmail.SMTP")
def test_email_share_nonexistent_mom(mock_smtp_cls, client):
    response = client.post(
        "/moms",
        data={"filename": "missing_mom.md", "emails": "a@example.com"},
    )
    assert response.status_code == 200
    assert b"File not found." in response.data
    mock_smtp_cls.assert_not_called()


def test_resolve_mom_file_rejects_absolute_path_outside_mom_dir(app):
    from routes.upload import resolve_mom_file

    outside = (app.test_paths["data_dir"] / "outside_secret.md").resolve()
    outside.write_text("SECRET_ABS", encoding="utf-8")

    assert resolve_mom_file(str(outside)) is None
    assert resolve_mom_file(outside.as_posix()) is None

    response = app.test_client().get(f"/mom/{outside.as_posix()}")
    assert response.status_code == 404
    assert b"SECRET_ABS" not in response.data


def test_resolve_mom_file_rejects_windows_drive_letter_path(app):
    from routes.upload import resolve_mom_file

    # Explicit drive-letter form pointing outside the MoM directory.
    drive_path = r"C:\Windows\Temp\automom_outside_mom.md"
    assert resolve_mom_file(drive_path) is None
    assert resolve_mom_file("C:/Windows/Temp/automom_outside_mom.md") is None

    response = app.test_client().post(
        "/moms",
        data={
            "filename": drive_path,
            "emails": "a@example.com",
        },
    )
    assert response.status_code == 200
    assert b"File not found." in response.data


def test_symlink_inside_mom_dir_pointing_outside_is_rejected(app):
    """Symlink escape: link in MOM_OUTPUT_DIR must not expose outside content."""
    from routes.upload import resolve_mom_file

    outside = app.test_paths["data_dir"] / "symlink_target_secret.md"
    outside.write_text("SECRET_SYMLINK", encoding="utf-8")
    link = app.test_paths["mom_output"] / "linked_escape_mom.md"

    try:
        link.symlink_to(outside.resolve())
    except OSError as exc:
        pytest.skip(
            "Cannot create symlink required for this test "
            f"(OS or permissions limitation): {exc}"
        )

    assert resolve_mom_file("linked_escape_mom.md") is None

    client = app.test_client()
    view = client.get("/mom/linked_escape_mom.md")
    assert view.status_code == 404
    assert b"SECRET_SYMLINK" not in view.data

    download = client.get("/mom/linked_escape_mom.md/download")
    assert download.status_code == 404
    assert b"SECRET_SYMLINK" not in download.data

    with patch("routes.upload.yagmail.SMTP") as mock_smtp_cls:
        email_resp = client.post(
            "/moms",
            data={
                "filename": "linked_escape_mom.md",
                "emails": "a@example.com",
            },
        )
        assert email_resp.status_code == 200
        assert b"File not found." in email_resp.data
        mock_smtp_cls.assert_not_called()
