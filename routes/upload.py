import logging
import os
import uuid
from pathlib import Path

import yagmail
from flask import (
    Blueprint,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    DATA_DIR,
    GISTS_DIR,
    MOM_OUTPUT_DIR,
    PROCESSED_DIR,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_SERVER,
    SMTP_USER,
    UPLOADS_DIR,
)
from pipelines.llm_inference import mom_temp_path, run_llm_on_transcript
from pipelines.preprocess import preprocess_audio
from pipelines.stt import load_whisper, transcribe

logger = logging.getLogger(__name__)

upload_bp = Blueprint("upload", __name__)


def _is_allowed_audio_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


def _new_request_id() -> str:
    return uuid.uuid4().hex


def request_upload_path(
    safe_name: str,
    request_id: str,
    uploads_dir: Path | None = None,
) -> Path:
    """
    Per-request uploaded source path: ``{stem}_{request_id}{suffix}``.

    Keeps the original extension for FFmpeg while isolating concurrent uploads
    that share the same sanitized filename.
    """
    directory = uploads_dir if uploads_dir is not None else UPLOADS_DIR
    original = Path(safe_name)
    return directory / f"{original.stem}_{request_id}{original.suffix}"


def request_gist_path(
    base_name: str,
    request_id: str,
    gists_dir: Path | None = None,
) -> Path:
    """Per-request gist path: ``{stem}_{request_id}.gist.txt``."""
    directory = gists_dir if gists_dir is not None else GISTS_DIR
    return directory / f"{base_name}_{request_id}.gist.txt"


def request_processed_path(
    base_name: str,
    request_id: str,
    processed_dir: Path | None = None,
) -> Path:
    """Per-request processed WAV path (never shared across uploads)."""
    directory = processed_dir if processed_dir is not None else PROCESSED_DIR
    return directory / f"{base_name}_{request_id}_processed.wav"


def request_transcript_path(
    base_name: str,
    request_id: str,
    transcripts_dir: Path | None = None,
) -> Path:
    """Per-request transcript path (never shared across uploads)."""
    directory = (
        transcripts_dir
        if transcripts_dir is not None
        else (DATA_DIR / "transcript")
    )
    return directory / f"{base_name}_{request_id}.txt"


def _remove_file_if_exists(path: Path | None) -> None:
    """Best-effort removal of a single temporary artifact (never raises)."""
    if path is None:
        return
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        logger.warning("Failed to remove temporary artifact %s", path.name)


def resolve_mom_file(filename: str):
    """
    Resolve a MoM filename strictly under MOM_OUTPUT_DIR.

    Returns a Path to an existing file inside the MoM directory, or None if the
    name is invalid, traverses outside the directory, or does not exist.
    Prefers a sibling .md file when a .txt path is requested.
    """
    if not filename or not isinstance(filename, str):
        return None
    if "\x00" in filename:
        return None

    normalized = filename.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None

    # Absolute paths (POSIX or Windows drive) are rejected.
    if Path(filename).is_absolute() or (len(filename) >= 2 and filename[1] == ":"):
        return None

    candidate_name = parts[-1]
    if candidate_name in (".", ".."):
        return None

    mom_dir = MOM_OUTPUT_DIR.resolve()
    file_path = (mom_dir / candidate_name).resolve()

    if file_path.suffix.lower() == ".txt":
        md_candidate = file_path.with_suffix(".md")
        if md_candidate.is_file():
            file_path = md_candidate.resolve()

    try:
        file_path.relative_to(mom_dir)
    except ValueError:
        return None

    if not file_path.is_file():
        return None

    return file_path


@upload_bp.route("/health", methods=["GET"])
def health():
    return {"status": "AutoMoM running"}


@upload_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    gist_text = request.form.get("gist", "")

    if file is None or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename."}), 400

    if not _is_allowed_audio_filename(safe_name):
        allowed = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        return jsonify({
            "error": f"Unsupported audio format. Allowed extensions: {allowed}."
        }), 400

    uploads_root = UPLOADS_DIR.resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)

    # Original sanitized stem drives final MoM naming; request id isolates storage.
    request_id = _new_request_id()
    base_name = Path(safe_name).stem
    save_path = request_upload_path(safe_name, request_id, uploads_root).resolve()
    if save_path.parent != uploads_root:
        logger.warning(
            "Rejected upload path outside uploads directory: %s (request %s)",
            safe_name,
            request_id,
        )
        return jsonify({"error": "Invalid filename."}), 400

    file.save(save_path)

    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = request_processed_path(base_name, request_id, processed_dir)

    transcripts_dir = DATA_DIR / "transcript"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = request_transcript_path(base_name, request_id, transcripts_dir)

    mom_tmp = mom_temp_path(base_name, request_id, MOM_OUTPUT_DIR)
    gist_file = request_gist_path(base_name, request_id, GISTS_DIR)

    # --- Stage: preprocessing ---
    try:
        # Always use the request-owned save_path; never rebuild from safe_name alone.
        preprocess_audio(str(save_path), str(processed_path))
    except Exception:
        logger.exception(
            "Audio preprocessing failed for upload %s (request %s)",
            safe_name,
            request_id,
        )
        _remove_file_if_exists(processed_path)
        return jsonify({"error": "Audio preprocessing failed."}), 500
    logger.info("Preprocessed audio ready for %s (request %s)", safe_name, request_id)

    # --- Stage: speech-to-text (model only) ---
    try:
        load_whisper()
        segments = transcribe(str(processed_path))
        lines = []
        for s in segments:
            start = f"{s.get('start', 0.0):.2f}"
            end = f"{s.get('end', 0.0):.2f}"
            text = s.get("text", "").strip()
            lines.append(f"[{start} - {end}] {text}")
        transcript = "\n".join(lines)
    except Exception:
        logger.exception(
            "Speech-to-text failed for upload %s (request %s)",
            safe_name,
            request_id,
        )
        _remove_file_if_exists(processed_path)
        return jsonify({"error": "Speech-to-text transcription failed."}), 500

    # --- Stage: persist transcript ---
    try:
        transcript_file.write_text(transcript, encoding="utf-8")
    except Exception:
        logger.exception(
            "Failed to save transcript for upload %s (request %s)",
            safe_name,
            request_id,
        )
        _remove_file_if_exists(processed_path)
        _remove_file_if_exists(transcript_file)
        return jsonify({"error": "Failed to save transcript."}), 500
    logger.info("Transcript saved for %s (request %s)", safe_name, request_id)

    # --- Stage: LLM MoM generation ---
    try:
        output_file = run_llm_on_transcript(
            transcript_file,
            output_stem=base_name,
            request_id=request_id,
            temp_file=mom_tmp,
        )
    except Exception:
        logger.exception(
            "MoM generation failed for upload %s (request %s)",
            safe_name,
            request_id,
        )
        _remove_file_if_exists(processed_path)
        # Only this request's MoM temp; never a finished MoM or another request's tmp.
        _remove_file_if_exists(mom_tmp)
        return jsonify({"error": "Minutes of Meeting generation failed."}), 500

    # Intermediate processed audio is no longer needed after a successful run.
    _remove_file_if_exists(processed_path)

    if gist_text:
        GISTS_DIR.mkdir(parents=True, exist_ok=True)
        gist_file.write_text(gist_text, encoding="utf-8")

    return jsonify({
        "redirect": url_for(
            "upload.list_moms",
            highlight=os.path.basename(str(output_file)),
        )
    })


@upload_bp.route("/moms", methods=["GET", "POST"])
def list_moms():
    mom_dir = MOM_OUTPUT_DIR
    mom_dir.mkdir(parents=True, exist_ok=True)
    mom_files = sorted([p.name for p in mom_dir.glob("*_mom.md") if p.is_file()])
    if not mom_files:
        mom_files = sorted([p.name for p in mom_dir.glob("*_mom.txt") if p.is_file()])

    highlight = request.args.get("highlight")
    error = None
    success = None

    if request.method == "POST":
        filename = request.form.get("filename")
        emails = request.form.get("emails", "").strip()

        if not filename or not emails:
            error = "Please provide both filename and email addresses."
        else:
            email_list = [e.strip() for e in emails.split(",") if e.strip()]
            if not email_list:
                error = "No valid email addresses found."
            else:
                file_path = resolve_mom_file(filename)
                if file_path is None:
                    error = "File not found."
                else:
                    content = file_path.read_text(encoding="utf-8")
                    try:
                        yag = yagmail.SMTP(
                            SMTP_USER,
                            SMTP_PASSWORD,
                            host=SMTP_SERVER,
                            port=SMTP_PORT,
                            smtp_starttls=True,
                            smtp_ssl=False,
                        )
                        subject = f"Minutes of Meeting: {file_path.name}"
                        body = (
                            "Attached is the Minutes of Meeting generated from the audio.\n\n"
                            f"{content[:500]}..."
                        )
                        yag.send(
                            to=email_list,
                            subject=subject,
                            contents=body,
                            attachments=str(file_path),
                        )
                        success = f"MoM '{file_path.name}' shared successfully!"
                    except Exception:
                        logger.exception(
                            "Email delivery failed for MoM %s",
                            file_path.name,
                        )
                        error = (
                            "Email delivery failed. "
                            "The Minutes of Meeting file was not changed."
                        )

    return render_template(
        "moms.html",
        mom_files=mom_files,
        highlight=highlight,
        error=error,
        success=success,
    )


@upload_bp.route("/mom/<path:filename>", methods=["GET"])
def show_mom(filename):
    file_path = resolve_mom_file(filename)
    if file_path is None:
        abort(404)

    content = file_path.read_text(encoding="utf-8")
    return render_template(
        "mom_detail.html",
        filename=file_path.name,
        content=content,
    )


@upload_bp.route("/mom/<path:filename>/download", methods=["GET"])
def download_mom(filename):
    file_path = resolve_mom_file(filename)
    if file_path is None:
        abort(404)

    mom_dir = MOM_OUTPUT_DIR.resolve()
    return send_from_directory(str(mom_dir), file_path.name, as_attachment=True)
