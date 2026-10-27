import logging
import os
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
from pipelines.llm_inference import run_llm_on_transcript
from pipelines.preprocess import preprocess_audio
from pipelines.stt import load_whisper, transcribe

logger = logging.getLogger(__name__)

upload_bp = Blueprint("upload", __name__)


def _is_allowed_audio_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


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
    save_path = (uploads_root / safe_name).resolve()
    if save_path.parent != uploads_root:
        logger.warning("Rejected upload path outside uploads directory: %s", safe_name)
        return jsonify({"error": "Invalid filename."}), 400

    file.save(save_path)

    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    base_name = save_path.stem
    processed_path = processed_dir / (base_name + "_processed.wav")

    try:
        preprocess_audio(str(save_path), str(processed_path))
        processed_saved_to = str(processed_path)
    except Exception:
        logger.exception("Audio preprocessing failed for upload %s", safe_name)
        return jsonify({"error": "Audio preprocessing failed."}), 500
    logger.info("Preprocessed audio saved to: %s", processed_saved_to)

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

        transcripts_dir = DATA_DIR / "transcript"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = transcripts_dir / (base_name + ".txt")
        transcript_file.write_text(transcript, encoding="utf-8")
        transcript_saved_to = str(transcript_file)
        output_file = run_llm_on_transcript(transcript_file)
    except Exception:
        logger.exception("Transcription or MoM generation failed for %s", safe_name)
        return jsonify({"error": "Transcription or MoM generation failed."}), 500
    logger.info("Transcript saved to: %s", transcript_saved_to)

    if gist_text:
        gist_dir = GISTS_DIR
        gist_dir.mkdir(parents=True, exist_ok=True)
        gist_file = gist_dir / (safe_name + ".gist.txt")
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
                        logger.exception("Failed to send MoM email for %s", file_path.name)
                        error = "Failed to send email."

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
