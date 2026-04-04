from flask import Blueprint, request, jsonify, render_template, abort, send_from_directory, redirect, url_for
from pathlib import Path
from pipelines.llm_inference import run_llm_on_transcript
from pipelines.preprocess import preprocess_audio
from config import DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, GISTS_DIR, SMTP_USER, SMTP_PASSWORD, SMTP_SERVER, SMTP_PORT
from pipelines.stt import transcribe, load_whisper
import yagmail
import os

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/health", methods=["GET"])
def health():
    return {"status": "AutoMoM running"}


@upload_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files["file"]
    gist_text = request.form.get("gist", "")

    save_dir = UPLOADS_DIR
    save_path = save_dir / file.filename
    file.save(save_path)

    # Preprocess the audio file
    processed_dir = PROCESSED_DIR
    # Create output filename with .wav extension
    base_name = save_path.stem
    processed_path = processed_dir / (base_name + "_processed.wav")
    
    try:
        preprocess_audio(str(save_path), str(processed_path))
        processed_saved_to = str(processed_path)
    except Exception as e:
        return jsonify({"error": f"preprocessing failed: {str(e)}"}), 500
    print(f"Preprocessed audio saved to: {processed_saved_to}")
    # Run speech-to-text on the preprocessed audio using the loaded Whisper model
    try:
        # ensure model is loaded (load_whisper is idempotent)
        load_whisper()
        segments = transcribe(str(processed_path))
        # join segment texts into a single transcript
        lines = []
        for s in segments:
            start = f"{s.get('start', 0.0):.2f}"
            end = f"{s.get('end', 0.0):.2f}"
            text = s.get("text", "").strip()
            lines.append(f"[{start} - {end}] {text}")

        transcript = "\n".join(lines)

        # Save transcript to data/transcript/<base_name>.txt
        transcripts_dir = DATA_DIR / "transcript"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = transcripts_dir / (base_name + ".txt")
        transcript_file.write_text(transcript, encoding="utf-8")
        transcript_saved_to = str(transcript_file)
        output_file = run_llm_on_transcript(transcript_file)
    except Exception as e:
        print(f"Transcription failed: {str(e)}")        
        return jsonify({"error": f"transcription failed: {str(e)}"}), 500
    print(f"Transcript saved to: {transcript_saved_to}")
    gist_saved_to = None
    if gist_text:
        gist_dir = GISTS_DIR
        gist_file = gist_dir / (file.filename + ".gist.txt")
        gist_file.write_text(gist_text, encoding="utf-8")
        gist_saved_to = str(gist_file)

    return jsonify({
        "redirect": url_for('upload.list_moms', highlight=os.path.basename(str(output_file)))
    })


@upload_bp.route("/moms", methods=["GET", "POST"])
def list_moms():
    mom_dir = Path("data/mom_output")
    mom_dir.mkdir(parents=True, exist_ok=True)
    mom_files = sorted([p.name for p in mom_dir.glob("*_mom.md") if p.is_file()])
    # fallback to text files for existing older MOMs
    if not mom_files:
        mom_files = sorted([p.name for p in mom_dir.glob("*_mom.txt") if p.is_file()])

    highlight = request.args.get('highlight')
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
                # Ensure we send the markdown file when available
                file_path = mom_dir / filename
                if file_path.suffix == ".txt":
                    md_candidate = file_path.with_suffix('.md')
                    if md_candidate.exists():
                        file_path = md_candidate
                        filename = md_candidate.name

                if not file_path.exists():
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
                            smtp_ssl=False
                        )
                        subject = f"Minutes of Meeting: {filename}"
                        body = f"Attached is the Minutes of Meeting generated from the audio.\n\n{content[:500]}..."  # Truncate for body
                        yag.send(to=email_list, subject=subject, contents=body, attachments=str(file_path))
                        success = f"MoM '{filename}' shared successfully!"
                    except Exception as e:
                        error = f"Failed to send email: {str(e)}"

    return render_template("moms.html", mom_files=mom_files, highlight=highlight, error=error, success=success)


@upload_bp.route("/mom/<path:filename>", methods=["GET"])
def show_mom(filename):
    mom_dir = Path("data/mom_output").resolve()
    file_path = (mom_dir / filename).resolve()

    # Prefer Markdown output file
    if file_path.suffix == ".txt":
        md_file = file_path.with_suffix('.md')
        if md_file.exists():
            file_path = md_file
            filename = md_file.name

    if not file_path.exists() or not file_path.is_file() or mom_dir not in file_path.parents:
        abort(404)

    content = file_path.read_text(encoding="utf-8")
    return render_template("mom_detail.html", filename=filename, content=content)

@upload_bp.route("/mom/<path:filename>/download", methods=["GET"])
def download_mom(filename):
    mom_dir = Path("data/mom_output")
    return send_from_directory(str(mom_dir), filename, as_attachment=True)

