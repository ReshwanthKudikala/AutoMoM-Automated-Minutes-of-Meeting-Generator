from pathlib import Path

# Path to ffmpeg executable (used by pipelines.preprocess)
ffmpeg_path = Path("tools/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe")

# Centralized data directories
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
GISTS_DIR = DATA_DIR / "gists"

# Ensure folders exist when imported
for p in (DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, GISTS_DIR):
	p.mkdir(parents=True, exist_ok=True)

# Email configuration
import os
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))