import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to ffmpeg executable (used by pipelines.preprocess)
ffmpeg_path = Path("tools/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe")

# Centralized data directories
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
GISTS_DIR = DATA_DIR / "gists"
MOM_OUTPUT_DIR = DATA_DIR / "mom_output"

# Ensure folders exist when imported
for p in (DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, GISTS_DIR, MOM_OUTPUT_DIR):
	p.mkdir(parents=True, exist_ok=True)

# Upload limits and allowed audio extensions (FFmpeg can decode these for preprocess).
# Override max size with env MAX_CONTENT_LENGTH (bytes). Default: 50 MiB.
DEFAULT_MAX_CONTENT_LENGTH = 50 * 1024 * 1024


def parse_max_content_length(raw_value, default=DEFAULT_MAX_CONTENT_LENGTH):
	"""
	Parse MAX_CONTENT_LENGTH from an environment-style value.

	Absent or blank values use ``default``. Invalid (non-integer / negative)
	values also fall back to ``default`` and emit a warning that does not
	include the raw environment contents.
	"""
	if raw_value is None:
		return default
	if isinstance(raw_value, str) and not raw_value.strip():
		return default
	try:
		value = int(str(raw_value).strip())
	except (TypeError, ValueError):
		logger.warning(
			"Invalid MAX_CONTENT_LENGTH environment value; "
			"using default of %s bytes.",
			default,
		)
		return default
	if value < 0:
		logger.warning(
			"Negative MAX_CONTENT_LENGTH is not allowed; "
			"using default of %s bytes.",
			default,
		)
		return default
	return value


MAX_CONTENT_LENGTH = parse_max_content_length(os.getenv("MAX_CONTENT_LENGTH"))
ALLOWED_AUDIO_EXTENSIONS = frozenset({
	".wav",
	".mp3",
	".m4a",
	".flac",
	".ogg",
	".aac",
	".webm",
	".wma",
	".opus",
})

# Flask debug mode (env FLASK_DEBUG). Default: disabled.
_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})


def parse_bool_env(raw_value, default=False, env_name="VALUE"):
	"""
	Parse a boolean environment-style value.

	Absent or blank values use ``default``. Recognized truthy values:
	1/true/yes/on. Recognized falsy values: 0/false/no/off (case-insensitive).
	Invalid values fall back to ``default`` and emit a warning that does not
	include the raw environment contents.
	"""
	if raw_value is None:
		return default
	if isinstance(raw_value, str) and not raw_value.strip():
		return default
	normalized = str(raw_value).strip().lower()
	if normalized in _BOOL_TRUE:
		return True
	if normalized in _BOOL_FALSE:
		return False
	logger.warning(
		"Invalid %s environment value; using default of %s.",
		env_name,
		default,
	)
	return default


FLASK_DEBUG = parse_bool_env(
	os.getenv("FLASK_DEBUG"),
	default=False,
	env_name="FLASK_DEBUG",
)

# Email configuration
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
