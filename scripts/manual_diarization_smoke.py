"""
Manual diarization integration smoke check (not collected by pytest).

Requires pyannote model access, local audio, and typically a GPU/HF token.
Usage (from repository root):

    python scripts/manual_diarization_smoke.py [path/to/audio.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.diarization import diarize


def main() -> int:
    default = Path("data/outputs/clean.wav")
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not audio_path.is_file():
        print(f"Audio not found: {audio_path}", file=sys.stderr)
        print(
            "Provide an existing audio path as the first argument.",
            file=sys.stderr,
        )
        return 1

    segments = diarize(str(audio_path))
    print(f"Total speaker segments: {len(segments)}")
    for segment in segments[:5]:
        print(segment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
