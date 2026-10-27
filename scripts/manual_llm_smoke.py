"""
Manual LLM integration smoke check (not collected by pytest).

Requires a real NVIDIA API token and a local transcript file.
Usage (from repository root):

    python scripts/manual_llm_smoke.py [path/to/transcript.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.llm_inference import run_llm_on_transcript


def main() -> int:
    default = Path("data/transcript/EarningsCall.txt")
    transcript_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not transcript_file.is_file():
        print(f"Transcript not found: {transcript_file}", file=sys.stderr)
        print(
            "Provide an existing transcript path as the first argument.",
            file=sys.stderr,
        )
        return 1

    output = run_llm_on_transcript(transcript_file)
    print(f"MoM written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
