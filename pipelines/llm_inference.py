import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import MOM_OUTPUT_DIR

load_dotenv()
logger = logging.getLogger(__name__)

# --------------------------------------------------
# NVIDIA OpenAI-compatible client
# --------------------------------------------------
nvidia_key = os.getenv("nvidia_token")
client = OpenAI(
    api_key=nvidia_key,
    base_url="https://integrate.api.nvidia.com/v1"
)


# --------------------------------------------------
# Prompt Builder
# --------------------------------------------------
def build_user_prompt(transcript_text: str) -> str:
    return f"""
Generate professional Minutes of Meeting (MoM) in Markdown format.

- Output only Markdown, no extra explanatory text.
- The top-level heading (#) should be the meeting title (not "Meeting Title").
- Do not include a placeholder section header such as "## Meeting Title".
- Use section headers in this order:
  - ## Attendees
  - ## Agenda Summary
  - ## Key Discussions
  - ## Decisions Made
  - ## Action Items
  - ## Next Steps
- Use bullet points (-) for list items.
- Use **bold** to emphasize owners or deadlines inside action items.

Example format:
# FinTech Plus SYNC's Second Quarter 2023 Earnings Call Minutes

## Attendees
- Alice, Bob

## Agenda Summary
- ... etc.

Transcript:
{transcript_text}
"""


# --------------------------------------------------
# LLM Call (NVIDIA Inference)
# --------------------------------------------------
def generate_mom_for_transcript(transcript_text: str):

    user_content = build_user_prompt(transcript_text)

    completion = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional corporate secretary tasked with "
                    "generating formal Minutes of Meeting (MoM)."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        temperature=0.0,
        top_p=0.3,
        max_tokens=4096,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        stream=True,
    )

    # Stream aggregation
    full_output = ""

    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            full_output += chunk.choices[0].delta.content

    return full_output.strip()


def mom_output_path(output_stem: str, output_dir: Path | None = None) -> Path:
    """Final MoM path: ``<stem>_mom.md`` under the MoM output directory."""
    directory = output_dir if output_dir is not None else MOM_OUTPUT_DIR
    return directory / f"{output_stem}_mom.md"


def mom_temp_path(
    output_stem: str,
    request_id: str,
    output_dir: Path | None = None,
) -> Path:
    """
    Request-owned MoM temp path: ``<stem>_<request_id>_mom.md.tmp``.

    Distinct from the final ``<stem>_mom.md`` so concurrent uploads sharing a
    stem cannot clobber each other's incomplete writes.
    """
    directory = output_dir if output_dir is not None else MOM_OUTPUT_DIR
    return directory / f"{output_stem}_{request_id}_mom.md.tmp"


# --------------------------------------------------
# PIPELINE ENTRYPOINT
# --------------------------------------------------
def run_llm_on_transcript(
    transcript_path: Path,
    *,
    output_stem: str | None = None,
    request_id: str | None = None,
    output_file: Path | None = None,
    temp_file: Path | None = None,
):
    """
    Generate a MoM from a transcript file and write it atomically.

    ``output_stem`` controls the final filename (``<stem>_mom.md``). When the
    transcript path includes a per-request id, callers must pass the original
    upload stem so the public MoM name stays stable.

    ``request_id`` (or an explicit ``temp_file``) makes the intermediate
    ``.tmp`` unique to this invocation so concurrent writers do not share it.
    """

    transcript_text = transcript_path.read_text(encoding="utf-8")

    logger.info("Calling NVIDIA LLaMA for MoM generation (%s)", transcript_path.name)

    mom_output = generate_mom_for_transcript(transcript_text)
    # Reject empty / whitespace-only without rewriting legitimate Markdown.
    if mom_output is None or not str(mom_output).strip():
        raise ValueError("LLM returned empty MoM output")

    output_dir = MOM_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem if output_stem is not None else transcript_path.stem
    rid = request_id if request_id is not None else uuid.uuid4().hex

    if output_file is None:
        output_file = mom_output_path(stem, output_dir)
    if temp_file is None:
        temp_file = mom_temp_path(stem, rid, output_dir)

    try:
        temp_file.write_text(mom_output, encoding="utf-8")
        temp_file.replace(output_file)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                logger.warning(
                    "Failed to remove incomplete MoM temp file %s",
                    temp_file.name,
                )
        raise

    logger.info("MoM saved to %s", output_file.name)
    return output_file
