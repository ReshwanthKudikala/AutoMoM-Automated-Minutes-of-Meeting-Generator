import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
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


# --------------------------------------------------
# PIPELINE ENTRYPOINT
# --------------------------------------------------
def run_llm_on_transcript(transcript_path: Path):

    transcript_text = transcript_path.read_text(encoding="utf-8")

    print("Calling NVIDIA LLaMA...")

    mom_output = generate_mom_for_transcript(transcript_text)

    output_dir = Path("data/mom_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (transcript_path.stem + "_mom.md")
    output_file.write_text(mom_output, encoding="utf-8")

    print("MoM saved to:", output_file)

    return output_file
