from pathlib import Path
from pipelines.llm_inference import run_llm_on_transcript

transcript_file = Path("data/transcript/EarningsCall.txt")

run_llm_on_transcript(transcript_file)
