"""Speaker diarization helpers.

Heavy dependencies (``torch``, ``pyannote.audio``) are imported only when the
diarization pipeline is loaded—not when formatting or aligning segments.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_pipeline = None


def load_diarization_pipeline():
    """Load (and cache) the pyannote pipeline. Imports torch/pyannote on first use."""
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading diarization pipeline on %s", device)

        # Gated Hugging Face model: HF_TOKEN / HUGGING_FACE_HUB_TOKEN when required.
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        load_kwargs = {}
        if token:
            # Newer huggingface_hub uses ``token=``; older pyannote accepted
            # ``use_auth_token=``. Prefer ``token`` and fall back if needed.
            load_kwargs["token"] = token

        try:
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                **load_kwargs,
            )
        except TypeError:
            if token:
                load_kwargs = {"use_auth_token": token}
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                **load_kwargs,
            )

        _pipeline.to(device)

    return _pipeline


def diarize(audio_path: str):
    """
    Returns a list of speaker segments:
    [{start, end, speaker}, ...]

    ``speaker`` values are anonymous cluster labels from pyannote
    (for example ``SPEAKER_00``), not verified personal identities.
    """
    pipeline = load_diarization_pipeline()
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })

    return segments


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_transcript(
    transcript_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:
    """
    Align Whisper-style transcript segments with diarization turns.

    Each transcript segment is labelled with the diarization speaker that has
    the greatest time overlap. Segments with no positive overlap keep
    ``speaker`` as ``None`` (callers must not invent a label).
    """
    labelled = []
    for seg in transcript_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()

        best_speaker = None
        best_overlap = 0.0
        for turn in diarization_segments:
            overlap = _overlap_seconds(
                start,
                end,
                float(turn.get("start", 0.0)),
                float(turn.get("end", 0.0)),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.get("speaker")

        labelled.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": best_speaker if best_overlap > 0.0 else None,
        })
    return labelled


def format_transcript_segments(segments: list[dict]) -> str:
    """
    Format transcript segments for LLM / file persistence.

    When a ``speaker`` key is present and non-empty, lines are:
    ``[start - end] SPEAKER: text``
    Otherwise lines remain:
    ``[start - end] text``
    """
    lines = []
    for seg in segments:
        start = f"{float(seg.get('start', 0.0)):.2f}"
        end = f"{float(seg.get('end', 0.0)):.2f}"
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")
        if speaker:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
        else:
            lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)
