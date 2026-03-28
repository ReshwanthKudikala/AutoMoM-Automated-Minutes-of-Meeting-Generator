from pyannote.audio import Pipeline
import torch

_pipeline = None

def load_diarization_pipeline():
    global _pipeline
    if _pipeline is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading diarization pipeline on {device}...")
        
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization"
        )

        # Move to GPU if available
        _pipeline.to(device)

    return _pipeline


def diarize(audio_path: str):
    """
    Returns a list of speaker segments:
    [{start, end, speaker}, ...]
    """
    pipeline = load_diarization_pipeline()
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })

    return segments