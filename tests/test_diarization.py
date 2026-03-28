from pipelines.diarization import diarize

audio_path = "../data/outputs/clean.wav"
segments = diarize(audio_path)

print(f"Total speaker segments: {len(segments)}")
for s in segments[:5]:
    print(s)