import sys
import json
import os
import glob
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python worker_transcribe.py <job_id> [model_size] [language] [beam_size]", file=sys.stderr)
        sys.exit(1)

    job_id = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "base"
    language = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("auto", "", "None") else None
    beam_size = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1

    audio_files = []
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac", "*.aac"):
        audio_files.extend(glob.glob(ext))
    
    if not audio_files:
        print("ERROR: No audio file found in current directory!", file=sys.stderr)
        sys.exit(1)

    audio_path = audio_files[0]
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Loaded audio: {audio_path} ({file_size_mb:.2f} MB)")
    print(f"Initializing Faster-Whisper '{model_size}' on 4 CPU threads (int8)...")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

    kwargs = {
        "beam_size": beam_size,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": dict(
            threshold=0.25,
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        ),
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": "Transcribe spoken dialogue, lyrics, emotional reactions, and speech.",
    }
    if language:
        kwargs["language"] = language

    print("Transcribing audio with VAD filter...")
    segments_gen, info = model.transcribe(audio_path, **kwargs)

    out = []
    texts = []
    for seg in segments_gen:
        clean_text = seg.text.strip()
        if not clean_text:
            continue
        words = [
            {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
            for w in (getattr(seg, "words", None) or [])
        ]
        out.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": clean_text,
            "words": words
        })
        texts.append(clean_text)

    # Fallback: if VAD filtered everything out, retry without VAD (essential for music videos / songs / soft speech)
    if not out:
        print("VAD returned 0 segments; retrying transcription with vad_filter=False (music/speech fallback)...")
        kwargs_no_vad = dict(kwargs)
        kwargs_no_vad["vad_filter"] = False
        kwargs_no_vad.pop("vad_parameters", None)
        segments_gen, info = model.transcribe(audio_path, **kwargs_no_vad)
        for seg in segments_gen:
            clean_text = seg.text.strip()
            if not clean_text:
                continue
            words = [
                {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in (getattr(seg, "words", None) or [])
            ]
            out.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": clean_text,
                "words": words
            })
            texts.append(clean_text)

    result = {
        "job_id": job_id,
        "status": "completed",
        "language": getattr(info, "language", "en"),
        "language_probability": round(getattr(info, "language_probability", 1.0), 3),
        "duration": round(getattr(info, "duration", 0.0), 2),
        "text": " ".join(texts),
        "segments": out
    }

    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"SUCCESS: Transcribed {len(out)} segments, {len(result['text'])} characters.")

if __name__ == "__main__":
    main()
