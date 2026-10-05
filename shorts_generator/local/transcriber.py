"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text]}.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    GEMINI_TRANSCRIBE_CHUNK_SECONDS,
    LOCAL_OUTPUT_DIR,
    LOCAL_TRANSCRIBER,
    LOCAL_WHISPER_DEVICE,
    LOCAL_WHISPER_MODEL,
)


def _transcript_cache_path(media_path: str) -> Path:
    """Return the .srt cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".srt")


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + (millis / 1000.0)


def _write_srt_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    lines = []
    for idx, segment in enumerate(transcript.get("segments", []), start=1):
        start = _format_srt_timestamp(float(segment["start"]))
        end = _format_srt_timestamp(float(segment["end"]))
        text = str(segment.get("text", "")).strip().replace("\r", "").replace("\n", " ")
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    cache_path.write_text("\n".join(lines), encoding="utf-8")
    return cache_path


def _load_srt_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}

    segments = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0] and len(lines) > 1 and "-->" in lines[1]:
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[0].split("-->", 1)]
        text = "\n".join(lines[1:]).strip()
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": text,
            }
        )

    duration = segments[-1]["end"] if segments else 0.0
    return {"duration": duration, "segments": segments}


def _resolve_device() -> str:
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Test that CUDA actually works (catches missing cuBLAS/cuDNN libs)
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def _media_duration(media_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", media_path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out or 0.0)


GEMINI_TRANSCRIBE_PROMPT = """Transcribe the {what} in this audio clip verbatim, in its original language{lang_hint}.
Split it into short segments of one sentence or phrase (roughly 2-10 seconds each).
Return ONLY a JSON array, no prose: [{{"start": <seconds>, "end": <seconds>, "text": "<spoken text>"}}]
Times are seconds from the start of this clip, as decimals. If there is no speech, return []."""


def transcribe_gemini(media_path: str, language: Optional[str] = None, lyrics: bool = False) -> Dict:
    """Transcribe with Gemini audio understanding (no Whisper model download needed).

    lyrics=True asks for sung lyrics, one line per segment, ignoring instrumentals.
    """
    from google.genai import types  # type: ignore

    from .llm import gemini_generate

    duration = _media_duration(media_path)
    chunk = max(60.0, GEMINI_TRANSCRIBE_CHUNK_SECONDS)
    lang_hint = f" (language code: {language})" if language else ""
    segments: List[Dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        offset = 0.0
        while offset < duration:
            length = min(chunk, duration - offset)
            audio_path = os.path.join(tmp, f"chunk_{int(offset)}.mp3")
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{length:.3f}",
                 "-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_path],
                check=True,
            )
            print(f"[transcribe/gemini] {offset:.0f}s-{offset + length:.0f}s of {duration:.0f}s", flush=True)
            raw = gemini_generate(
                [
                    types.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/mp3"),
                    GEMINI_TRANSCRIBE_PROMPT.format(
                        what="sung lyrics (one lyric line per segment; skip instrumental parts)" if lyrics else "speech",
                        lang_hint=lang_hint,
                    ),
                ],
                {"temperature": 0.0, "response_mime_type": "application/json", "max_output_tokens": 32768},
            )
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                items = json.loads(match.group(0)) if match else []
            for item in items if isinstance(items, list) else []:
                try:
                    start = float(item["start"]) + offset
                    end = float(item["end"]) + offset
                except (KeyError, TypeError, ValueError):
                    continue
                text = str(item.get("text", "")).strip()
                if text and end > start:
                    segments.append({"start": start, "end": min(end, duration), "text": text})
            offset += length

    segments.sort(key=lambda seg: seg["start"])
    return {"duration": duration, "segments": segments}


def transcribe_local(media_path: str, language: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path, caching the result as .srt."""
    cache_path = _transcript_cache_path(media_path)
    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_srt_cache(cache_path)
            # Treat empty cache as invalid (likely from a failed/partial run) — delete and re-transcribe
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    if LOCAL_TRANSCRIBER == "gemini":
        transcript = transcribe_gemini(media_path, language)
        print(
            f"[transcribe/gemini] {len(transcript['segments'])} segments, "
            f"{transcript['duration']:.0f}s of audio",
            flush=True,
        )
        if transcript["segments"]:
            cache_path = _write_srt_cache(media_path, transcript)
            print(f"[transcribe/gemini] wrote cache: {cache_path}", flush=True)
        return transcript

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[transcribe/local] faster-whisper model={LOCAL_WHISPER_MODEL} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
    }
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
        })

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1]["end"] if segments else 0.0)
    print(f"[transcribe/local] {len(segments)} segments, {duration:.0f}s of audio", flush=True)
    transcript = {"duration": duration, "segments": segments}
    cache_path = _write_srt_cache(media_path, transcript)
    print(f"[transcribe/local] wrote cache: {cache_path}", flush=True)
    return transcript
