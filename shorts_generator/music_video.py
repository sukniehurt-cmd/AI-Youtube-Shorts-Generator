"""Song → vertical music-video short.

Takes an audio file (mp3/wav/...), transcribes the lyrics with Gemini, lets
Gemini pick the most catchy section (usually the chorus) and describe a few
visual scenes, then renders a 9:16 video with the song's lyrics as captions and
the audio section underneath. Each scene is, depending on `visuals`:

- "veo":      a moving AI video clip (Veo, up to 8 s per scene)
- "images":   an AI image with a slow camera move
- "gradient": an animated colour gradient in the song's palette

Each step falls back to the next one if it is unavailable (e.g. a free-tier
Gemini key has no Veo or image quota).
"""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    GEMINI_IMAGE_MODEL,
    LOCAL_OUTPUT_DIR,
    MUSIC_VISUALS,
    VEO_MODEL,
    require_gemini_key,
)
from .local.llm import call_gemini_llm
from .local.transcriber import transcribe_gemini

WIDTH, HEIGHT, FPS = 1080, 1920, 30

PLAN_PROMPT = """You are a music-video director making a vertical short (TikTok / Reels / YouTube Shorts) for a song.

Song duration: {duration:.1f}s. Timestamped lyrics:
{lyrics}

1. Pick the single most catchy, replayable section (usually the chorus or the hook), between {min_len:.0f} and {max_len:.0f} seconds long.
   Start on a line start, end on a line end. If there are no lyrics, pick by position (e.g. the first chorus is often around 25-35% into the song).
2. Describe exactly {num_scenes} visual scenes that follow the lyrics of that section in order, as image-generation prompts:
   cinematic, vertical 9:16 composition, consistent style and characters across scenes, no text or letters in the image.
   Each scene is also used as a prompt for an AI video clip of up to 8 seconds, so describe the subject, setting
   and the motion in it (what moves, how the camera moves).

Return ONLY JSON:
{{"start": <seconds>, "end": <seconds>, "title": "<short title>", "mood": "<a few words>",
  "palette": ["#rrggbb", "#rrggbb", "#rrggbb"],
  "scenes": [{{"prompt": "<image prompt>"}}]}}"""


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out or 0.0)


def _parse_json(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(match.group(0)) if match else {}


def plan_music_short(
    transcript: Dict, min_len: float, max_len: float, num_scenes: int
) -> Dict:
    duration = transcript["duration"]
    lyrics = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in transcript["segments"]
    ) or "(no lyrics detected — instrumental)"
    plan = _parse_json(call_gemini_llm(PLAN_PROMPT.format(
        duration=duration, lyrics=lyrics, min_len=min_len, max_len=max_len, num_scenes=num_scenes,
    )))

    # Keep the section inside the song and inside the requested length.
    max_len = min(max_len, duration)
    start = max(0.0, float(plan.get("start", 0.0)))
    end = float(plan.get("end", start + max_len))
    if end - start > max_len:
        end = start + max_len
    if end - start < min(min_len, duration):
        end = start + min(min_len, duration)
    if end > duration:
        start, end = max(0.0, duration - (end - start)), duration
    plan["start"], plan["end"] = start, end

    scenes = [s for s in plan.get("scenes", []) if isinstance(s, dict) and s.get("prompt")]
    plan["scenes"] = scenes[:num_scenes] or [{"prompt": f"abstract cinematic visuals, {plan.get('mood', 'energetic')}"}]
    palette = [c for c in plan.get("palette", []) if re.fullmatch(r"#[0-9a-fA-F]{6}", str(c))]
    plan["palette"] = palette or ["#1a1a40", "#7a0bc0", "#fa58b6"]
    return plan


class _ImageGenerator:
    """Gemini image generation that disables itself after the first hard failure."""

    def __init__(self) -> None:
        self.enabled = True
        self.reason = ""

    def generate(self, prompt: str, out_path: str) -> bool:
        if not self.enabled:
            return False
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=require_gemini_key())
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=prompt,
                config={"response_modalities": ["IMAGE"], "image_config": {"aspect_ratio": "9:16"}},
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    Path(out_path).write_bytes(part.inline_data.data)
                    return True
            raise RuntimeError("no image in response")
        except Exception as e:  # quota, model access, safety block, ...
            self.enabled = False
            self.reason = str(e).splitlines()[0][:200]
            print(f"[music/images] image generation unavailable, using gradients: {self.reason}", flush=True)
            return False


class _VideoGenerator:
    """Veo text-to-video that disables itself after the first hard failure."""

    def __init__(self) -> None:
        self.enabled = True
        self.reason = ""

    def generate(self, prompt: str, seconds: float, out_path: str, timeout: float = 600) -> bool:
        if not self.enabled:
            return False
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=require_gemini_key())
            operation = client.models.generate_videos(
                model=VEO_MODEL,
                prompt=f"{prompt}. Smooth cinematic motion, music video style, no text, no captions.",
                config={
                    "aspect_ratio": "9:16",
                    "duration_seconds": 8 if seconds > 6 else (6 if seconds > 4 else 4),
                    "negative_prompt": "text, letters, subtitles, watermark",
                },
            )
            deadline = time.time() + timeout
            while not operation.done:
                if time.time() > deadline:
                    raise TimeoutError(f"Veo did not finish within {timeout:.0f}s")
                time.sleep(10)
                operation = client.operations.get(operation)
            if operation.error:
                raise RuntimeError(f"Veo error: {operation.error}")
            videos = operation.response.generated_videos if operation.response else None
            if not videos:
                raise RuntimeError("Veo returned no video (possibly blocked by safety filters)")
            client.files.download(file=videos[0].video)
            videos[0].video.save(out_path)
            return True
        except Exception as e:  # quota, model access, safety block, timeout, ...
            self.enabled = False
            self.reason = str(e).splitlines()[0][:200]
            print(f"[music/veo] video generation unavailable, falling back: {self.reason}", flush=True)
            return False


def _render_video_scene(video: str, seconds: float, out: str) -> None:
    # Fit to 9:16, drop Veo's own audio, and hold the last frame if the clip is shorter than the scene.
    _run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", video,
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},"
        f"fps={FPS},tpad=stop_mode=clone:stop_duration={seconds:.3f},format=yuv420p",
        "-an", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast", out,
    ])


def _render_image_scene(image: str, seconds: float, index: int, out: str) -> None:
    frames = max(1, int(round(seconds * FPS)))
    # Alternate slow zoom-in / zoom-out so cuts feel like camera moves.
    zoom = f"1+0.15*on/{frames}" if index % 2 == 0 else f"1.15-0.15*on/{frames}"
    _run([
        "ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", image,
        "-vf",
        f"scale={WIDTH * 2}:{HEIGHT * 2}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH * 2}:{HEIGHT * 2},"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        "format=yuv420p",
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", out,
    ])


def _render_gradient_scene(palette: List[str], seconds: float, index: int, out: str) -> None:
    colors = palette[index % len(palette):] + palette[: index % len(palette)]
    color_args = ":".join(f"c{i}=0x{c.lstrip('#')}" for i, c in enumerate(colors[:8]))
    _run([
        "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
        "-i", f"gradients=s={WIDTH}x{HEIGHT}:r={FPS}:d={seconds:.3f}:speed=0.02:n={len(colors[:8])}:{color_args}",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", out,
    ])


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _write_lyrics_ass(segments: List[Dict], start: float, end: float, path: str) -> int:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,DejaVu Sans,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,3,2,80,80,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for seg in segments:
        s, e = max(seg["start"], start), min(seg["end"], end)
        if e <= s:
            continue
        text = str(seg["text"]).replace("\n", " ").replace("{", "(").replace("}", ")")
        lines.append(
            f"Dialogue: 0,{_ass_time(s - start)},{_ass_time(e - start)},Lyrics,,0,0,0,,{{\\fad(150,150)}}{text}"
        )
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def generate_music_short(
    audio_path: str,
    out_path: Optional[str] = None,
    min_len: float = 15.0,
    max_len: float = 40.0,
    num_scenes: int = 5,
    language: Optional[str] = None,
    captions: bool = True,
    visuals: Optional[str] = None,
) -> Dict:
    """Render one vertical music-video short from a song file. Returns the plan + output path."""
    audio_path = str(Path(audio_path).expanduser())
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    out_dir = Path(LOCAL_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or str(out_dir / f"{Path(audio_path).stem}_short.mp4")

    print(f"[music] transcribing lyrics: {audio_path}", flush=True)
    transcript = transcribe_gemini(audio_path, language=language, lyrics=True)
    if not transcript["duration"]:
        transcript["duration"] = _duration(audio_path)
    print(f"[music] {len(transcript['segments'])} lyric lines, {transcript['duration']:.0f}s", flush=True)

    plan = plan_music_short(transcript, min_len, max_len, num_scenes)
    start, end = plan["start"], plan["end"]
    length = end - start
    print(f"[music] section {start:.1f}s → {end:.1f}s: {plan.get('title')} ({plan.get('mood')})", flush=True)

    visuals = (visuals or MUSIC_VISUALS).lower()
    if visuals not in ("veo", "images", "gradient"):
        raise ValueError(f"visuals must be veo, images or gradient, not {visuals!r}")
    videos = _VideoGenerator()
    videos.enabled = visuals == "veo"
    images = _ImageGenerator()
    images.enabled = visuals in ("veo", "images")
    scene_len = length / len(plan["scenes"])
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, scene in enumerate(plan["scenes"]):
            clip = os.path.join(tmp, f"scene_{i:02d}.mp4")
            image = os.path.join(tmp, f"scene_{i:02d}.png")
            print(f"[music] scene {i + 1}/{len(plan['scenes'])}: {scene['prompt'][:80]}", flush=True)
            veo_clip = os.path.join(tmp, f"scene_{i:02d}_veo.mp4")
            if videos.generate(scene["prompt"], scene_len, veo_clip):
                _render_video_scene(veo_clip, scene_len, clip)
                scene["visual"] = "veo"
            elif images.generate(scene["prompt"], image):
                _render_image_scene(image, scene_len, i, clip)
                scene["visual"] = "image"
            else:
                _render_gradient_scene(plan["palette"], scene_len, i, clip)
                scene["visual"] = "gradient"
            clips.append(clip)

        concat_list = os.path.join(tmp, "scenes.txt")
        Path(concat_list).write_text("".join(f"file '{c}'\n" for c in clips))
        visuals = os.path.join(tmp, "visuals.mp4")
        _run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", visuals])

        vf = "format=yuv420p"
        if captions:
            ass_path = os.path.join(tmp, "lyrics.ass")
            if _write_lyrics_ass(transcript["segments"], start, end, ass_path):
                vf = f"ass={ass_path},format=yuv420p"

        fade_out = max(0.0, length - 1.5)
        _run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", visuals,
            "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", audio_path,
            "-vf", vf,
            "-af", f"afade=t=in:d=0.5,afade=t=out:st={fade_out:.3f}:d=1.5",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path,
        ])

    print(f"[music] wrote {out_path}", flush=True)
    return {
        "audio": audio_path,
        "output": out_path,
        "plan": plan,
        "visuals": visuals,
        "video_error": videos.reason,
        "image_error": images.reason,
        "transcript": transcript,
    }
