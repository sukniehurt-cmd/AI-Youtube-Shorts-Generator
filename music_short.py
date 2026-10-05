"""CLI: turn a song (mp3/wav/...) into a vertical music-video short.

Usage:
    python music_short.py song.mp3 --min-len 15 --max-len 45 --scenes 5
Needs GEMINI_API_KEY in .env (image scenes need a paid Gemini plan; otherwise
the scenes fall back to animated colour gradients).
"""
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator.music_video import generate_music_short


def main() -> int:
    parser = argparse.ArgumentParser(description="Song → vertical music-video short")
    parser.add_argument("audio", help="Path to the song (mp3, wav, m4a, ...)")
    parser.add_argument("-o", "--output", default=None, help="Output mp4 (default: output/<song>_short.mp4)")
    parser.add_argument("--min-len", type=float, default=15.0, help="Shortest section in seconds (default 15)")
    parser.add_argument("--max-len", type=float, default=45.0, help="Longest section in seconds (default 45)")
    parser.add_argument("--scenes", type=int, default=5, help="Number of visual scenes (default 5)")
    parser.add_argument("--language", default=None, help="Lyrics language code, e.g. 'pl' (default: auto)")
    parser.add_argument("--no-captions", action="store_true", help="Do not burn lyrics into the video")
    parser.add_argument("--output-json", default=None, help="Write the plan + lyrics to this JSON file")
    args = parser.parse_args()

    try:
        result = generate_music_short(
            args.audio,
            out_path=args.output,
            min_len=args.min_len,
            max_len=args.max_len,
            num_scenes=args.scenes,
            language=args.language,
            captions=not args.no_captions,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    plan = result["plan"]
    print("\n" + "=" * 72)
    print(f"Short:    {result['output']}")
    print(f"Section:  {plan['start']:.1f}s → {plan['end']:.1f}s  ({plan.get('title')}, {plan.get('mood')})")
    print(f"Scenes:   {sum(1 for s in plan['scenes'] if s.get('image'))}/{len(plan['scenes'])} AI images")
    if result["image_error"]:
        print(f"Images:   fell back to gradients ({result['image_error']})")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
