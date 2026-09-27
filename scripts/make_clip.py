"""Cut a frame-accurate dev clip out of a long match broadcast.

The raw broadcasts are 4-5 hours long and mostly not live play, so every pass is developed
and benchmarked on short clips of delivery sequences cut with this script.

    python scripts/make_clip.py --video "Cricket match/replay-final-sa-vs-ind.mp4" \
        --start 00:50:00 --duration 60 --name final_0050

Writes data/<name>.mp4 by default. Re-encodes rather than stream-copying so the clip
starts on the exact requested frame instead of the nearest keyframe.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_time(value: str) -> float:
    """Seconds from 'SS', 'MM:SS' or 'HH:MM:SS' (fractions allowed)."""
    parts = value.split(":")
    if not 1 <= len(parts) <= 3:
        raise argparse.ArgumentTypeError(f"bad time {value!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad time {value!r}") from None
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"time must be >= 0, got {value!r}")
    return seconds


def build_command(video: Path, start: float, duration: float, out: Path) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out),
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=parse_time, required=True, help="SS, MM:SS or HH:MM:SS")
    span = parser.add_mutually_exclusive_group(required=True)
    span.add_argument("--duration", type=parse_time)
    span.add_argument("--end", type=parse_time)
    parser.add_argument("--name", help="clip name, becomes the video_id (default: <stem>_<start seconds>)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg not found on PATH")
    if not args.video.is_file():
        parser.error(f"no such video: {args.video}")
    duration = args.duration if args.duration is not None else args.end - args.start
    if duration <= 0:
        parser.error("clip must have positive duration")

    name = args.name or f"{args.video.stem}_{int(args.start)}"
    out = args.out_dir / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(args.video, args.start, duration, out), check=True)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
