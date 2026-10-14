"""Build the demo video from configs/demo.yaml: title cards and clips' overlays, joined.

    python scripts/make_demo.py                       # configs/demo.yaml
    python scripts/make_demo.py --config my_demo.yaml

Each clip must have been run with voice first (cricsense run --video data/<clip>.mp4 --voice),
so outputs/<clip>/overlay.mp4 exists. Cards are rendered as stills, made into short videos with
silent audio, and everything is re-encoded to 1920x1080, 25 fps, stereo, and joined.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from cricsense.render.draw import chip, paste

ROOT = Path(__file__).resolve().parents[1]
COLOURS = {"white": (255, 255, 255), "grey": (190, 190, 190), "pink": (230, 120, 255)}
W, H = 1920, 1080


def card_image(lines: list, path: Path) -> None:
    img = np.zeros((H, W, 4), np.uint8)
    img[..., :3] = (24, 22, 20)
    img[..., 3] = 255
    y = H // 2 - 45 * len(lines) // 2
    for text, size, bold, colour in lines:
        c = chip(text, int(size), COLOURS[colour], None, bool(bold), (0, 0))
        paste(img, c, W // 2 - c.shape[1] // 2, y)
        y += c.shape[0] + 26
    cv2.imwrite(str(path), img[..., :3])


def card_video(png: Path, seconds: float, out: Path) -> None:
    frames = int(seconds * 25)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", str(seconds), "-i", str(png),
        "-f", "lavfi", "-t", str(seconds), "-i", "anullsrc=r=48000:cl=stereo",
        "-vf", f"fps=25,format=yuv420p,fade=in:0:12,fade=out:{frames - 12}:12",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out),
    ], check=True)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "demo.yaml")
    args = ap.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        ap.error("ffmpeg not found on PATH")

    cfg = yaml.safe_load(args.config.read_text())
    out = ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, section in enumerate(cfg["sections"]):
            if "card" in section:
                png, mp4 = Path(tmp) / f"card{k}.png", Path(tmp) / f"card{k}.mp4"
                card_image(section["card"], png)
                card_video(png, float(cfg.get("card_seconds", 4)), mp4)
                parts.append(mp4)
            else:
                overlay = ROOT / "outputs" / section["clip"] / "overlay.mp4"
                if not overlay.is_file():
                    ap.error(f"{overlay} not found: run  cricsense run --video data/{section['clip']}.mp4 --voice")
                parts.append(overlay)
        n = len(parts)
        chains = "".join(f"[{i}:v]scale={W}:{H},fps=25,format=yuv420p,setsar=1[v{i}];[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}];" for i in range(n))
        joins = "".join(f"[v{i}][a{i}]" for i in range(n))
        cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        for p in parts:
            cmd += ["-i", str(p)]
        cmd += ["-filter_complex", f"{chains}{joins}concat=n={n}:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
        print(f"joining {n} parts into {out} (several minutes) ...", flush=True)
        subprocess.run(cmd, check=True)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
