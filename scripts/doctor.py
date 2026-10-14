"""Is this machine ready to run CricSense? Checks everything the pipeline needs, says what is missing.

    python scripts/doctor.py

Never prints a key's value - only whether it is set.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {  # file: what needs it
    "models/ball_best.pt": "ball - finds deliveries",
    "models/ball_ft.pt": "ball - tracks them (optional: the first model is used alone without it)",
    "models/bat_best.pt": "fuse - bat contact",
    "models/yolov8m.pt": "players",
    "models/yolo11x-pose.pt": "pose",
    "models/tts/kokoro-v1.0.int8.onnx": "voice (local speech)",
    "models/tts/voices-v1.0.bin": "voice (local speech)",
}
OPTIONAL_MODELS = {"models/ball_ft.pt"}
# Ultralytics downloads these by name on first use; `make models` fetches them (and Kokoro) now
AUTO_DOWNLOAD = {"models/yolov8m.pt", "models/yolo11x-pose.pt"}
DOWNLOADABLE = AUTO_DOWNLOAD | {"models/tts/kokoro-v1.0.int8.onnx", "models/tts/voices-v1.0.bin"}
PACKAGES = ("cv2", "numpy", "torch", "ultralytics", "pytesseract", "pydantic", "yaml", "sqlalchemy", "fastapi", "kokoro_onnx")


def env_keys() -> dict[str, bool]:
    keys = {k: bool(os.environ.get(k)) for k in ("LIGHTNING_API_KEY", "GEMINI_API_KEY")}
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in keys and value.strip():
                keys[name.strip()] = True
    return keys


def main() -> int:
    problems = 0

    def row(ok: bool, what: str, fix: str = "", required: bool = True) -> None:
        nonlocal problems
        mark = "ok  " if ok else ("MISS" if required else "warn")
        problems += (not ok) and required
        print(f"  [{mark}] {what}" + ("" if ok or not fix else f"\n         -> {fix}"))

    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print("programs")
    for prog, fix in (("ffmpeg", "install ffmpeg (sudo pacman -S ffmpeg)"), ("tesseract", "install tesseract (sudo pacman -S tesseract tesseract-data-eng)")):
        row(shutil.which(prog) is not None, prog, fix)
    print("python packages")
    for pkg in PACKAGES:
        try:
            importlib.import_module(pkg)
            row(True, pkg)
        except Exception:
            row(False, pkg, "pip install -r requirements.txt  (in the conda env 'ball')")
    print("models")
    for path, why in MODELS.items():
        here = (ROOT / path).is_file()
        if path in AUTO_DOWNLOAD and not here:
            row(False, f"{path}  ({why}) - downloads by itself on first use", "or now: make models", required=False)
        elif path in DOWNLOADABLE:
            row(here, f"{path}  ({why})", "make models")
        else:  # trained for this project: not downloadable
            row(here, f"{path}  ({why})", "trained for this project - copy it into models/ (README: Install)", path not in OPTIONAL_MODELS)
    print("configs")
    for path in ("configs/match.yaml", "configs/scoreboard.yaml", "configs/scoreboard_digits.npz"):
        row((ROOT / path).is_file(), path, "see docs/RUNBOOK.md: a new match / a new broadcaster")
    print("GPU")
    try:
        import torch

        gpu = torch.cuda.is_available()
        row(gpu, torch.cuda.get_device_name(0) if gpu else "CUDA GPU", "runs on CPU too, much slower", required=False)
    except Exception:
        row(False, "torch", "pip install -r requirements.txt")
    print("API keys (optional: without one, commentary uses templates)")
    for k, ok in env_keys().items():
        row(ok, f"{k} {'set' if ok else 'not set'}", "add it to .env (never commit .env)", required=False)
    print("\nREADY" if problems == 0 else f"\n{problems} problem(s) - fix the MISS lines above")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
