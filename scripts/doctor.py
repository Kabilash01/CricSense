"""Is this machine ready to run CricSense? Checks everything the pipeline needs, says what is missing.

    python scripts/doctor.py
    python scripts/doctor.py --llm     # also make one tiny call to each commentary model: does the key work?

Never prints a key's value - only whether it is set.
"""

from __future__ import annotations

import importlib
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


def model_configs() -> list[tuple[str, object]]:
    """The commentary models configs/commentary.yaml uses: (what for, LLMConfig)."""
    import yaml

    from cricsense.commentary.llm import LLMConfig

    raw = yaml.safe_load((ROOT / "configs" / "commentary.yaml").read_text()) or {}
    out = []
    for part, why in (("llm", "commentary lines"), ("vlm", "shot reading")):
        c = raw.get(part) or {}
        if c.get("model") and (part != "vlm" or c.get("enabled")):
            out.append((why, LLMConfig(provider=c.get("provider", "anthropic"), model=c["model"], base_url=c.get("base_url"), key_env=c.get("key_env"),
                                           timeout_s=float(c.get("timeout_s", LLMConfig.timeout_s)))))
    return out


def key_name(cfg) -> str:
    from cricsense.commentary.llm import DEFAULT_KEY_ENV

    return cfg.key_env or DEFAULT_KEY_ENV.get(cfg.provider, "?")


def live_check() -> int:
    """One tiny call to each commentary model (the shot reader with a grey image): 0 if all answer."""
    import cv2
    import numpy as np
    import requests

    from cricsense.commentary.llm import api_key, complete

    failed = 0
    for why, cfg in model_configs():
        label = f"{why}: {cfg.provider} {cfg.model}"
        key = api_key(cfg, ROOT)
        if not key:
            print(f"  [MISS] {label} - no {key_name(cfg)} in .env")
            failed += 1
            continue
        images = [cv2.imencode(".jpg", np.full((64, 64, 3), 90, np.uint8))[1].tobytes()] if why == "shot reading" else None
        try:
            reply = complete(cfg, key, "Answer only with the JSON asked for.", 'Return {"ok": true}', images)
            print(f"  [ok  ] {label} - answered {reply.strip()[:40]!r}")
        except requests.HTTPError as e:  # the API's own message says what is wrong (never contains the key)
            print(f"  [FAIL] {label} - HTTP {e.response.status_code}: {e.response.text.strip()[:300]}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {label} - {type(e).__name__}: {str(e)[:300]}")
            failed += 1
    return failed


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--llm" in argv:
        print("commentary models (one tiny call each)")
        failed = live_check()
        print("\nOK" if failed == 0 else f"\n{failed} model(s) not answering - commentary falls back to templates")
        return 0 if failed == 0 else 1
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
            row(here, f"{path}  ({why})", "make models  (this project's GitHub release models-v1; needs gh auth login)", path not in OPTIONAL_MODELS)
    print("configs")
    for path in ("configs/matches", "configs/scoreboard.yaml", "configs/scoreboard_digits.npz"):
        row((ROOT / path).exists(), path, "see docs/RUNBOOK.md: a new match / a new broadcaster")
    print("GPU")
    try:
        import torch

        gpu = torch.cuda.is_available()
        row(gpu, torch.cuda.get_device_name(0) if gpu else "CUDA GPU", "runs on CPU too, much slower", required=False)
    except Exception:
        row(False, "torch", "pip install -r requirements.txt")
    print("API keys (optional: without one, commentary uses templates; test them with make llm-check)")
    try:
        from cricsense.commentary.llm import api_key

        for why, cfg in model_configs():
            ok = api_key(cfg, ROOT) is not None
            row(ok, f"{key_name(cfg)} {'set' if ok else 'not set'} - {why} ({cfg.provider} {cfg.model})",
                "add it to .env (never commit .env); see configs/commentary.yaml for where to get one", required=False)
    except Exception as e:
        row(False, f"configs/commentary.yaml: {e}", required=False)
    print("\nREADY" if problems == 0 else f"\n{problems} problem(s) - fix the MISS lines above")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
