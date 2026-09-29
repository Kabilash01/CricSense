"""Ball (and stump) candidates from the detector, cached per clip and per weights file."""

from __future__ import annotations

import json
from pathlib import Path

from cricsense.io import VideoReader


def detect_candidates(
    video: Path,
    frames: set[int],
    weights: Path,
    cache_dir: Path,
    imgsz: int = 1920,
    conf: float = 0.1,
    device: str = "cuda:0",
    fresh: bool = False,
    tag: str = "ball_candidates",
) -> dict[int, list[dict]]:
    """frame -> [{"cls", "conf", "box": [x1, y1, x2, y2]}] for the given frames (any YOLO model;
    the ball model's classes are 0 ball, 1 stump; the bat model's 1 bat).

    Runs at native resolution: the ball is 6-15 px and a 1920 frame shrunk to 1280 loses it
    (on the dev clip: candidates in 25 of 30 flight frames at 1920 vs 14 at 1280).
    """
    cache = cache_dir / f"{tag}_{weights.stem}.json"
    cached: dict[int, list[dict]] = {}
    if not fresh and cache.is_file():
        cached = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        if frames <= set(cached):
            return cached
    frames = frames - set(cached)  # only what is not cached yet
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[int, list[dict]] = dict(cached)
    with VideoReader(video) as reader:
        for f in reader:
            if f.frame_id not in frames:
                continue
            r = model.predict(f.image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            out[f.frame_id] = [
                {"cls": int(c), "conf": round(float(s), 3), "box": [round(v, 1) for v in b]}
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
            ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out
