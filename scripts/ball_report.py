"""Per-delivery ball report for a clip: release, bounce (line/length), speed estimate vs radar.

    python scripts/ball_report.py final_0051_6min [--weights models/ball_best.pt] [--fresh]

Needs outputs/<clip>/camera.json and scoreboard.json. Ball candidates are cached per
weights file in outputs/<clip>/ball_candidates_<weights stem>.json (--fresh re-detects).
Each new radar value shown on the score bar is matched to the last delivery shot before it
first appeared - the broadcaster shows the speed after the ball.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import cricsense  # noqa: F401
from cricsense.ball.delivery import Candidate, FrameGeometry, find_delivery
from cricsense.io import VideoReader, read_doc
from cricsense.schema import CameraDoc, ScoreboardDoc

ROOT = Path(__file__).resolve().parents[1]


def candidates(clip: str, weights: Path, frames: set[int], fresh: bool) -> dict[int, list[dict]]:
    cache = ROOT / "outputs" / clip / f"ball_candidates_{weights.stem}.json"
    legacy = ROOT / "outputs" / clip / "ball_candidates_raw.json"
    if not fresh and not cache.is_file() and weights.stem == "ball_best" and legacy.is_file():
        cache = legacy
    if not fresh and cache.is_file():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out: dict[int, list[dict]] = {}
    with VideoReader(ROOT / "data" / f"{clip}.mp4") as reader:
        for f in reader:
            if f.frame_id in frames:
                r = model.predict(f.image, imgsz=1920, conf=0.1, device="cuda:0", verbose=False)[0]
                out[f.frame_id] = [
                    {"cls": int(c), "conf": round(float(s), 3), "box": [round(v, 1) for v in b]}
                    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
                ]
    cache = ROOT / "outputs" / clip / f"ball_candidates_{weights.stem}.json"
    cache.write_text(json.dumps(out))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_best.pt")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    cam = read_doc(ROOT / "outputs" / args.clip / "camera.json", CameraDoc)
    fps, height = cam.video.fps, cam.video.height
    geo = {
        f.frame_id: FrameGeometry(np.array(f.image_to_pitch), None if f.projection is None else np.array(f.projection))
        for f in cam.frames
        if f.image_to_pitch
    }
    cands = candidates(args.clip, args.weights, set(geo), args.fresh)
    radar: list[tuple[float, float]] = []
    for r in read_doc(ROOT / "outputs" / args.clip / "scoreboard.json", ScoreboardDoc).readings:
        if r.speed_kph and (not radar or radar[-1][0] != r.speed_kph):
            radar.append((r.speed_kph, r.t))

    frames = sorted(geo)
    shots, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            shots.append(cur)
            cur = [b]
    shots.append(cur)

    found = bounced = 0
    errors = []
    for shot in shots:
        cs = [
            Candidate(f, f / fps, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, d["conf"])
            for f in shot
            for d in cands.get(f, [])
            if d["cls"] == 0 and d["conf"] >= 0.15 and (d["box"][1] + d["box"][3]) / 2 < 0.85 * height
        ]
        d = find_delivery(cs, geo, fps)
        later_shots = [s[0] / fps for s in shots if s[0] > shot[0]]
        next_start = min(later_shots) if later_shots else float("inf")
        rd = next((k for k, t in radar if shot[0] / fps < t < next_start), None)
        head = f"{shot[0] / fps:6.1f}s"
        if d is None:
            print(f"{head}  no delivery found" + (f"  | radar {rd}" if rd else ""))
            continue
        found += 1
        b = d.bounce
        if b is None:
            print(f"{head}  release frame {d.release_frame}, {len(d.track)} detections, no bounce" + (f"  | radar {rd}" if rd else ""))
            continue
        bounced += 1
        sp = d.speed_estimate_kph
        cmp = ""
        if rd and sp:
            errors.append(sp / rd - 1)
            cmp = f" ({100 * (sp / rd - 1):+.0f}%)"
        print(
            f"{head}  release frame {d.release_frame}, {len(d.track)} detections | bounce frame {b.t * fps:6.1f}: "
            f"length {b.x:5.2f} m, line {b.y:+.2f} m{'' if b.confirmed else ' (unconfirmed)'} | "
            f"estimate {sp:5.1f} km/h" + (f" | radar {rd}{cmp}" if rd else "")
        )
    summary = f"\n{len(shots)} shots: delivery found {found}, bounce found {bounced}"
    if errors:
        e = np.abs(errors)
        summary += f"; speed estimate vs radar on {len(e)}: median |error| {100 * np.median(e):.0f}%, worst {100 * e.max():.0f}%"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
