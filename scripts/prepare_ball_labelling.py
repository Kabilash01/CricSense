"""A small, pre-filled labelling batch for the ball detector, from this broadcast.

    python scripts/prepare_ball_labelling.py                 # select frames, pre-fill, previews
    python scripts/prepare_ball_labelling.py --upload        # ... and upload to Roboflow

Why: the ball model was trained on other footage and on this broadcast sees only 4-10 of the
~15 frames of a flight, which is what limits bounce timing and our speed estimate.

  1. the main (behind-the-bowler) view: the frames the calibrate pass registered to a reference
     view (outputs/<clip>/camera.json). Clips without them are skipped - the second innings
     needs its own reference views first (stump detections and image similarity both proved
     unreliable at telling this view from replays and side-on shots);
  2. in each main-view shot, find the flight: ball candidates that move (not the same spot in
     several frames) and take frames spread across it - including the frames the model missed;
  3. pre-fill boxes from the current model (ball >= 0.3, stumps >= 0.5), so the labeller only
     corrects: add missed balls, delete false ones;
  4. write full 1920x1080 JPEGs + Pascal VOC XML (self-contained, Roboflow reads it directly).

The evaluation clips (final_0051_*) are never used, so before/after numbers stay honest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

import cricsense  # noqa: F401  (disables ultralytics auto-install)
from cricsense.ball.candidates import detect_candidates
from cricsense.io import VideoReader

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ["st_001500", "st_003000", "st_010500", "st_011500", "st_014500", "st_020000", "st_022000", "st_024000", "st_025500"]
MIN_SHOT = 20  # frames
PER_SHOT = 6
BALL_PREFILL, STUMP_PREFILL = 0.3, 0.5
NAMES = {0: "ball", 1: "stump"}
PROJECT = "cricsense-ball-broadcast"


def main_view_shots(clip: str) -> list[tuple[int, int]]:
    cam_path = ROOT / "outputs" / clip / "camera.json"
    if not cam_path.is_file():
        return []
    frames = sorted(f["frame_id"] for f in json.loads(cam_path.read_text())["frames"] if f["image_to_pitch"])
    out, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f - prev <= 2:
            prev = f
        else:
            out.append((start, prev))
            start = prev = f
    if start is not None:
        out.append((start, prev))
    return [(a, b) for a, b in out if b - a + 1 >= MIN_SHOT]


def moving_balls(cands: dict[int, list[dict]], a: int, b: int, height: int) -> list[tuple[int, float]]:
    pts = [
        (f, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, d["conf"])
        for f in range(a, b + 1)
        for d in cands.get(f, [])
        if d["cls"] == 0 and d["conf"] >= 0.15 and (d["box"][1] + d["box"][3]) / 2 < 0.85 * height
    ]
    moving = []
    for f, u, v, c in pts:
        same = sum(1 for g, x, y, _ in pts if abs(g - f) >= 2 and abs(x - u) < 4 and abs(y - v) < 4)
        if same < 2:
            moving.append((f, c))
    return moving


def pick_frames(moving: list[tuple[int, float]], a: int, b: int) -> list[int]:
    """Frames spread over the flight: from just before the first moving candidate to just after
    the last, preferring frames where the model is unsure or silent."""
    if len(moving) >= 3:
        fs = sorted({f for f, _ in moving})
        # the densest 30-frame window of moving candidates is the delivery
        best = max(fs, key=lambda s: sum(1 for f in fs if s <= f < s + 30))
        lo, hi = max(a, best - 3), min(b, best + 30)
    else:
        mid = (a + b) // 2
        lo, hi = max(a, mid - 15), min(b, mid + 15)
    conf = {}
    for f, c in moving:
        conf[f] = max(conf.get(f, 0), c)
    window = list(range(lo, hi + 1))
    chunks = np.array_split(np.array(window), PER_SHOT)
    return [int(min(ch, key=lambda f: conf.get(int(f), 0))) for ch in chunks if len(ch)]


def voc_xml(name: str, w: int, h: int, boxes: list[tuple[int, list[float]]]) -> str:
    objs = "".join(
        f"<object><name>{NAMES[c]}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{max(0, round(b[0]))}</xmin><ymin>{max(0, round(b[1]))}</ymin>"
        f"<xmax>{min(w, round(b[2]))}</xmax><ymax>{min(h, round(b[3]))}</ymax></bndbox></object>"
        for c, b in boxes
    )
    return (
        f"<annotation><folder></folder><filename>{escape(name)}</filename>"
        f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>{objs}</annotation>\n"
    )


def upload(out: Path, batch: str) -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set (.env)")
    import roboflow

    ws = roboflow.Roboflow(api_key=key).workspace()
    existing = {p["id"].split("/")[-1] for p in ws.project_list}
    if PROJECT in existing:
        project = ws.project(PROJECT)
    else:
        project = ws.create_project(project_name=PROJECT, project_type="object-detection", project_license="private", annotation="ball")
    images = sorted((out / "images").glob("*.jpg"))
    for i, img in enumerate(images, 1):
        xml = out / "annotations" / f"{img.stem}.xml"
        project.single_upload(image_path=str(img), annotation_path=str(xml), batch_name=batch, num_retry_uploads=3)
        if i % 20 == 0 or i == len(images):
            print(f"uploaded {i}/{len(images)}")
    print(f"project: https://app.roboflow.com/{ws.url}/{PROJECT}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", nargs="+", default=CLIPS)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "ball_labelling")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_best.pt")
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--upload", action="store_true", help="upload the batch to Roboflow")
    parser.add_argument("--batch", default="prefilled-v1")
    args = parser.parse_args(argv)

    if args.upload and (args.out / "images").is_dir():
        upload(args.out, args.batch)
        return 0

    picks: list[tuple[str, int, list[tuple[int, list[float]]]]] = []
    for clip in args.clips:
        video = ROOT / "data" / f"{clip}.mp4"
        cache = ROOT / "outputs" / clip
        with VideoReader(video) as r:
            n, h = r.n_frames, r.height
        clip_shots = main_view_shots(clip)
        if not clip_shots:
            print(f"{clip}: no calibrated main-view frames, skipped")
            continue
        full = {f for a, b in clip_shots for f in range(a, min(b, n - 1) + 1)}
        cands = detect_candidates(video, full, args.weights, cache, device=args.device)
        n_pick = 0
        for a, b in clip_shots:
            for f in pick_frames(moving_balls(cands, a, b, h), a, min(b, n - 1)):
                boxes = [
                    (d["cls"], d["box"])
                    for d in cands.get(f, [])
                    if (d["cls"] == 0 and d["conf"] >= BALL_PREFILL) or (d["cls"] == 1 and d["conf"] >= STUMP_PREFILL)
                ]
                picks.append((clip, f, boxes))
                n_pick += 1
        print(f"{clip}: {len(clip_shots)} main-view shots, {n_pick} frames")

    if len(picks) > args.max_images:  # keep every clip represented
        idx = np.linspace(0, len(picks) - 1, args.max_images).round().astype(int)
        picks = [picks[i] for i in sorted(set(idx))]
    for sub in ("images", "annotations", "previews"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    wanted: dict[str, dict[int, list]] = {}
    for clip, f, boxes in picks:
        wanted.setdefault(clip, {})[f] = boxes
    n_ball = n_empty = 0
    for clip, frames in wanted.items():
        with VideoReader(ROOT / "data" / f"{clip}.mp4") as reader:
            for fr in reader:
                if fr.frame_id not in frames:
                    continue
                boxes = frames[fr.frame_id]
                stem = f"{clip}_{fr.frame_id:06d}"
                h, w = fr.image.shape[:2]
                cv2.imwrite(str(args.out / "images" / f"{stem}.jpg"), fr.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (args.out / "annotations" / f"{stem}.xml").write_text(voc_xml(f"{stem}.jpg", w, h, boxes))
                has_ball = any(c == 0 for c, _ in boxes)
                n_ball += has_ball
                n_empty += not has_ball
                prev = fr.image.copy()
                for c, b in boxes:
                    p = [int(v) for v in b]
                    cv2.rectangle(prev, (p[0] - 8, p[1] - 8), (p[2] + 8, p[3] + 8), (0, 255, 0) if c == 0 else (255, 160, 0), 2)
                cv2.imwrite(str(args.out / "previews" / f"{stem}.jpg"), cv2.resize(prev, (960, 540)))
    print(f"{len(picks)} images -> {args.out}: {n_ball} with a pre-filled ball, {n_empty} without (check these for a missed ball)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
