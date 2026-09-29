"""Self-training dataset for the ball detector, from this broadcast, with no hand labelling.

    python scripts/build_ball_dataset.py --clips st_001500 st_003000 ... --val st_025500 st_024000

For each clip (already calibrated: outputs/<clip>/camera.json):
  1. run the current ball model at native resolution on the main-view frames;
  2. find each shot's delivery (release anchor + physically constrained path, see
     cricsense.ball.delivery) and keep only confident ones;
  3. label the ball in every frame of the delivery: the detector's box where it found the
     ball, and a box interpolated from its neighbours where it missed it (gaps of up to 3
     frames) - the misses are exactly what the fine-tuned model must learn to catch;
  4. add stump boxes the model already finds confidently, so both classes keep training;
  5. save 1280x736 crops at native resolution around the ball (the ball is 6-15 px;
     downscaling a 1920 frame to 1280 would shrink it further).
Clips given with --val go to the validation split, so no delivery appears in both.
Outputs data/ball_selftrain/ (YOLO format) and a few overlay previews for spot checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import cricsense  # noqa: F401  (disables ultralytics auto-install)
from cricsense.ball.delivery import Candidate, FrameGeometry, find_delivery
from cricsense.io import VideoReader, read_doc
from cricsense.schema import CameraDoc

ROOT = Path(__file__).resolve().parents[1]
CROP_W, CROP_H = 1280, 736
MIN_TRACK = 7
MIN_MEAN_CONF = 0.35
MAX_INTERP_GAP = 3
STUMP_MIN_CONF = 0.6


def shots_of(frames: list[int]) -> list[list[int]]:
    shots, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            shots.append(cur)
            cur = [b]
    shots.append(cur)
    return shots


def detect(model, clip: str, frames: set[int], device: str) -> dict[int, list[dict]]:
    cache = ROOT / "outputs" / clip / "ball_candidates_raw.json"
    if cache.is_file():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    out: dict[int, list[dict]] = {}
    with VideoReader(ROOT / "data" / f"{clip}.mp4") as reader:
        for f in reader:
            if f.frame_id not in frames:
                continue
            r = model.predict(f.image, imgsz=1920, conf=0.1, device=device, verbose=False)[0]
            out[f.frame_id] = [
                {"cls": int(c), "conf": round(float(s), 3), "box": [round(v, 1) for v in b]}
                for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
            ]
    cache.write_text(json.dumps(out))
    return out


def ball_labels(delivery, cands: dict[int, list[dict]]) -> dict[int, tuple[np.ndarray, bool]]:
    """frame -> (box xyxy, detected?) along the delivery, interpolating short gaps."""
    boxes: dict[int, np.ndarray] = {}
    for c, _, _ in delivery.track:
        best = min(
            (d for d in cands[c.frame_id] if d["cls"] == 0),
            key=lambda d: np.hypot((d["box"][0] + d["box"][2]) / 2 - c.u, (d["box"][1] + d["box"][3]) / 2 - c.v),
        )
        boxes[c.frame_id] = np.array(best["box"], float)
    out = {f: (b, True) for f, b in boxes.items()}
    known = sorted(boxes)
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= MAX_INTERP_GAP + 1:
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                out[f] = ((1 - w) * boxes[a] + w * boxes[b], False)
    return out


def crop_window(cx: float, cy: float, w: int, h: int, rng: np.random.Generator) -> tuple[int, int]:
    # the ball somewhere in the middle half of the crop, not always dead centre
    x0 = int(np.clip(cx - CROP_W * rng.uniform(0.3, 0.7), 0, w - CROP_W))
    y0 = int(np.clip(cy - CROP_H * rng.uniform(0.3, 0.7), 0, h - CROP_H))
    return x0, y0


def yolo_line(cls: int, box: np.ndarray, x0: int, y0: int) -> str | None:
    x1, y1, x2, y2 = box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0
    if x2 <= 0 or y2 <= 0 or x1 >= CROP_W or y1 >= CROP_H:
        return None
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, CROP_W), min(y2, CROP_H)
    return f"{cls} {(x1 + x2) / 2 / CROP_W:.6f} {(y1 + y2) / 2 / CROP_H:.6f} {(x2 - x1) / CROP_W:.6f} {(y2 - y1) / CROP_H:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", nargs="+", required=True, help="training clips (video ids)")
    parser.add_argument("--val", nargs="+", required=True, help="validation clips (video ids)")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "ball_selftrain")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_best.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    rng = np.random.default_rng(0)
    previews = args.out / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    totals = {"train": [0, 0, 0, 0], "valid": [0, 0, 0, 0]}  # deliveries, images, detected, interpolated
    for clip in [*args.clips, *args.val]:
        split = "valid" if clip in args.val else "train"
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
        cam = read_doc(ROOT / "outputs" / clip / "camera.json", CameraDoc)
        geo = {
            f.frame_id: FrameGeometry(np.array(f.image_to_pitch), None if f.projection is None else np.array(f.projection))
            for f in cam.frames
            if f.image_to_pitch
        }
        if not geo:
            print(f"{clip}: no calibrated frames, skipped")
            continue
        cands = detect(model, clip, set(geo), args.device)
        fps = cam.video.fps
        labels: dict[int, list[str]] = {}
        centres: dict[int, tuple[float, float]] = {}
        kept = 0
        for shot in shots_of(sorted(geo)):
            cs = [
                Candidate(f, f / fps, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, d["conf"])
                for f in shot
                for d in cands.get(f, [])
                if d["cls"] == 0 and d["conf"] >= 0.15 and (d["box"][1] + d["box"][3]) / 2 < 0.85 * cam.video.height
            ]
            delivery = find_delivery(cs, geo, fps)
            if delivery is None or len(delivery.track) < MIN_TRACK or delivery.bounce is None:
                continue
            if np.mean([c.conf for c, _, _ in delivery.track]) < MIN_MEAN_CONF:
                continue
            kept += 1
            for f, (box, detected) in ball_labels(delivery, cands).items():
                labels.setdefault(f, []).append(("ball", box, detected))
                centres[f] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if not labels:
            print(f"{clip}: no confident deliveries")
            continue
        n_det = n_int = n_img = 0
        with VideoReader(ROOT / "data" / f"{clip}.mp4") as reader:
            for frame in reader:
                if frame.frame_id not in labels:
                    continue
                h, w = frame.image.shape[:2]
                x0, y0 = crop_window(*centres[frame.frame_id], w, h, rng)
                lines = []
                for _, box, detected in labels[frame.frame_id]:
                    line = yolo_line(0, box, x0, y0)
                    if line:
                        lines.append(line)
                        n_det += detected
                        n_int += not detected
                for d in cands.get(frame.frame_id, []):
                    if d["cls"] == 1 and d["conf"] >= STUMP_MIN_CONF:
                        line = yolo_line(1, np.array(d["box"]), x0, y0)
                        if line:
                            lines.append(line)
                if not lines:
                    continue
                stem = f"{clip}_{frame.frame_id:06d}"
                crop = frame.image[y0 : y0 + CROP_H, x0 : x0 + CROP_W]
                cv2.imwrite(str(args.out / split / "images" / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (args.out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_img += 1
                if rng.random() < 0.05:  # previews for spot checks
                    prev = crop.copy()
                    for _, box, detected in labels[frame.frame_id]:
                        b = (box - [x0, y0, x0, y0]).astype(int)
                        cv2.rectangle(prev, (b[0] - 6, b[1] - 6), (b[2] + 6, b[3] + 6), (0, 255, 0) if detected else (0, 0, 255), 2)
                    cv2.imwrite(str(previews / f"{stem}.jpg"), prev)
        t = totals[split]
        t[0] += kept
        t[1] += n_img
        t[2] += n_det
        t[3] += n_int
        print(f"{clip} [{split}]: {kept} confident deliveries -> {n_img} images, ball boxes {n_det} detected + {n_int} interpolated")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: valid/images\nnc: 2\nnames: [ball, stump]\n"
    )
    for split, (d, i, det, interp) in totals.items():
        print(f"TOTAL {split}: {d} deliveries, {i} images, ball boxes {det} detected + {interp} interpolated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
