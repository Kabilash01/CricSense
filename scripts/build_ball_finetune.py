"""Fine-tuning data for the ball detector from the hand-reviewed batch (data/ball_labelling).

    python scripts/build_ball_finetune.py --val st_010500

Converts the reviewed Pascal VOC labels (scripts/label_ball.py) into YOLO crops:
  - 1280x736 windows cut from the 1920x1080 frame at native resolution (the ball is 6-15 px;
    the model runs on full frames at 1920, so it must learn the ball at this size);
  - frames with a ball: CROPS_PER_FRAME windows, the ball placed at different positions;
  - frames without a ball: windows over the pitch - crease marks, stumps and shoes are what
    the old model mistook for the ball, so these are the hard negatives;
  - skipped frames (wrong view) are left out; whole clips go to --val so no delivery is in both.
Outputs data/ball_finetune/ (YOLO format, classes ball, stump).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ball_dataset import CROP_H, CROP_W, crop_window, yolo_line  # noqa: E402
from label_ball import CLASSES, read_voc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CROPS_PER_FRAME = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "ball_labelling")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "ball_finetune")
    parser.add_argument("--val", nargs="+", default=["st_010500"], help="clips for validation")
    args = parser.parse_args(argv)

    review = json.loads((args.src / "review.json").read_text())
    rng = np.random.default_rng(0)
    counts = {"train": [0, 0, 0], "valid": [0, 0, 0]}  # images, ball boxes, no-ball images
    for split in counts:
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)
    for img_path in sorted((args.src / "images").glob("*.jpg")):
        stem = img_path.stem
        if review.get(stem) != "done":
            continue
        clip = stem.rsplit("_", 1)[0]
        split = "valid" if clip in args.val else "train"
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        boxes = [(CLASSES[b["cls"]], np.array(b["box"])) for b in read_voc(args.src / "annotations" / f"{stem}.xml")]
        balls = [b for c, b in boxes if c == 0]
        stumps = [b for c, b in boxes if c == 1]
        if balls:
            centre = ((balls[0][0] + balls[0][2]) / 2, (balls[0][1] + balls[0][3]) / 2)
        elif stumps:  # the pitch: where the look-alikes are
            centre = tuple(np.mean([((s[0] + s[2]) / 2, (s[1] + s[3]) / 2) for s in stumps], axis=0))
        else:
            centre = (w / 2, h / 2)
        for k in range(CROPS_PER_FRAME if split == "train" else 1):
            x0, y0 = crop_window(*centre, w, h, rng)
            lines = [line for c, b in boxes if (line := yolo_line(c, b, x0, y0))]
            name = f"{stem}_{k}"
            cv2.imwrite(str(args.out / split / "images" / f"{name}.jpg"), image[y0 : y0 + CROP_H, x0 : x0 + CROP_W], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (args.out / split / "labels" / f"{name}.txt").write_text("".join(f"{line}\n" for line in lines))
            c = counts[split]
            c[0] += 1
            c[1] += sum(line.startswith("0 ") for line in lines)
            c[2] += not any(line.startswith("0 ") for line in lines)
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: valid/images\nnc: 2\nnames: [ball, stump]\n")
    for split, (n, nb, neg) in counts.items():
        print(f"{split}: {n} crops, {nb} ball boxes, {neg} without a ball (hard negatives)")
    print(f"data: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
