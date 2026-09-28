"""Train the player detector (YOLOv8m, 5 classes) on data/players_v2.

    python scripts/train_players.py                  # full run
    python scripts/train_players.py --smoke          # ~1 min check of data, GPU and memory

Hyperparameters match the original yolov8m_production4 run (65 epochs, 640 px, AdamW,
lr0 0.002, mosaic 0.7, close_mosaic 10, patience 20) so the result is comparable. Batch 8
fits the 8 GB RTX 5060 with AMP. The best checkpoint is copied to models/players_best.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
HYPERPARAMS = dict(
    imgsz=640,
    epochs=65,
    batch=8,
    optimizer="AdamW",
    lr0=0.002,
    mosaic=0.7,
    close_mosaic=10,
    patience=20,
    workers=6,
    seed=0,
    deterministic=False,
    plots=True,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data/players_v2/data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/yolov8m.pt", help="COCO-pretrained start point")
    parser.add_argument("--name", default="players_v2")
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 2%% of the data")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error(f"{args.data} not found; run scripts/prepare_players_dataset.py first")
    if not args.weights.is_file():
        parser.error(f"{args.weights} not found; download yolov8m.pt from the ultralytics assets release")

    params = dict(HYPERPARAMS)
    name = args.name
    if args.smoke:
        params |= dict(epochs=1, fraction=0.02, plots=False)
        name = f"{name}_smoke"

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        device=args.device,
        project=str(ROOT / "outputs/training"),
        name=name,
        exist_ok=args.smoke,
        **params,
    )
    best = Path(model.trainer.best)
    if not args.smoke and best.is_file():
        dest = ROOT / "models/players_best.pt"
        shutil.copy2(best, dest)
        print(f"copied {best} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
