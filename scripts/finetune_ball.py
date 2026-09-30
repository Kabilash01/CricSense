"""Fine-tune the ball detector on this broadcast (data/ball_finetune, scripts/build_ball_finetune.py).

    python scripts/finetune_ball.py            # -> models/ball_ft.pt  (models/ball_best.pt untouched)

The original training set of ball_best.pt is not available, so this is a gentle fine-tune that
keeps what the model knows: backbone frozen, low learning rate, few epochs, and no mosaic
(mosaic shrinks the 6-15 px ball further). Only models/ball_ft.pt is written; the pipeline keeps
ball_best.pt until the new model is shown to be better on the held-out 6-minute clip.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cricsense  # noqa: F401  (disables ultralytics auto-install)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ball_finetune" / "data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "ball_best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "ball_ft.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10, help="backbone layers kept fixed")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        imgsz=1280,
        epochs=args.epochs,
        patience=15,
        batch=args.batch,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.2,
        warmup_epochs=1,
        mosaic=0.0,
        scale=0.2,
        translate=0.05,
        fliplr=0.5,
        device=args.device,
        workers=4,
        project=str(ROOT / "outputs" / "training"),
        name="ball_ft",
        exist_ok=True,
        seed=0,
        plots=True,
    )
    best = ROOT / "outputs" / "training" / "ball_ft" / "weights" / "best.pt"
    shutil.copy2(best, args.out)
    print(f"fine-tuned model: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
