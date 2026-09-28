"""Build the 5-class player dataset from the Roboflow export.

    python scripts/prepare_players_dataset.py --src data/roboflow_v2 --dst data/players_v2

The Roboflow project (kabilash/cricket-player-detection-rtdetr) has 7 classes. The player
detector keeps the five people classes and drops Ball (the ball has its own 1280 px model)
and ScoreCards. Images are hard-linked, not copied, so this costs no extra disk space.
Images left with no boxes are kept as background examples.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

PLAYER_CLASSES = ["Batsman", "Bowler", "Umpire", "Wicket_Keeper", "Player"]
# Roboflow class name -> index in PLAYER_CLASSES; anything not listed is dropped.
SOURCE_TO_PLAYER = {
    "Batsman": 0,
    "Bowler": 1,
    "Umpire": 2,
    "Wicket Keeper": 3,
    "players": 4,
}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def convert(src: Path, dst: Path) -> dict[str, Counter]:
    source_names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    remap = {i: SOURCE_TO_PLAYER[n] for i, n in enumerate(source_names) if n in SOURCE_TO_PLAYER}
    missing = set(SOURCE_TO_PLAYER) - set(source_names)
    if missing:
        raise SystemExit(f"{src}/data.yaml lacks classes {sorted(missing)}")

    counts: dict[str, Counter] = {}
    for split in SPLITS.values():
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        c: Counter = Counter()
        for image in sorted((src / split / "images").iterdir()):
            label = src / split / "labels" / f"{image.stem}.txt"
            kept = []
            for line in label.read_text().splitlines() if label.exists() else []:
                parts = line.split()
                if len(parts) != 5:
                    raise SystemExit(f"{label}: expected 5 columns, got {len(parts)}")
                old = int(parts[0])
                if old in remap:
                    kept.append(" ".join([str(remap[old]), *parts[1:]]))
                    c[PLAYER_CLASSES[remap[old]]] += 1
                else:
                    c[f"dropped:{source_names[old]}"] += 1
            c["images"] += 1
            c["background images"] += not kept
            link = dst / split / "images" / image.name
            if not link.exists():
                os.link(image, link)
            (dst / split / "labels" / f"{image.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        counts[split] = c

    data = {"path": str(dst.resolve()), **{k: f"{v}/images" for k, v in SPLITS.items()}}
    data |= {"nc": len(PLAYER_CLASSES), "names": PLAYER_CLASSES}
    (dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("data/roboflow_v2"))
    parser.add_argument("--dst", type=Path, default=Path("data/players_v2"))
    args = parser.parse_args(argv)
    if args.dst.exists() and any(args.dst.iterdir()):
        parser.error(f"{args.dst} already exists and is not empty")
    for split, c in convert(args.src, args.dst).items():
        print(f"{split}: {dict(sorted(c.items()))}")
    print(f"wrote {args.dst / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
