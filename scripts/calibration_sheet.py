"""One image with every clip's automatic calibration drawn, to check by eye.

    python scripts/calibration_sheet.py "outputs/we_*" --out outputs/calibration_sheet.jpg

Each clip's calibration_check.png shows the regulation creases (green) where the calibration
puts them. A good calibration lies on the white painted creases at both ends. A wrong one -
lines across the grass, on an advert, at the wrong end - must be removed before anything else
uses it: delete configs/calibration/<clip>.yaml and outputs/<clip>/, then calibrate that clip
with --cameras-from <a good calibration of the same match>, or by hand (cricsense calibrate).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

TILE = (640, 360)
PER_ROW = 4


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="output folders or globs, e.g. 'outputs/we_*'")
    ap.add_argument("--out", type=Path, default=Path("outputs/calibration_sheet.jpg"))
    args = ap.parse_args(argv)

    folders = sorted({Path(p) for pattern in args.dirs for p in glob.glob(pattern)})
    tiles, missing = [], []
    for folder in folders:
        check = folder / "calibration_check.png"
        if not check.is_file():
            missing.append(folder.name)
            continue
        tile = cv2.resize(cv2.imread(str(check)), TILE)
        cv2.rectangle(tile, (0, 0), (TILE[0], 44), (0, 0, 0), -1)
        cv2.putText(tile, folder.name, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        print("no calibration_check.png found")
        return 1
    while len(tiles) % PER_ROW:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + PER_ROW]) for i in range(0, len(tiles), PER_ROW)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet)
    print(f"{args.out}: {len(folders) - len(missing)} calibrated")
    if missing:
        print(f"not calibrated ({len(missing)}): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
