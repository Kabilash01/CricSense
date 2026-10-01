"""Collect templates of the score box's font from a broadcast (cricsense.identity.digits).

    python scripts/harvest_digits.py --video "Cricket match/replay-final-sa-vs-ind.mp4" \
        --out configs/scoreboard_digits.npz

Every --step seconds the score box is found by colour and read by Tesseract; a reading is used
only if it is a whole score ("82-3") with exactly as many digit glyphs as digits and the dash in
the same place, so each glyph gets the character Tesseract put there. Per character, glyphs far
from the character's median are dropped (a misread that happened to have the right count), and
the most typical are kept. Leave-one-out accuracy on the kept glyphs is printed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from cricsense.identity import digits
from cricsense.identity.scoreboard_ocr import ScoreboardLayout, ScoreboardReader

ROOT = Path(__file__).resolve().parents[1]
OUTLIER = 0.15  # mean absolute difference to the character's median glyph
KEEP = 25  # templates per character


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--layout", type=Path, default=ROOT / "configs" / "scoreboard.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "scoreboard_digits.npz")
    ap.add_argument("--step", type=float, default=10.0, help="seconds between samples")
    args = ap.parse_args(argv)

    layout = ScoreboardLayout.from_yaml(args.layout)
    reader = ScoreboardReader(layout)
    reader.digit_reader = None  # label with Tesseract only
    cap = cv2.VideoCapture(str(args.video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples: dict[str, list[np.ndarray]] = {str(d): [] for d in range(10)}
    used = 0
    for k, frame_id in enumerate(range(0, n, max(1, int(args.step * fps)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = cap.read()
        if not ok:
            continue
        for _, box in reader.score_boxes(image):
            white = digits.white_mask(box)
            m = re.fullmatch(r"-?(\d{1,3}-\d{1,2})", reader.tesseract_box(white).strip().replace(" ", ""))
            if not m:  # a leading "-" is the box's rounded corner
                continue
            text = m.group(1)
            gl = digits.glyphs(white)
            if len(gl) != len(text) or any(isinstance(g, str) != (c == "-") for g, c in zip(gl, text)):
                continue
            for g, c in zip(gl, text):
                if c != "-":
                    samples[c].append(g)
            used += 1
            break
        if k % 200 == 0:
            print(f"{frame_id / fps:7.0f} s  scores used {used}", file=sys.stderr)

    templates = {}
    for c, gs in samples.items():
        if not gs:
            print(f"no samples of {c!r}: the broadcast never showed it cleanly", file=sys.stderr)
            continue
        arr = np.stack(gs)
        med = np.median(arr, axis=0)
        dist = np.abs(arr - med).mean(axis=(1, 2))
        good = arr[dist <= OUTLIER]
        templates[c] = good[np.argsort(dist[dist <= OUTLIER])][:KEEP]
        print(f"{c}: {len(gs)} glyphs, {len(good)} kept as consistent, {len(templates[c])} templates")

    # leave-one-out on the kept templates
    right = total = 0
    for c, ts in templates.items():
        for i in range(len(ts)):
            others = {d: (np.delete(t, i, axis=0) if d == c else t) for d, t in templates.items()}
            others = {d: t for d, t in others.items() if len(t)}
            got, _ = digits.DigitReader(others).classify(ts[i])
            right += got == c
            total += 1
    print(f"leave-one-out: {right}/{total}")
    digits.DigitReader(templates).save(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
