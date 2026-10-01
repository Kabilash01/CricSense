"""The score box's digits, read glyph by glyph against templates of the broadcaster's font.

Tesseract reads the score box's bold, narrow digits poorly where 1s sit together: "11-1" came
back as "11", "1-1", "111" or "-1" under every page mode and scaling tried, so a chase's score
stalled for overs. The box is a fixed font on a flat colour, so its white digits separate into
one connected component each; each is matched against templates of that font, harvested from
the broadcast itself where Tesseract's reading was unambiguous (scripts/harvest_digits.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

GLYPH_SIZE = (16, 24)  # (w, h) every glyph is resized to before comparing
MIN_DIGIT_HEIGHT = 0.6  # of the tallest component: shorter ones are the dash or noise
DASH_MAX_HEIGHT = 0.35  # a dash: this short against the digits, and wider than tall
MAX_DISTANCE = 0.25  # mean absolute difference to the nearest template; worse is not a digit
MIN_AREA_PX = 15


def white_mask(box_bgr: np.ndarray) -> np.ndarray:
    """The box's white digits, 0/1."""
    hsv = cv2.cvtColor(box_bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 200) & (hsv[..., 1] < 80)).astype(np.uint8)


def glyphs(white: np.ndarray) -> list[str | np.ndarray]:
    """Left to right: each digit as a normalised GLYPH_SIZE float image, each dash as "-"."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    comps = [(k, *(int(v) for v in st)) for k, st in enumerate(stats) if k > 0 and st[4] >= MIN_AREA_PX]
    if not comps:
        return []
    top = max(h for *_, h, _ in comps)
    out: list[tuple[int, str | np.ndarray]] = []
    for k, x, y, w, h, _ in comps:
        if h >= MIN_DIGIT_HEIGHT * top:
            crop = (labels[y : y + h, x : x + w] == k).astype(np.float32)
            out.append((x, cv2.resize(crop, GLYPH_SIZE, interpolation=cv2.INTER_AREA)))
        elif h <= DASH_MAX_HEIGHT * top and w > h:
            out.append((x, "-"))
    ordered = [g for _, g in sorted(out, key=lambda xg: xg[0])]
    # a dash sits between digits; one before or after them is the box's rounded corner
    first = next((i for i, g in enumerate(ordered) if not isinstance(g, str)), len(ordered))
    last = max((i for i, g in enumerate(ordered) if not isinstance(g, str)), default=-1)
    return [g for i, g in enumerate(ordered) if not isinstance(g, str) or first < i < last]


class DigitReader:
    def __init__(self, templates: dict[str, np.ndarray]) -> None:
        if not templates:
            raise ValueError("no digit templates")
        self._chars = list(templates)
        self._templates = [np.asarray(templates[c], np.float32).reshape(-1, GLYPH_SIZE[1], GLYPH_SIZE[0]) for c in self._chars]

    @classmethod
    def load(cls, path: Path) -> DigitReader:
        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **dict(zip(self._chars, self._templates)))

    def classify(self, glyph: np.ndarray) -> tuple[str, float]:
        """The nearest template's character and its distance."""
        best = min(
            ((c, float(np.abs(t - glyph).mean(axis=(1, 2)).min())) for c, t in zip(self._chars, self._templates)),
            key=lambda cd: cd[1],
        )
        return best

    def read(self, white: np.ndarray) -> str | None:
        """The box's text ("11-1"), or None if any digit matches no template well enough."""
        text = []
        for g in glyphs(white):
            if isinstance(g, str):
                text.append(g)
                continue
            char, dist = self.classify(g)
            if dist > MAX_DISTANCE:
                return None
            text.append(char)
        return "".join(text) or None
