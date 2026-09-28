"""Read the broadcast score bar: batters, striker, score, overs, bowler and radar speed.

Replaces CricSense-AI src/ocr/scoreboard_ocr.py, which only allowed the characters
"0123456789/.:" (so it could not read a single name) and expected scores as "123/4"
where this broadcast shows "82-3". The layout lives in configs/scoreboard.yaml.

Readings are evidence: the score engine cross-checks against them, and identity uses the
names, but nothing here is treated as authoritative.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import yaml

from cricsense.schema import BatterLine, ScoreboardReading

Rect = tuple[int, int, int, int]

_NAME_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
_NUMBER_CHARS = "0123456789-."
_MIN_SPEED_KPH, _MAX_SPEED_KPH = 50.0, 170.0  # anything outside is a misread, not a delivery


@dataclass(frozen=True)
class ScoreboardLayout:
    name: str
    reference_size: tuple[int, int]
    presence_rect: Rect
    presence_hue: tuple[int, int]
    presence_min_fraction: float
    fields: dict[str, Rect]
    striker_icons: tuple[Rect, Rect]
    icon_hue: tuple[int, int]
    icon_min_fraction: float
    sample_every: int
    squad_names: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> ScoreboardLayout:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        icon = raw["striker_icon"]
        return cls(
            name=raw["layout"],
            reference_size=tuple(raw["reference_size"]),
            presence_rect=tuple(raw["presence"]["rect"]),
            presence_hue=tuple(raw["presence"]["hue"]),
            presence_min_fraction=float(raw["presence"]["min_fraction"]),
            fields={k: tuple(v) for k, v in raw["fields"].items()},
            striker_icons=(tuple(icon["batter1"]), tuple(icon["batter2"])),
            icon_hue=tuple(icon["hue"]),
            icon_min_fraction=float(icon["min_fraction"]),
            sample_every=int(raw.get("sample_every", 12)),
            squad_names=tuple(n for names in (raw.get("squads") or {}).values() for n in names),
        )


# --- parsing (pure) ----------------------------------------------------------------------


def parse_int(text: str) -> int | None:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def parse_score(text: str) -> tuple[int | None, int | None]:
    """'82-3' -> (82, 3). Wickets are 0-10."""
    m = re.search(r"(\d{1,3})\s*[-/]\s*(\d{1,2})", text)
    if not m or int(m.group(2)) > 10:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_overs(text: str) -> int | None:
    """'11.1 OVERS' -> 67 legal balls; '11 OVERS' -> 66. A ball digit above 5 is a misread."""
    m = re.search(r"(\d{1,2})(?:\s*\.\s*(\d))?", text)
    if not m:
        return None
    balls = int(m.group(2) or 0)
    return None if balls > 5 else int(m.group(1)) * 6 + balls


def parse_speed_kph(text: str) -> float | None:
    """'SPEED 81.4KPH | 50.6 MPH' -> 81.4, only when it is a plausible delivery speed."""
    m = re.search(r"(\d{2,3}(?:\.\d)?)\s*K\s*P\s*H", text.upper())
    if not m:
        return None
    kph = float(m.group(1))
    return kph if _MIN_SPEED_KPH <= kph <= _MAX_SPEED_KPH else None


def clean_name(text: str, squad: tuple[str, ...] = ()) -> str | None:
    """Uppercase letters only; snapped to a squad name when one is close enough."""
    name = re.sub(r"[^A-Z ]", "", text.upper())
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) < 2:
        return None
    if squad:
        close = difflib.get_close_matches(name, squad, n=1, cutoff=0.75)
        if close:
            return close[0]
    return name


# --- image side --------------------------------------------------------------------------


# Mean absolute grey-level change (0-255) below which a field is treated as unchanged.
_UNCHANGED = 4.0


class ScoreboardReader:
    """Stateful: fields whose pixels have not changed since the last read reuse their text.

    The bar changes a few times per delivery, so this skips almost all OCR calls.
    """

    def __init__(self, layout: ScoreboardLayout) -> None:
        self.layout = layout
        self._cache: dict[str, tuple[np.ndarray, str]] = {}
        self.ocr_calls = 0

    def is_visible(self, image: np.ndarray) -> bool:
        return self._hue_fraction(image, self.layout.presence_rect, self.layout.presence_hue, min_sat=120) >= (
            self.layout.presence_min_fraction
        )

    def striker_index(self, image: np.ndarray) -> int | None:
        """0 or 1: which batter row carries the bat icon; None if neither or both do."""
        marks = [
            self._hue_fraction(image, rect, self.layout.icon_hue, min_sat=100) >= self.layout.icon_min_fraction
            for rect in self.layout.striker_icons
        ]
        return marks.index(True) if marks.count(True) == 1 else None

    def read(self, image: np.ndarray, frame_id: int, t: float) -> ScoreboardReading | None:
        if not self.is_visible(image):
            return None
        raw = {name: self._ocr(image, rect, name) for name, rect in self.layout.fields.items()}
        striker = self.striker_index(image)
        squad = self.layout.squad_names
        batters = [
            BatterLine(
                name=clean_name(raw[f"batter{i}_name"], squad),
                runs=parse_int(raw[f"batter{i}_runs"]),
                balls=parse_int(raw[f"batter{i}_balls"]),
                on_strike=striker == i - 1,
            )
            for i in (1, 2)
        ]
        runs, wickets = parse_score(raw["score"])
        bowler_wickets, bowler_runs = parse_score(raw["bowler_figures"])
        return ScoreboardReading(
            frame_id=frame_id,
            t=t,
            batters=batters,
            runs=runs,
            wickets=wickets,
            legal_balls=parse_overs(raw["overs"]),
            bowler=clean_name(raw["bowler_name"], squad),
            bowler_wickets=bowler_wickets,
            bowler_runs=bowler_runs,
            bowler_legal_balls=parse_overs(raw["bowler_overs"]),
            speed_kph=parse_speed_kph(raw["lower"]),
            raw=raw,
        )

    def _scaled(self, image: np.ndarray, rect: Rect) -> tuple[int, int, int, int]:
        h, w = image.shape[:2]
        rw, rh = self.layout.reference_size
        x0, y0, x1, y1 = rect
        return round(x0 * w / rw), round(y0 * h / rh), round(x1 * w / rw), round(y1 * h / rh)

    def _hue_fraction(self, image: np.ndarray, rect: Rect, hue: tuple[int, int], min_sat: int) -> float:
        x0, y0, x1, y1 = self._scaled(image, rect)
        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if hsv.size == 0:
            return 0.0
        mask = (hsv[..., 0] >= hue[0]) & (hsv[..., 0] <= hue[1]) & (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= 120)
        return float(mask.mean())

    def _ocr(self, image: np.ndarray, rect: Rect, field: str) -> str:
        x0, y0, x1, y1 = self._scaled(image, rect)
        gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        cached = self._cache.get(field)
        if cached is not None and cached[0].shape == gray.shape:
            if float(np.abs(cached[0].astype(np.int16) - gray).mean()) < _UNCHANGED:
                return cached[1]
        text = self._tesseract(gray, field)
        self._cache[field] = (gray, text)
        return text

    def _tesseract(self, gray: np.ndarray, field: str) -> str:
        self.ocr_calls += 1
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary.mean() < 127:  # light text on a dark panel: tesseract wants dark on light
            binary = 255 - binary
        binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        if field.endswith("_name"):
            config = f"--psm 7 -c tessedit_char_whitelist={_NAME_CHARS.replace(' ', '')}"
        elif field == "lower":
            config = "--psm 7"
        else:
            config = f"--psm 7 -c tessedit_char_whitelist={_NUMBER_CHARS}"
        return pytesseract.image_to_string(binary, config=config).strip()
