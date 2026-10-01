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
    anchor: dict | None = None  # the score box, found by colour where the layout misses it (see read)

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
            anchor=raw.get("anchor"),
            sample_every=int(raw.get("sample_every", 12)),
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
        # Bars often print a surname prefix in small lower case ("de KOCK", "van der DUSSEN")
        # that OCR drops or mangles: match the last word, unless two players share it.
        surnames: dict[str, list[str]] = {}
        for full in squad:
            surnames.setdefault(full.split()[-1], []).append(full)
        unique = {k: v[0] for k, v in surnames.items() if len(v) == 1}
        close = difflib.get_close_matches(name.split()[-1], list(unique), n=1, cutoff=0.75)
        if close:
            return unique[close[0]]
    return name


# --- image side --------------------------------------------------------------------------


# Mean absolute grey-level change (0-255) below which a field is treated as unchanged.
_UNCHANGED = 4.0


class ScoreboardReader:
    """Stateful: fields whose pixels have not changed since the last read reuse their text.

    The bar changes a few times per delivery, so this skips almost all OCR calls.
    """

    def __init__(self, layout: ScoreboardLayout, squad_names: tuple[str, ...] = ()) -> None:
        self.layout = layout
        self.squad_names = squad_names  # names as the bar prints them, to snap OCR slips to
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
        """The bar's fields, from the layout's rectangles. Where they miss the score - at times
        the bar sits ~20 px higher or slides sideways, and in a chase a compact bar shows the
        score beside "NEED 26 FROM 24 BALLS" - the score box is found by its colour and its
        digits read, with the overs to its right: a score-only reading (other fields are kept
        only if the layout read them). Panels shift independently, so the other fields are not
        re-read from the box's offset; a misread score is caught downstream by the rule that a
        score cannot go backwards or faster than the game."""
        reading = self._read_fields(image, frame_id, t) if self.is_visible(image) else None
        if not self.layout.anchor or (reading is not None and reading.runs is not None and reading.legal_balls is not None):
            return reading
        found = self._find_score_box(image)
        if found is None:
            return reading
        runs, wickets, (bx, by, bw, bh) = found
        ox0, ox1 = self.layout.anchor["overs_right"]
        crop = image[by : by + bh, min(bx + bw + ox0, image.shape[1]) : min(bx + bw + ox1, image.shape[1])]
        overs = self._tesseract(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), "overs") if crop.size else ""  # box at the frame's edge
        base = reading or ScoreboardReading(
            frame_id=frame_id, t=t, batters=[BatterLine(name=None, runs=None, balls=None, on_strike=False)] * 2,
            runs=None, legal_balls=None, bowler=None, bowler_runs=None, bowler_legal_balls=None,
        )
        return base.model_copy(update={
            "runs": runs,
            "wickets": _within(wickets, 0, 10),
            "legal_balls": parse_overs(overs),
            "raw": {**base.raw, "variant": "score box", "score": f"{runs}-{wickets}", "overs": overs},
        })

    def _find_score_box(self, image: np.ndarray) -> tuple[int, int, tuple[int, int, int, int]] | None:
        """(runs, wickets, box x, y, w, h in pixels) of a coloured box whose white digits read as a score."""
        a = self.layout.anchor
        h_img = image.shape[0]
        y0, y1 = (round(v * h_img / self.layout.reference_size[1]) for v in a["band"])
        hsv = cv2.cvtColor(image[y0:y1], cv2.COLOR_BGR2HSV)
        lo, hi = a["hue"]
        mask = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi) & (hsv[..., 1] > a["min_sat"]) & (hsv[..., 2] > a["min_val"])).astype(np.uint8)
        # the score's white digits split the box: close them over
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        (wmin, wmax), (hmin, hmax) = a["width"], a["height"]
        for x, y, w, h, _ in sorted((tuple(int(v) for v in st) for st in stats[1:]), key=lambda st: -st[4]):
            if not (wmin <= w <= wmax and hmin <= h <= hmax and 1.8 <= w / h <= 5.5):
                continue
            # only the rows that are the box itself: a border or graphic above it is not text
            rows = np.where(mask[y : y + h, x : x + w].mean(axis=1) > 0.5)[0]
            if len(rows) < hmin:
                continue
            y, h = y + int(rows.min()), int(rows.max() - rows.min() + 1)
            box = image[y0 + y : y0 + y + h, x : x + w]
            if box.size == 0:
                continue
            roi = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            white = ((roi[..., 2] > 200) & (roi[..., 1] < 80)).astype(np.uint8) * 255  # the digits
            text_img = cv2.copyMakeBorder(
                255 - cv2.resize(white, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
            )
            self.ocr_calls += 1
            text = pytesseract.image_to_string(text_img, config="--psm 7 -c tessedit_char_whitelist=0123456789-/")
            # the box's rounded corners can read as a leading "-": search, do not match whole
            m = re.search(r"(\d{1,3})[-/](\d{1,2})", text.replace(" ", ""))
            if m:
                return int(m.group(1)), int(m.group(2)), (x, y0 + y, w, h)
        return None

    def _read_fields(self, image: np.ndarray, frame_id: int, t: float) -> ScoreboardReading:
        raw = {name: self._ocr(image, rect, name) for name, rect in self.layout.fields.items()}
        striker = self.striker_index(image)
        squad = self.squad_names
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
        # an OCR misread costs that field, never the reading: impossible values become unknown
        wickets, bowler_wickets = _within(wickets, 0, 10), _within(bowler_wickets, 0, 10)
        speed = parse_speed_kph(raw["lower"])
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
            speed_kph=speed if speed is None or 40 <= speed <= 170 else None,
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


def _within(value: int | None, lo: int, hi: int) -> int | None:
    return value if value is not None and lo <= value <= hi else None

