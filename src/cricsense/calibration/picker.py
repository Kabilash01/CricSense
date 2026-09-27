"""Interactive landmark picker and the calibration check drawing.

PickerSession holds the clicking logic and is tested without a display; run_picker is the
thin OpenCV window around it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from cricsense.calibration import pitch_model
from cricsense.calibration.homography import CalibrationError, PitchHomography

DEFAULT_CHECK_LANDMARKS = (
    "striker_bowling_left",
    "striker_bowling_right",
    "bowler_bowling_left",
    "bowler_bowling_right",
)

_PITCH_COLOUR = (0, 255, 255)
_FIT_COLOUR = (0, 0, 255)
_CHECK_COLOUR = (255, 128, 0)


@dataclass
class PickerSession:
    """Click the fit landmarks in order, then optionally the check landmarks (skippable)."""

    fit_names: Sequence[str] = pitch_model.DEFAULT_FIT_LANDMARKS
    check_names: Sequence[str] = DEFAULT_CHECK_LANDMARKS
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        unknown = [n for n in (*self.fit_names, *self.check_names) if n not in pitch_model.LANDMARKS]
        if unknown:
            raise ValueError(f"unknown landmarks: {unknown}")
        if len(self.fit_names) < 4:
            raise ValueError("need at least 4 fit landmarks")

    @property
    def order(self) -> list[str]:
        return [*self.fit_names, *self.check_names]

    def current(self) -> str | None:
        for name in self.order:
            if name not in self.points and name not in self.skipped:
                return name
        return None

    @property
    def fit_complete(self) -> bool:
        return all(n in self.points for n in self.fit_names)

    def click(self, x: float, y: float) -> None:
        name = self.current()
        if name is not None:
            self.points[name] = (float(x), float(y))

    def skip(self) -> bool:
        """Skip the current landmark. Only check landmarks can be skipped."""
        name = self.current()
        if name is None or name in self.fit_names:
            return False
        self.skipped.add(name)
        return True

    def undo(self) -> None:
        done = [n for n in self.order if n in self.points or n in self.skipped]
        if done:
            last = done[-1]
            self.points.pop(last, None)
            self.skipped.discard(last)

    def reset(self) -> None:
        self.points.clear()
        self.skipped.clear()

    def fit(self) -> PitchHomography:
        if not self.fit_complete:
            raise CalibrationError("not all fit landmarks have been clicked")
        return PitchHomography.fit(
            np.array([self.points[n] for n in self.fit_names]),
            np.array([pitch_model.LANDMARKS[n] for n in self.fit_names]),
        )

    def check_points(self) -> dict[str, tuple[float, float]]:
        return {n: self.points[n] for n in self.check_names if n in self.points}


def draw_pitch(image: np.ndarray, homography: PitchHomography, thickness: int = 2) -> np.ndarray:
    """The image with the calibrated creases, pitch outline and stumps drawn on top."""
    out = image.copy()
    for a, b in pitch_model.crease_segments():
        pa, pb = homography.pitch_to_image(np.array([a, b]))
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(out, _pt(pa), _pt(pb), _PITCH_COLOUR, thickness, cv2.LINE_AA)
    for p in homography.pitch_to_image(np.array(pitch_model.stump_points())):
        if np.all(np.isfinite(p)):
            cv2.circle(out, _pt(p), 3, _PITCH_COLOUR, -1, cv2.LINE_AA)
    return out


def draw_session(image: np.ndarray, session: PickerSession, homography: PitchHomography | None) -> np.ndarray:
    """Clicked points, check-point predictions and (once fitted) the projected pitch."""
    out = draw_pitch(image, homography, 1) if homography is not None else image.copy()
    for name, (x, y) in session.points.items():
        colour = _FIT_COLOUR if name in session.fit_names else _CHECK_COLOUR
        cv2.drawMarker(out, (round(x), round(y)), colour, cv2.MARKER_CROSS, 16, 2)
        if homography is not None and name in session.check_names:
            pred = homography.pitch_to_image(np.array([pitch_model.LANDMARKS[name]]))[0]
            if np.all(np.isfinite(pred)):
                cv2.line(out, (round(x), round(y)), _pt(pred), _CHECK_COLOUR, 1, cv2.LINE_AA)
    return out


def _pt(p: np.ndarray) -> tuple[int, int]:
    return round(float(p[0])), round(float(p[1]))


# --- interactive window ------------------------------------------------------------------

_HELP = "click: place  s: skip check point  u: undo  r: reset  n/p: +/-5 frames  N/P: +/-50  Enter: accept  q: quit"


@dataclass
class PickResult:
    frame_id: int
    session: PickerSession


def run_picker(
    get_frame: Callable[[int], np.ndarray | None],
    start_frame: int,
    n_frames: int,
    display_width: int = 1600,
    window: str = "cricsense calibrate",
) -> PickResult | None:
    """Show frames, let the user click landmarks; None if they quit."""
    frame_id = start_frame
    image = get_frame(frame_id)
    if image is None:
        raise ValueError(f"cannot read frame {frame_id}")
    session = PickerSession()
    scale = min(1.0, display_width / image.shape[1])
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            session.click(x / scale, y / scale)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            homography, error = None, ""
            if session.fit_complete:
                try:
                    homography = session.fit()
                except CalibrationError as e:
                    error = str(e)
            canvas = draw_session(image, session, homography)
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale) if scale != 1 else canvas
            _draw_loupe(canvas, image, mouse["x"], mouse["y"], scale)
            prompt = session.current()
            status = (
                f"frame {frame_id} | next: {prompt}" + (" (s to skip)" if prompt in session.check_names else "")
                if prompt
                else f"frame {frame_id} | all points placed - Enter to accept"
            )
            _banner(canvas, [status, error or _HELP], bad=bool(error))
            cv2.imshow(window, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (13, 10) and homography is not None:
                return PickResult(frame_id=frame_id, session=session)
            if key == ord("u"):
                session.undo()
            elif key == ord("r"):
                session.reset()
            elif key == ord("s"):
                session.skip()
            elif key in (ord("n"), ord("p"), ord("N"), ord("P")):
                step = {ord("n"): 5, ord("p"): -5, ord("N"): 50, ord("P"): -50}[key]
                new_id = min(max(frame_id + step, 0), n_frames - 1)
                new_image = get_frame(new_id)
                if new_image is not None and new_id != frame_id:
                    # The camera moves between frames, so points from another frame are invalid.
                    frame_id, image = new_id, new_image
                    session.reset()
    finally:
        cv2.destroyWindow(window)


def _banner(canvas: np.ndarray, lines: list[str], bad: bool) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        colour = (0, 0, 255) if bad and i == 1 else (255, 255, 255)
        cv2.putText(canvas, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def _draw_loupe(canvas: np.ndarray, image: np.ndarray, mx: int, my: int, scale: float, radius: int = 24, zoom: int = 6) -> None:
    """A magnified view around the cursor, top right, for clicking thin crease lines."""
    cx, cy = int(mx / scale), int(my / scale)
    h, w = image.shape[:2]
    x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
    patch = image[y0 : min(cy + radius, h), x0 : min(cx + radius, w)]
    if patch.size == 0:
        return
    big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    px, py = (cx - x0) * zoom + zoom // 2, (cy - y0) * zoom + zoom // 2
    cv2.drawMarker(big, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
    bh, bw = big.shape[:2]
    top, left = 60, canvas.shape[1] - bw - 10
    if left > 0 and top + bh < canvas.shape[0]:
        canvas[top : top + bh, left : left + bw] = big
        cv2.rectangle(canvas, (left, top), (left + bw, top + bh), (255, 255, 255), 1)
