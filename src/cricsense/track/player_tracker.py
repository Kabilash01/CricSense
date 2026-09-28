"""Multi-object tracking of players: ByteTrack or BoT-SORT (ultralytics implementations).

BoT-SORT normally estimates camera motion itself from image features. Here that estimate
is replaced, whenever both frames are calibrated, by the motion implied by our own per-frame
pitch homographies (camera.json), which are fitted against the whole scene and checked
against the reference calibration. Uncalibrated frames fall back to BoT-SORT's own method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils.gmc import GMC
from ultralytics.utils import IterableSimpleNamespace, YAML

_GRID = np.array([(x, y) for x in np.linspace(0, 1, 5) for y in np.linspace(0, 1, 5)])


class CameraMotionGMC:
    """Drop-in for ultralytics' GMC: `apply()` returns the 2x3 previous->current frame warp."""

    def __init__(self, fallback_method: str, frame_size: tuple[int, int]) -> None:
        self.method = "camera.json"
        self._fallback = GMC(method=fallback_method)
        self._grid = _GRID * np.array(frame_size, dtype=float)  # sample points across the frame
        self._prev_H: np.ndarray | None = None
        self._cur_H: np.ndarray | None = None
        self.used_calibration = 0
        self.used_fallback = 0

    def set_frame(self, image_to_pitch: np.ndarray | None) -> None:
        self._prev_H, self._cur_H = self._cur_H, image_to_pitch

    def apply(self, raw_frame: np.ndarray, detections: np.ndarray | None = None) -> np.ndarray:
        # Always run the fallback so its own previous-frame state stays current.
        fallback = self._fallback.apply(raw_frame, detections)
        if self._prev_H is None or self._cur_H is None:
            self.used_fallback += 1
            return fallback
        self.used_calibration += 1
        # previous pixels -> pitch -> current pixels, as a full homography...
        M = np.linalg.inv(self._cur_H) @ self._prev_H
        src = np.c_[self._grid, np.ones(len(self._grid))]
        dst = src @ M.T
        dst = dst[:, :2] / dst[:, 2:3]
        # ...then the best affine approximation, which is what the Kalman state can absorb.
        A, *_ = np.linalg.lstsq(src, dst, rcond=None)
        return A.T

    def reset_params(self) -> None:
        self._fallback.reset_params()
        self._prev_H = self._cur_H = None


@dataclass
class PlayerTracker:
    """Wraps an ultralytics tracker; call update() once per processed frame, in order."""

    tracker: BYTETracker
    gmc: CameraMotionGMC | None

    @classmethod
    def from_config(
        cls, tracker_yaml: Path, frame_size: tuple[int, int], root: Path, device: str | None = None
    ) -> PlayerTracker:
        args = IterableSimpleNamespace(**YAML.load(tracker_yaml))
        kind = args.tracker_type
        reid = getattr(args, "model", None)
        if reid and reid != "auto" and not Path(reid).is_absolute():
            args.model = str(root / reid)
        if kind == "bytetrack":
            return cls(BYTETracker(args), None)
        if kind != "botsort":
            raise ValueError(f"{tracker_yaml}: tracker_type must be bytetrack or botsort, got {kind!r}")
        if device is not None:
            args.device = device
        tracker = BOTSORT(args)
        gmc = None
        if getattr(args, "camera_motion_from_calibration", False):
            gmc = CameraMotionGMC(args.gmc_method, frame_size)
            tracker.gmc = gmc
        return cls(tracker, gmc)

    def update(self, boxes, image: np.ndarray, image_to_pitch: np.ndarray | None) -> np.ndarray:
        """boxes: ultralytics Boxes (numpy). Returns rows (x1, y1, x2, y2, track_id, score, cls, det_idx)."""
        if self.gmc is not None:
            self.gmc.set_frame(image_to_pitch)
        out = self.tracker.update(boxes, image)
        return out if len(out) else np.zeros((0, 8), dtype=np.float32)

    def reset(self) -> None:
        self.tracker.reset()
