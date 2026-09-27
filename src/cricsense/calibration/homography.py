"""Pixel <-> pitch mapping for the ground plane.

A homography is only valid for points ON the ground: player foot points and the ball at
the moment it bounces. It says nothing about a ball in flight.

The broadcast camera pans and zooms during every delivery, so a homography fitted on one
reference frame is only correct for that frame; camera_motion.py carries it to the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from cricsense.calibration import pitch_model

# Any three points whose triangle is thinner than this (area / longest side^2) are collinear.
_MIN_TRIANGLE_SHAPE = 0.01


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class PitchHomography:
    """image pixels -> pitch metres. `H` is 3x3 and normalised so that H[2, 2] == 1."""

    H: np.ndarray

    def __post_init__(self) -> None:
        H = np.asarray(self.H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            raise CalibrationError("homography must be a finite, invertible 3x3 matrix")
        object.__setattr__(self, "H", H / H[2, 2])

    @classmethod
    def fit(cls, image_pts: np.ndarray, pitch_pts: np.ndarray) -> PitchHomography:
        """Least-squares fit from >= 4 correspondences; exact for 4."""
        image_pts = _as_points(image_pts)
        pitch_pts = _as_points(pitch_pts)
        if len(image_pts) != len(pitch_pts):
            raise CalibrationError(f"{len(image_pts)} image points but {len(pitch_pts)} pitch points")
        if len(image_pts) < 4:
            raise CalibrationError(f"need at least 4 points, got {len(image_pts)}")
        _check_not_collinear(pitch_pts, "pitch")
        _check_not_collinear(image_pts, "image")
        H, _ = cv2.findHomography(image_pts, pitch_pts, 0)
        if H is None:
            raise CalibrationError("could not fit a homography to these points")
        homography = cls(H)
        homography.check_pitch_visible()
        return homography

    @property
    def H_inv(self) -> np.ndarray:
        H_inv = np.linalg.inv(self.H)
        return H_inv / H_inv[2, 2]

    def image_to_pitch(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) metres. Points at or beyond the horizon come back as NaN."""
        return _apply(self.H, _as_points(pts), self._image_side)

    def pitch_to_image(self, pts: np.ndarray) -> np.ndarray:
        """(N, 2) metres -> (N, 2) pixels. Points behind the camera come back as NaN."""
        return _apply(self.H_inv, _as_points(pts), self._pitch_side)

    def for_frame(self, H_ref_from_frame: np.ndarray) -> PitchHomography:
        """The mapping for another frame, given that frame's pixels -> reference pixels."""
        return PitchHomography(self.H @ np.asarray(H_ref_from_frame, dtype=np.float64))

    def check_pitch_visible(self) -> None:
        """Reject fits where the horizon cuts through the pitch.

        Four points always admit an exact homography, even when two labels are swapped;
        such a fit folds the pitch across the horizon, which shows up as the homogeneous
        coordinate changing sign across the pitch rectangle.
        """
        xs = np.linspace(pitch_model.STRIKER_STUMPS_X, pitch_model.BOWLER_STUMPS_X, 9)
        ys = np.linspace(-pitch_model.PITCH_HALF_WIDTH, pitch_model.PITCH_HALF_WIDTH, 5)
        grid = np.array([(x, y) for x in xs for y in ys])
        w = _homogeneous_w(self.H_inv, grid)
        if not (np.all(w > 0) or np.all(w < 0)):
            raise CalibrationError(
                "the fitted mapping folds the pitch across the horizon; "
                "two landmarks are probably swapped or mislabelled"
            )

    # Sign of the homogeneous coordinate on the visible side of each plane.
    @property
    def _pitch_side(self) -> float:
        return float(np.sign(_homogeneous_w(self.H_inv, np.array([[10.0, 0.0]]))[0]))

    @property
    def _image_side(self) -> float:
        centre = self.pitch_to_image(np.array([[10.0, 0.0]]))
        return float(np.sign(_homogeneous_w(self.H, centre)[0]))


def reprojection_errors(
    homography: PitchHomography, image_pts: np.ndarray, pitch_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point error in pixels (pitch -> image) and in metres (image -> pitch)."""
    image_pts, pitch_pts = _as_points(image_pts), _as_points(pitch_pts)
    px = np.linalg.norm(homography.pitch_to_image(pitch_pts) - image_pts, axis=1)
    m = np.linalg.norm(homography.image_to_pitch(image_pts) - pitch_pts, axis=1)
    return px, m


def _as_points(pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(arr)):
        raise CalibrationError("points must be finite")
    return arr


def _homogeneous_w(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + H[2, 2]


def _apply(H: np.ndarray, pts: np.ndarray, visible_side: float) -> np.ndarray:
    homog = np.c_[pts, np.ones(len(pts))] @ H.T
    w = homog[:, 2]
    out = homog[:, :2] / np.where(np.abs(w) < 1e-12, np.nan, w)[:, None]
    out[np.sign(w) != visible_side] = np.nan
    return out


def _check_not_collinear(pts: np.ndarray, which: str) -> None:
    """Require some 4 of the points with no 3 of them collinear.

    Extra points may share a line (e.g. both bowling-crease corners and the middle stump);
    what must not happen is every quadruple being degenerate.
    """

    def thin(i: int, j: int, k: int) -> bool:
        a, b, c = pts[i], pts[j], pts[k]
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        longest = max(np.sum((a - b) ** 2), np.sum((b - c) ** 2), np.sum((a - c) ** 2))
        return longest == 0 or area / longest < _MIN_TRIANGLE_SHAPE

    for quad in combinations(range(len(pts)), 4):
        if not any(thin(*tri) for tri in combinations(quad, 3)):
            return
    raise CalibrationError(f"no 4 of the {which} points are free of collinear or repeated triples")
