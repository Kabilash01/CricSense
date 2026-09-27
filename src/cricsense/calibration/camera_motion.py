"""Carry a reference-frame calibration to every frame of a video.

Broadcast cameras pan, tilt and zoom about a fixed mount. For that motion the mapping
between two frames of the same camera is a homography for the whole scene, stands
included, so matching image features anywhere outside the on-screen graphics gives the
frame -> reference homography, and composing it with the reference calibration gives the
frame -> pitch mapping.

Each frame is matched directly against every reference view first, so errors do not
accumulate. If that fails (a big zoom leaves too few shared features) the frame is chained
through the previous tracked frame. A frame that matches nothing - a cut to a replay, a
close-up, an aerial shot, the camera at the other end - gets no calibration at all, which
the later passes use to skip it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cricsense.calibration.homography import CalibrationError, PitchHomography
from cricsense.config import CameraOptions
from cricsense.io import VideoReader
from cricsense.schema import CalibrationDoc, CameraDoc, CameraFrame, RunInfo, VideoMeta

_LOWE_RATIO = 0.8
# A PTZ camera cannot zoom by more than this between a frame and its reference.
_MAX_ZOOM = 3.0
# Inliers must spread over at least this fraction of the frame. A tight cluster cannot
# constrain a homography, and in practice it is an unmasked on-screen graphic matching
# itself at zero displacement (seen with the tournament logo on real footage).
_MIN_INLIER_SPREAD = 0.05


@dataclass(frozen=True)
class _Features:
    points: np.ndarray  # (N, 2) keypoint positions in full-resolution pixels
    descriptors: np.ndarray | None
    frame_area: float  # full-resolution pixels


@dataclass(frozen=True)
class CameraEstimate:
    view: str | None
    homography: PitchHomography | None  # this frame's pixels -> pitch metres
    inliers: int
    chained: bool = False  # True if found via the previous frame rather than directly


class CameraTracker:
    """Stateful: feed frames in order with estimate()."""

    def __init__(
        self,
        references: dict[str, tuple[np.ndarray, PitchHomography]],
        opts: CameraOptions,
    ) -> None:
        if not references:
            raise ValueError("need at least one reference view")
        self._opts = opts
        self._orb = cv2.ORB_create(nfeatures=opts.orb_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._masks: dict[tuple[int, int], np.ndarray] = {}
        self._refs = {name: (self._features(img), hom) for name, (img, hom) in references.items()}
        # (features, view, frame -> reference homography) of the last tracked frame
        self._prev: tuple[_Features, str, np.ndarray] | None = None
        self._chain_len = 0  # consecutive frames tracked only via the previous frame

    def estimate(self, image: np.ndarray) -> CameraEstimate:
        feats = self._features(image)

        best: tuple[str, np.ndarray, int] | None = None
        for name, (ref_feats, _) in self._refs.items():
            found = self._match(feats, ref_feats)
            if found is not None and (best is None or found[1] > best[2]):
                best = (name, found[0], found[1])
        if best is not None and best[2] >= self._opts.min_inliers:
            estimate = self._finish(feats, best[0], best[1], best[2], chained=False)
            if estimate.view is not None:
                return estimate

        if self._prev is not None and self._chain_len < self._opts.max_chain_frames:
            prev_feats, view, prev_to_ref = self._prev
            found = self._match(feats, prev_feats)
            if found is not None and found[1] >= self._opts.min_inliers:
                return self._finish(feats, view, prev_to_ref @ found[0], found[1], chained=True)

        self._prev = None
        self._chain_len = 0
        return CameraEstimate(view=None, homography=None, inliers=best[2] if best else 0)

    def _finish(
        self, feats: _Features, view: str, frame_to_ref: np.ndarray, inliers: int, *, chained: bool
    ) -> CameraEstimate:
        try:
            homography = self._refs[view][1].for_frame(frame_to_ref)
            homography.check_pitch_visible()
        except CalibrationError:
            self._prev = None
            self._chain_len = 0
            return CameraEstimate(view=None, homography=None, inliers=inliers)
        self._prev = (feats, view, frame_to_ref / frame_to_ref[2, 2])
        self._chain_len = self._chain_len + 1 if chained else 0
        return CameraEstimate(view=view, homography=homography, inliers=inliers, chained=chained)

    def _features(self, image: np.ndarray) -> _Features:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        scale = self._opts.match_scale
        small = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = self._orb.detectAndCompute(small, self._mask(small.shape[:2]))
        points = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2) / scale
        return _Features(points=points, descriptors=descriptors, frame_area=float(gray.shape[0] * gray.shape[1]))

    def _mask(self, shape: tuple[int, int]) -> np.ndarray:
        if shape not in self._masks:
            h, w = shape
            mask = np.full((h, w), 255, np.uint8)
            for x0, y0, x1, y1 in self._opts.overlay_mask:
                mask[int(y0 * h) : int(np.ceil(y1 * h)), int(x0 * w) : int(np.ceil(x1 * w))] = 0
            self._masks[shape] = mask
        return self._masks[shape]

    def _match(self, src: _Features, dst: _Features) -> tuple[np.ndarray, int] | None:
        """Homography taking src pixels to dst pixels, and its RANSAC inlier count."""
        if src.descriptors is None or dst.descriptors is None or len(src.points) < 8 or len(dst.points) < 8:
            return None
        pairs = self._matcher.knnMatch(src.descriptors, dst.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < _LOWE_RATIO * p[1].distance]
        if len(good) < 8:
            return None
        a = src.points[[m.queryIdx for m in good]]
        b = dst.points[[m.trainIdx for m in good]]
        H, inlier_mask = cv2.findHomography(a, b, cv2.RANSAC, self._opts.ransac_px / self._opts.match_scale)
        if H is None or not np.all(np.isfinite(H)):
            return None
        zoom = np.sqrt(abs(np.linalg.det(H[:2, :2] / H[2, 2])))
        if not 1 / _MAX_ZOOM < zoom < _MAX_ZOOM:
            return None
        inliers = a[inlier_mask.ravel() == 1]
        if len(inliers) < 4 or not self._spread_enough(inliers, src.frame_area):
            return None
        return H, len(inliers)

    @staticmethod
    def _spread_enough(points: np.ndarray, frame_area: float) -> bool:
        hull = cv2.convexHull(points.astype(np.float32))
        return cv2.contourArea(hull) >= _MIN_INLIER_SPREAD * frame_area


def read_frames(path: Path, frame_ids: Iterable[int], resize: tuple[int, int] | None = None) -> dict[int, np.ndarray]:
    """The given frames of a video, decoded in one forward pass."""
    wanted = set(frame_ids)
    if not wanted:
        return {}
    with VideoReader(path, end=max(wanted) + 1, resize=resize) as reader:
        found = {f.frame_id: f.image for f in reader if f.frame_id in wanted}
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{path}: frames {sorted(missing)} could not be read")
    return found


def track_video(
    path: Path,
    video: VideoMeta,
    calibration: CalibrationDoc,
    opts: CameraOptions,
    resize: tuple[int, int] | None = None,
) -> CameraDoc:
    """A CameraFrame for every frame of the video."""
    if (video.width, video.height) != tuple(calibration.image_size):
        raise ValueError(
            f"calibration was clicked at {calibration.image_size} but frames are "
            f"{video.width}x{video.height}; recalibrate after changing video.resize"
        )
    ref_images = read_frames(path, (v.frame_id for v in calibration.views), resize)
    tracker = CameraTracker(
        {v.name: (ref_images[v.frame_id], PitchHomography(np.array(v.image_to_pitch))) for v in calibration.views},
        opts,
    )
    frames: list[CameraFrame] = []
    t0 = time.perf_counter()
    with VideoReader(path, resize=resize) as reader:
        for frame in reader:
            est = tracker.estimate(frame.image)
            frames.append(
                CameraFrame(
                    frame_id=frame.frame_id,
                    t=frame.t,
                    view=est.view,
                    image_to_pitch=None if est.homography is None else est.homography.H.tolist(),
                    inliers=est.inliers,
                )
            )
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=len(frames), device="cpu")
    return CameraDoc(video=video, run=run, frames=frames)
