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

from cricsense.calibration.camera_pose import CameraPose, PoseError
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

    def reset(self) -> None:
        """Forget the last tracked frame (a cut away from the calibrated view)."""
        self._prev = None
        self._chain_len = 0

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


# Pitch points whose image positions stand in for a frame's homography while smoothing.
_ANCHORS = np.array([[1.22, -1.32], [1.22, 1.32], [18.9, -1.32], [18.9, 1.32]])
_MIN_SEGMENT = 5  # shorter runs of calibrated frames are left as they are


def smooth_homographies(
    frames: list[tuple[int, np.ndarray | None]], window: int, outlier_px: float
) -> tuple[list[np.ndarray | None], int]:
    """Temporally smooth per-frame image->pitch homographies within each unbroken run.

    Each frame is estimated independently against the reference, so consecutive estimates
    jitter by several pixels (and a few are grossly wrong) while the real camera moves
    smoothly. Per run: project four crease corners into every frame, replace frames that
    disagree with their neighbours' median by more than `outlier_px` by interpolation, then
    apply a Savitzky-Golay filter and rebuild each homography from the smoothed corners.
    Returns the smoothed homographies (None where there was none) and the outlier count.
    """
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter

    out: list[np.ndarray | None] = [H for _, H in frames]
    outliers = 0
    runs, run = [], []
    for i, (_, H) in enumerate(frames):
        if H is not None:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)

    for run in runs:
        if len(run) < _MIN_SEGMENT:
            continue
        P = np.stack([_project_inverse(frames[i][1], _ANCHORS) for i in run])  # (T, 4, 2)
        flat = P.reshape(len(run), -1)
        med = median_filter(flat, size=(min(7, len(run)), 1), mode="nearest")
        bad = np.abs(flat - med).max(axis=1) > outlier_px
        outliers += int(bad.sum())
        good = np.where(~bad)[0]
        if len(good) < _MIN_SEGMENT:
            for i in run:
                out[i] = None  # too unreliable to keep at all
            continue
        t = np.arange(len(run))
        flat = np.stack([np.interp(t, good, flat[good, k]) for k in range(flat.shape[1])], axis=1)
        w = min(window, len(run) // 2 * 2 - 1)
        if w >= 5:
            flat = savgol_filter(flat, window_length=w, polyorder=2, axis=0)
        for j, i in enumerate(run):
            img_pts = flat[j].reshape(4, 2).astype(np.float32)
            H = cv2.getPerspectiveTransform(img_pts, _ANCHORS.astype(np.float32))
            out[i] = H / H[2, 2]
    return out, outliers


def _project_inverse(image_to_pitch: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(image_to_pitch)
    p = np.c_[pitch_pts, np.ones(len(pitch_pts))] @ Hi.T
    return p[:, :2] / p[:, 2:3]


def _pose(
    image_to_pitch: np.ndarray, image_size: tuple[int, int], previous: CameraPose | None, reference: CameraPose
) -> CameraPose | None:
    """This frame's full 3D camera: the view's fixed camera position, zoom and rotation fitted
    to this frame's own ground calibration (None if no fit exists).

    Ground positions keep using the 8-parameter calibration directly (~2-4 px on detected
    stump bases). A camera fitted freely per frame wandered 67-106 m within one delivery,
    so sight lines to the flying ball from different frames did not meet; fixing the
    position at the reference camera keeps the stump check (base ~10 px, height ratio
    0.97-0.98) and makes the frames agree with each other.
    """
    try:
        return (previous or reference).with_centre(image_to_pitch, image_size, reference.centre)
    except (PoseError, np.linalg.LinAlgError):
        return None


def _shows_pitch(image: np.ndarray, min_share: float) -> bool:
    """A cheap test on a thumbnail: is there pitch-coloured ground in view at all? Measured on
    the dev clip: skips a third of the frames with the same frames calibrated, 1.4x faster."""
    from cricsense.calibration.auto import pitch_colour

    h = image.shape[0]
    small = cv2.resize(image[: int(0.85 * h)], (480, 230), interpolation=cv2.INTER_AREA)
    return float(pitch_colour(small).mean()) >= min_share


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


def reference_images(
    path: Path, calibration: CalibrationDoc, root: Path, resize: tuple[int, int] | None = None
) -> dict[str, np.ndarray]:
    """Each view's reference frame, read from the view's own source video where it has one."""
    by_source: dict[Path, list] = {}
    for v in calibration.views:
        source = path if v.source_video is None else (root / v.source_video)
        by_source.setdefault(source, []).append(v)
    images: dict[str, np.ndarray] = {}
    for source, views in by_source.items():
        if not source.is_file():
            raise ValueError(f"reference video {source} for views {[v.name for v in views]} not found")
        frames = read_frames(source, (v.frame_id for v in views), resize)
        images |= {v.name: frames[v.frame_id] for v in views}
    return images


def track_video(
    path: Path,
    video: VideoMeta,
    calibration: CalibrationDoc,
    opts: CameraOptions,
    resize: tuple[int, int] | None = None,
    root: Path | None = None,
) -> CameraDoc:
    """A CameraFrame for every frame of the video."""
    if (video.width, video.height) != tuple(calibration.image_size):
        raise ValueError(
            f"calibration was clicked at {calibration.image_size} but frames are "
            f"{video.width}x{video.height}; recalibrate after changing video.resize"
        )
    ref_images = reference_images(path, calibration, root or Path.cwd(), resize)
    tracker = CameraTracker(
        {v.name: (ref_images[v.name], PitchHomography(np.array(v.image_to_pitch))) for v in calibration.views},
        opts,
    )
    raw: list[tuple[int, float, CameraEstimate]] = []
    t0 = time.perf_counter()
    with VideoReader(path, resize=resize) as reader:
        for frame in reader:
            if opts.gate_min_pitch > 0 and not _shows_pitch(frame.image, opts.gate_min_pitch):
                # crowd, close-up, graphic: cannot be a calibrated view, so skip the feature matching
                tracker.reset()
                raw.append((frame.frame_id, frame.t, CameraEstimate(view=None, homography=None, inliers=0)))
                continue
            raw.append((frame.frame_id, frame.t, tracker.estimate(frame.image)))

    smoothed, _ = smooth_homographies(
        [(fid, None if e.homography is None else e.homography.H) for fid, _, e in raw],
        opts.smooth_window,
        opts.outlier_px,
    )
    size = (video.width, video.height)
    references = {v.name: CameraPose.from_homography(np.array(v.image_to_pitch), size) for v in calibration.views}
    frames = []
    prev_pose: CameraPose | None = None
    prev_view: str | None = None
    for (fid, t, est), H in zip(raw, smoothed):
        view = est.view if H is not None else None
        seed = prev_pose if view is not None and view == prev_view else None
        pose = None if view is None else _pose(H, size, seed, references[view])
        prev_pose, prev_view = pose, view
        frames.append(
            CameraFrame(
                frame_id=fid,
                t=t,
                view=est.view if H is not None else None,
                image_to_pitch=None if H is None else H.tolist(),
                projection=None if pose is None else pose.P.tolist(),
                inliers=est.inliers,
            )
        )
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=len(frames), device="cpu")
    return CameraDoc(video=video, run=run, frames=frames)
