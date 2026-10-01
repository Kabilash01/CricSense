"""Automatic calibration of the main (behind-the-bowler) view: no clicks.

Every pitch has the same painted creases (Laws 6 and 7), so:
  1. white paint: pixels brighter than their neighbourhood and unsaturated;
  2. line candidates: long near-horizontal lines (bowling and popping creases, which the
     broadcast shows running across the pitch) and short steep segments (return creases);
  3. hypotheses: a popping crease at each end, each cut by a pair of return creases, gives
     the four crease corners and so a homography;
  4. each hypothesis is scored by projecting ALL the regulation creases and measuring how
     much of them lands on white paint (a chamfer score), plus whether the striker's stumps
     stand where the middle-stump base projects (a hypothesis that takes the stumps for a
     return crease fits the paint almost as well, but puts the stumps on empty pitch); the
     best is refined by a local search on its four corners.
The score also says whether to trust the result: a frame without a clear pitch view scores
low and is rejected rather than calibrated badly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from cricsense.calibration import pitch_model as pm

TOPHAT_PX = 15
WHITE_TOPHAT = 25
WHITE_MAX_SAT = 70
SIGMA_PX = 2.0
MIN_SCORE = 0.45  # a calibration that explains less of the painted creases than this is rejected
CORNER_NAMES = ("striker_popping_left", "striker_popping_right", "bowler_popping_left", "bowler_popping_right")


@dataclass(frozen=True)
class AutoCalibration:
    image_to_pitch: np.ndarray
    corners: dict[str, tuple[float, float]]  # the four crease corners found in the image
    score: float  # 0..1: how much of the regulation creases lands on white paint


def white_mask(img: np.ndarray, bottom: float = 0.85) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_PX, TOPHAT_PX)))
    mask = ((tophat > WHITE_TOPHAT) & (hsv[..., 1] < WHITE_MAX_SAT)).astype(np.uint8)
    mask[int(bottom * img.shape[0]) :] = 0  # the score bar
    return mask


def pitch_colour(img: np.ndarray) -> np.ndarray:
    """The pitch's dry, pale-brown ground: warm hue, some saturation, fairly bright. Grass is
    too green, boards and crowds are anything but this, so it locates where creases can be."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 8) & (h <= 32) & (s >= 20) & (s <= 150) & (v >= 100)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


def _on_pitch(segs: np.ndarray, ground: np.ndarray, gap: int = 6) -> np.ndarray:
    """Segments with pitch-coloured ground just above and below their middle."""
    h, w = ground.shape
    keep = []
    for x1, y1, x2, y2 in segs:
        u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if gap <= v < h - gap and 0 <= u < w and ground[v - gap, u] and ground[v + gap, u]:
            keep.append((x1, y1, x2, y2))
    return np.array(keep, dtype=segs.dtype).reshape(-1, 4)


def _model_samples(step_m: float = 0.05) -> np.ndarray:
    """Points along every painted crease (not the pitch edges, which are not painted)."""
    pts = []
    for (x0, y0), (x1, y1) in pm.crease_segments():
        if abs(abs(y0) - pm.PITCH_HALF_WIDTH) < 1e-6 and abs(abs(y1) - pm.PITCH_HALF_WIDTH) < 1e-6:
            continue
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step_m))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1))
    return np.concatenate(pts)


MODEL = _model_samples()


STUMP_WEIGHT = 0.3


def stump_evidence(P: np.ndarray, mask: np.ndarray) -> float:
    """White in a thin column rising from the projected striker middle-stump base: the stumps.
    The column's height is set by the image size of the popping-to-bowling crease gap."""
    base = P @ np.array([0.0, 0.0, 1.0])
    front = P @ np.array([pm.POPPING_CREASE, 0.0, 1.0])
    if base[2] <= 0 or front[2] <= 0:
        return 0.0
    u, v = base[:2] / base[2]
    gap = abs(front[1] / front[2] - v)
    h = int(max(8, 2.0 * gap))  # stumps (0.71 m) stand taller than the 1.22 m gap looks, seen from behind
    x0, x1, y0, y1 = int(u - 3), int(u + 4), int(v - h), int(v)
    if x0 < 0 or y0 < 0 or x1 > mask.shape[1] or y1 > mask.shape[0]:
        return 0.0
    col = mask[y0:y1, x0:x1]
    return float(col.any(axis=1).mean()) if col.size else 0.0


def score(image_to_pitch: np.ndarray, dist: np.ndarray, bottom_px: int, mask: np.ndarray | None = None) -> float:
    """Fraction of the regulation creases (visible part) lying on white paint, soft within ~2 px,
    plus the stumps where they should stand."""
    try:
        P = np.linalg.inv(image_to_pitch)
    except np.linalg.LinAlgError:
        return 0.0
    q = P @ np.vstack([MODEL.T, np.ones(len(MODEL))])
    if np.any(q[2] <= 0):
        return 0.0
    u, v = q[0] / q[2], q[1] / q[2]
    h, w = dist.shape
    inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < bottom_px)
    if inside.mean() < 0.6:  # most of the pitch must be in the picture
        return 0.0
    d = dist[v[inside].astype(int), u[inside].astype(int)]
    creases = float(np.mean(np.exp(-(d**2) / (2 * SIGMA_PX**2))) * inside.mean())
    if mask is None:
        return creases
    return (creases + STUMP_WEIGHT * stump_evidence(P, mask)) / (1 + STUMP_WEIGHT)


def _horizontal_lines(mask: np.ndarray, segs: np.ndarray, w: int) -> list[tuple[float, float, float]]:
    """(slope, intercept, support) of near-horizontal lines, merged across segments."""
    cands = []
    for x1, y1, x2, y2 in segs:
        if x2 == x1:
            continue
        k = (y2 - y1) / (x2 - x1)
        if abs(k) < 0.12:
            cands.append((k, y1 - k * x1, float(np.hypot(x2 - x1, y2 - y1)), (x1 + x2) / 2))
    merged: list[list[float]] = []
    for k, c, length, _ in sorted(cands, key=lambda t: -t[2]):
        v_mid = k * w / 2 + c
        for m in merged:
            if abs(m[0] * w / 2 + m[1] - v_mid) < 6 and abs(m[0] - k) < 0.03:
                m[2] += length
                break
        else:
            merged.append([k, c, length])
    return [tuple(m) for m in sorted(merged, key=lambda m: -m[2])]


def _steep_segments(segs: np.ndarray) -> list[tuple[float, float, float, float]]:
    out = []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if 55 < ang < 125 and np.hypot(x2 - x1, y2 - y1) >= 18:
            out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def _cross(line: tuple[float, float, float], seg: tuple[float, float, float, float]) -> tuple[float, float] | None:
    """Where a (steep) segment, extended a little, meets the line v = k u + c."""
    k, c, _ = line
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    den = dy - k * dx
    if abs(den) < 1e-6:
        return None
    t = (k * x1 + c - y1) / den
    length = np.hypot(dx, dy)
    if not (-40 / length <= t <= 1 + 40 / length):  # within 40 px of the painted segment
        return None
    return x1 + t * dx, y1 + t * dy


def calibrate(img: np.ndarray, bottom: float = 0.85) -> AutoCalibration | None:
    h, w = img.shape[:2]
    bottom_px = int(bottom * h)
    mask = white_mask(img, bottom)
    dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=40, minLineLength=18, maxLineGap=6)
    if segs is None:
        return None
    segs = segs.reshape(-1, 4)
    # creases are painted on the pitch: boards, logos and crowds make long white lines too
    lines = _horizontal_lines(mask, _on_pitch(segs, pitch_colour(img)), w)[:8]
    # the far end's return creases are only ~40 px long: a more sensitive pass for them
    short = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=20, minLineLength=14, maxLineGap=4)
    steep = _steep_segments(segs if short is None else np.vstack([segs, short.reshape(-1, 4)]))

    def ends_on(line) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Pairs of return-crease crossings on this line: crossings merged into clusters and
        ranked by how many segments support them (a player's leg rarely lines up twice)."""
        pts = sorted(p for s in steep if (p := _cross(line, s)) is not None and 10 < p[0] < w - 10)
        clusters: list[list[tuple[float, float]]] = []
        for p in pts:
            if clusters and p[0] - clusters[-1][-1][0] < 6:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        top = sorted(clusters, key=len, reverse=True)[:10]
        cents = sorted((float(np.mean([q[0] for q in c])), float(np.mean([q[1] for q in c]))) for c in top)
        return [(p, q) for p, q in itertools.combinations(cents, 2) if 0.06 * w < q[0] - p[0] < 0.6 * w]

    best: tuple[float, np.ndarray, dict] | None = None
    model = np.array([pm.LANDMARKS[n] for n in CORNER_NAMES], np.float64)
    for far, near in itertools.combinations(lines, 2):
        if far[0] * w / 2 + far[1] > near[0] * w / 2 + near[1]:
            far, near = near, far  # the striker's end is higher in the picture
        if (near[0] * w / 2 + near[1]) - (far[0] * w / 2 + far[1]) < 0.15 * h:
            continue
        far_pairs, near_pairs = ends_on(far), ends_on(near)
        for fl, fr in far_pairs:
            for nl, nr in near_pairs:
                wf, wn = fr[0] - fl[0], nr[0] - nl[0]
                if not (0.9 * wf < wn < 3.0 * wf):  # the nearer end looks wider
                    continue
                if abs((fl[0] + fr[0]) / 2 - (nl[0] + nr[0]) / 2) > 0.5 * wn:  # both ends on one pitch
                    continue
                img_pts = np.array([fl, fr, nl, nr], np.float64)
                H, _ = cv2.findHomography(img_pts, model)
                if H is None:
                    continue
                s = score(H, dist, bottom_px, mask)
                if best is None or s > best[0]:
                    best = (s, H, dict(zip(CORNER_NAMES, map(tuple, img_pts))))
    if best is None:
        return None
    s, H, corners = _refine(best, dist, bottom_px, model, mask)
    if s < MIN_SCORE or not _real_camera(H, (w, h)) or _pitch_ground(H, img, bottom_px) < MIN_PITCH_GROUND:
        return None
    return AutoCalibration(H, corners, s)


MIN_PITCH_GROUND = 0.5


def _pitch_ground(image_to_pitch: np.ndarray, img: np.ndarray, bottom_px: int) -> float:
    """Share of the projected pitch (between the creases) that looks like pitch: rejects white
    lines found on a stadium graphic, an advert or a dugout, which can repeat over many frames."""
    P = np.linalg.inv(image_to_pitch)
    corners = []
    for x, y in ((pm.STRIKER_POPPING_X, -pm.PITCH_HALF_WIDTH), (pm.STRIKER_POPPING_X, pm.PITCH_HALF_WIDTH),
                 (pm.BOWLER_POPPING_X, pm.PITCH_HALF_WIDTH), (pm.BOWLER_POPPING_X, -pm.PITCH_HALF_WIDTH)):
        q = P @ np.array([x, y, 1.0])
        if q[2] <= 0:
            return 0.0
        corners.append((q[0] / q[2], q[1] / q[2]))
    poly = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(poly, [np.array(corners, np.int32)], 1)
    poly[bottom_px:] = 0
    area = int(poly.sum())
    if area < 200:
        return 0.0
    return float((pitch_colour(img) & poly).sum()) / area


def _real_camera(image_to_pitch: np.ndarray, size: tuple[int, int]) -> bool:
    """Paint can be matched by a mapping no camera could produce; such a result is rejected."""
    from cricsense.calibration.camera_pose import CameraPose, PoseError

    try:
        CameraPose.from_homography(image_to_pitch, size)
    except (PoseError, np.linalg.LinAlgError, ValueError):
        return False
    return True


def _refine(best, dist, bottom_px, model, mask, steps=(4.0, 2.0, 1.0, 0.5)):
    """Coordinate search on the four corners, keeping any move that raises the score."""
    s, H, corners = best
    pts = np.array([corners[n] for n in CORNER_NAMES], np.float64)
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for axis in (0, 1):
                    for sign in (-1, 1):
                        trial = pts.copy()
                        trial[i, axis] += sign * step
                        Ht, _ = cv2.findHomography(trial, model)
                        if Ht is None:
                            continue
                        st = score(Ht, dist, bottom_px, mask)
                        if st > s + 1e-4:
                            s, H, pts, improved = st, Ht, trial, True
    return s, H, dict(zip(CORNER_NAMES, map(tuple, pts)))


ACCEPT = 0.72  # a frame this well explained by the regulation creases can join a view
STRONG = 0.78  # a view needs two agreeing frames this good, or ...
SUPPORTED = 3  # ... this many agreeing frames (a batter on a crease can fool one or two frames)
SAME_CAMERA_M = 6.0  # calibrations placing the camera this close: the same fixed camera


def find_views(
    frames: dict[int, np.ndarray], bottom: float = 0.85, known_cameras: list[np.ndarray] | None = None
) -> list[tuple[int, AutoCalibration, int]]:
    """Reference views for a video: [(frame_id, calibration, frames agreeing)] best first.

    Frames are calibrated independently; confident ones are grouped by camera (the broadcast
    films each end from a fixed camera, so a view's crease corners barely move between its
    frames), and each group's best frame becomes the reference. A group of one is dropped:
    a single frame is not enough to trust a view."""
    found = [(f, r) for f, img in sorted(frames.items()) if (r := calibrate(img, bottom)) is not None and r.score >= ACCEPT]
    # A broadcast camera turns and zooms but stays put, so calibrations of one camera place it at
    # the same spot whatever the zoom; frames are grouped by that position, not by pixels.
    from cricsense.calibration.camera_pose import CameraPose

    size = next(iter(frames.values())).shape[1::-1] if frames else (1920, 1080)
    groups: list[list[tuple[int, AutoCalibration]]] = []
    centres: list[np.ndarray] = []
    for f, r in sorted(found, key=lambda fr: -fr[1].score):
        c = CameraPose.from_homography(r.image_to_pitch, size).centre
        for g, c0 in zip(groups, centres):
            if np.linalg.norm(c - c0) < SAME_CAMERA_M:
                g.append((f, r))
                break
        else:
            groups.append([(f, r)])
            centres.append(c)
    def known(c: np.ndarray) -> bool:  # where a camera of this match is already known to stand
        return any(np.linalg.norm(c - k) < SAME_CAMERA_M for k in (known_cameras or []))

    return [
        (g[0][0], g[0][1], len(g))
        for g, c in zip(groups, centres)
        if (len(g) >= 2 and all(r.score >= STRONG for _, r in g[:2]))
        or len(g) >= SUPPORTED
        or (known(c) and g[0][1].score >= STRONG)  # a short clip: one strong frame at a known camera
    ]
