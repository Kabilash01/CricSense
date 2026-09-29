"""Find each delivery in a shot: release, the ball's track, the bounce, a speed estimate.

Input: ball candidates per frame (detector output, pixels) and each frame's calibration.
Steps:
  1. place every candidate where its sight line meets the ground (camera motion removed);
  2. drop static candidates - the same ground spot in 5+ frames is a pitch marking, not a ball;
  3. release: a candidate whose sight line passes through the release zone (hand height at
     the bowler's crease) - the ball has just left the hand;
  4. follow the ball from there, one candidate per frame, staying on the delivery's line;
  5. bounce: bounce.find_bounce on that track;
  6. speed estimate: the drag-and-gravity flight joining release and bounce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cricsense.ball.bounce import Bounce, TrackPoint, find_bounce
from cricsense.ball.physics import release_velocity

RELEASE_HEIGHT_M = 2.0  # hand height at release, ~2.0-2.3 m for most bowlers
RELEASE_ZONE_X = (16.0, 20.5)  # around the bowler's popping crease (18.9 m)
RELEASE_ZONE_Y = 2.5
CORRIDOR_X = (-4.0, 22.0)
CORRIDOR_Y = 3.5
STATIC_RADIUS_M = 0.3
# Pitch markings stay put for the whole shot; the ball's ground point also barely moves for
# ~5 frames around the bounce (it turns there), so only much longer stays count as static.
STATIC_MIN_FRAMES = 10
MAX_FLIGHT_FRAMES = 36  # ~1.4 s at 25 fps: slower than any delivery
MIN_TRACK = 5


@dataclass(frozen=True)
class Candidate:
    frame_id: int
    t: float
    u: float
    v: float
    conf: float


@dataclass(frozen=True)
class FrameGeometry:
    image_to_pitch: np.ndarray  # 3x3
    projection: np.ndarray | None  # 3x4, world metres (z up) -> pixels


@dataclass
class Delivery:
    release_frame: int
    release_t: float
    release_point: np.ndarray  # metres, on the release detection's sight line at hand height
    track: list[tuple[Candidate, float, float]] = field(default_factory=list)  # (candidate, gx, gy)
    bounce: Bounce | None = None
    speed_estimate_kph: float | None = None


def ground_point(image_to_pitch: np.ndarray, u: float, v: float) -> tuple[float, float] | None:
    p = image_to_pitch @ np.array([u, v, 1.0])
    if p[2] <= 0:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def release_point(projection: np.ndarray, u: float, v: float, height: float = RELEASE_HEIGHT_M) -> np.ndarray | None:
    """Where this pixel's sight line is at hand height, if that is in front of the camera."""
    M = projection[:, :3]
    C = -np.linalg.solve(M, projection[:, 3])
    d = np.linalg.solve(M, np.array([u, v, 1.0]))
    if d @ (np.array([10.0, 0.0, 0.0]) - C) < 0:
        d = -d
    if abs(d[2]) < 1e-9:
        return None
    s = (height - C[2]) / d[2]
    return C + s * d if s > 0 else None


def find_delivery(candidates: list[Candidate], geometry: dict[int, FrameGeometry], fps: float) -> Delivery | None:
    """The best-supported delivery among one shot's ball candidates, or None."""
    placed = []
    for c in candidates:
        geo = geometry.get(c.frame_id)
        g = None if geo is None else ground_point(geo.image_to_pitch, c.u, c.v)
        if g is None or not (CORRIDOR_X[0] < g[0] < CORRIDOR_X[1] and abs(g[1]) < CORRIDOR_Y):
            continue
        placed.append((c, g[0], g[1]))
    placed = _drop_static(placed)
    if not placed:
        return None

    best: tuple[float, Delivery] | None = None
    for c0, gx0, gy0 in placed:
        geo = geometry[c0.frame_id]
        if geo.projection is None:
            continue
        R = release_point(geo.projection, c0.u, c0.v)
        if R is None or not (RELEASE_ZONE_X[0] < R[0] < RELEASE_ZONE_X[1] and abs(R[1]) < RELEASE_ZONE_Y):
            continue
        track = _follow(placed, (c0, gx0, gy0))
        if len(track) < MIN_TRACK:
            continue
        bounce = find_bounce([TrackPoint(c.t, gx, gy) for c, gx, gy in track])
        score = len(track) + sum(c.conf for c, _, _ in track) + (3.0 if bounce is not None else 0.0)
        if best is None or score > best[0]:
            # released between the previous frame and this one
            delivery = Delivery(c0.frame_id, c0.t - 0.5 / fps, R, track, bounce)
            best = (score, delivery)
    if best is None:
        return None
    delivery = best[1]
    if delivery.bounce is not None and delivery.bounce.t > delivery.release_t:
        B = np.array([delivery.bounce.x, delivery.bounce.y, 0.0])
        V = release_velocity(delivery.release_point, B, delivery.bounce.t - delivery.release_t)
        delivery.speed_estimate_kph = float(np.linalg.norm(V) * 3.6)
    return delivery


def _drop_static(placed: list[tuple[Candidate, float, float]]) -> list[tuple[Candidate, float, float]]:
    if not placed:
        return placed
    G = np.array([[gx, gy] for _, gx, gy in placed])
    F = np.array([c.frame_id for c, _, _ in placed])
    keep = []
    for (c, gx, gy), g in zip(placed, G):
        near = np.hypot(G[:, 0] - g[0], G[:, 1] - g[1]) < STATIC_RADIUS_M
        if len(set(F[near])) < STATIC_MIN_FRAMES:
            keep.append((c, gx, gy))
    return keep


MAX_GAP_FRAMES = 6
STEP_M = 2.2  # max ground-point movement per frame: ~1 m early, up to ~2 m just after the bounce
LATERAL_M = 0.35  # max sideways movement per frame


def _follow(placed, start) -> list[tuple[Candidate, float, float]]:
    """The best-supported path of candidates from the release onward (dynamic programming).

    Following the nearest candidate frame by frame lets one false detection after a gap
    take over the track. Instead every candidate within MAX_FLIGHT_FRAMES of the release
    gets the best score of any path reaching it: +1 + confidence per detection, minus a
    penalty for jumps, with jumps limited to what a ball's ground point can do per frame.
    """
    c0 = start[0]
    nodes = sorted(
        (p for p in placed if c0.frame_id < p[0].frame_id < c0.frame_id + MAX_FLIGHT_FRAMES),
        key=lambda p: p[0].frame_id,
    )
    nodes = [start] + nodes
    score = [1.0 + c0.conf] + [-np.inf] * (len(nodes) - 1)
    prev = [-1] * len(nodes)
    for j in range(1, len(nodes)):
        cj, xj, yj = nodes[j]
        for i in range(j):
            ci, xi, yi = nodes[i]
            gap = cj.frame_id - ci.frame_id
            if score[i] == -np.inf or not 0 < gap <= MAX_GAP_FRAMES:
                continue
            dx, dy = abs(xj - xi), abs(yj - yi)
            if dx > STEP_M * gap + 0.3 or dy > LATERAL_M * gap + 0.2:
                continue
            s_ = score[i] + 1.0 + cj.conf - 0.3 * (gap - 1) - 0.5 * (dx / (STEP_M * gap)) ** 2
            if s_ > score[j]:
                score[j], prev[j] = s_, i
    end = int(np.argmax(score))
    path = []
    while end != -1:
        path.append(nodes[end])
        end = prev[end]
    return path[::-1]
