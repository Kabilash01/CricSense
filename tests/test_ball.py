"""Ball physics, bounce detection and delivery finding on simulated deliveries with known truth."""

import numpy as np
import pytest

from cricsense.ball.bounce import TrackPoint, find_bounce
from cricsense.ball.delivery import Candidate, FrameGeometry, find_delivery
from cricsense.ball.physics import GRAVITY, fly, release_velocity
from tests.synthetic import CAMERA_CENTRE, camera_projection

FPS = 25.0


def simulate(release, velocity, e=0.55, mu=0.85, seconds=1.2, dt=0.001):
    """Ball positions every dt, with one bounce; returns (times, positions, bounce_time, bounce_point)."""
    X, V = np.array(release, float), np.array(velocity, float)
    ts, xs, bounce = [], [], None
    for k in range(int(seconds / dt)):
        ts.append(k * dt)
        xs.append(X.copy())
        Xn, Vn = fly(X, V, dt, dt)
        if bounce is None and Xn[2] < 0 <= X[2]:
            frac = X[2] / (X[2] - Xn[2])
            bounce = (k * dt + frac * dt, X + (Xn - X) * frac)
            Vn = np.array([Vn[0] * mu, Vn[1] * mu, -Vn[2] * e])
            Xn[2] = -Xn[2] * e
        X, V = Xn, Vn
    return np.array(ts), np.array(xs), bounce[0], bounce[1]


def at_frames(ts, xs, t0, n):
    return [(k, xs[np.argmin(np.abs(ts - k / FPS))]) for k in range(n) if k / FPS <= ts[-1]]


def ground_point(X):
    """Where the camera's sight line through X meets the ground."""
    C = CAMERA_CENTRE
    return C + (X - C) * (C[2] / (C[2] - X[2]))


def test_release_velocity_inverts_the_flight():
    V = np.array([-33.0, 0.4, -2.0])  # ~120 km/h
    B, _ = fly(np.array([18.6, 0.2, 2.1]), V, 0.45)
    assert np.allclose(release_velocity(np.array([18.6, 0.2, 2.1]), B, 0.45), V, atol=0.05)


def test_drag_slows_a_pace_ball_by_the_bounce():
    V = np.array([-39.0, 0.0, -1.0])  # 140 km/h
    _, V_after = fly(np.array([18.6, 0.0, 2.2]), V, 0.4)
    assert 0.85 < np.linalg.norm(V_after) / np.linalg.norm(V) < 0.95


@pytest.mark.parametrize("velocity", [[-23.0, 0.3, -0.5], [-38.0, -0.2, -3.0]])  # spin ~83 km/h, pace ~139 km/h
def test_bounce_found_from_ground_points(velocity):
    ts, xs, tb, B = simulate([18.6, 0.2, 2.1], velocity)
    rng = np.random.default_rng(1)
    track = [
        TrackPoint(k / FPS, *(ground_point(X)[:2] + rng.normal(0, 0.05, 2)))
        for k, X in at_frames(ts, xs, 0, 40)
        if k % 5 != 3  # the detector misses some frames
    ]
    b = find_bounce(track)
    assert b is not None and b.confirmed
    assert abs(b.t - tb) < 1.5 / FPS
    assert abs(b.x - B[0]) < 0.4 and abs(b.y - B[1]) < 0.3


def test_no_bounce_without_the_rise():
    track = [TrackPoint(k / FPS, 6.0 - 0.6 * k, 0.1) for k in range(8)]  # only the early flight
    assert find_bounce(track) is None


def synthetic_candidates(release, velocity, n_frames=40, seed=0):
    ts, xs, tb, B = simulate(release, velocity)
    P = camera_projection()
    H = np.linalg.inv(P[:, [0, 1, 3]])
    geo = {f: FrameGeometry(H / H[2, 2], P) for f in range(n_frames)}
    rng = np.random.default_rng(seed)
    cands = []
    for f, X in at_frames(ts, xs, 0, n_frames):
        if X[0] < -1 or f % 6 == 4:  # past the batter, or missed by the detector
            continue
        p = P @ np.r_[X, 1.0]
        cands.append(Candidate(f, f / FPS, p[0] / p[2] + rng.normal(0, 0.7), p[1] / p[2] + rng.normal(0, 0.7), 0.7))
    for f in range(n_frames):  # a crease mark detected in every frame, and random junk
        p = P @ np.r_[12.0, 1.32, 0.0, 1.0]
        cands.append(Candidate(f, f / FPS, p[0] / p[2], p[1] / p[2], 0.5))
        if rng.random() < 0.3:
            cands.append(Candidate(f, f / FPS, rng.uniform(300, 660), rng.uniform(100, 450), 0.2))
    return cands, geo, tb, B


@pytest.mark.parametrize("velocity", [[-23.0, 0.3, -0.5], [-38.0, -0.2, -3.0]])
def test_delivery_found_with_bounce_and_speed(velocity):
    release = np.array([18.6, 0.2, 2.0])
    cands, geo, tb, B = synthetic_candidates(release, velocity)
    d = find_delivery(cands, geo, FPS)
    assert d is not None and d.bounce is not None
    assert abs(d.bounce.x - B[0]) < 0.5 and abs(d.bounce.y - B[1]) < 0.3
    true_kph = np.linalg.norm(velocity) * 3.6
    # release is only known to the frame; the estimate is honest about that (+-15%)
    assert abs(d.speed_estimate_kph / true_kph - 1) < 0.15
