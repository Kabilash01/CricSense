"""Cricket ball flight: gravity plus quadratic air drag.

Used to turn a release point and a bounce point, each measured with its time, into the
release velocity. A broadcast camera behind the bowler looks almost along the ball's path,
so depth along the flight is poorly observed from a single view (a plain curve fit to the
detections is ill-conditioned, condition number ~1e4 on real footage); the two anchors are
not: the release sight line crosses hand height at the bowler's crease, and at the bounce
the ball is on the ground, where the pitch calibration is accurate to a few pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

GRAVITY = np.array([0.0, 0.0, -9.81])
# 0.5 * air density * drag coefficient * cross-section / mass, for a 156 g, 72 mm ball
# with Cd ~ 0.4: a ball released at 140 km/h loses ~10% of its speed by the bounce.
DRAG_PER_M = 0.0061


def fly(position: np.ndarray, velocity: np.ndarray, seconds: float, dt: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity after `seconds` of flight (no bounce)."""
    X, V = np.asarray(position, float).copy(), np.asarray(velocity, float).copy()
    steps = max(1, int(round(seconds / dt)))
    h = seconds / steps
    for _ in range(steps):
        A = GRAVITY - DRAG_PER_M * np.linalg.norm(V) * V
        X, V = X + V * h + 0.5 * A * h * h, V + A * h
    return X, V


def release_velocity(release: np.ndarray, bounce: np.ndarray, seconds: float) -> np.ndarray:
    """Velocity at release that carries the ball from `release` to `bounce` in `seconds`."""
    if seconds <= 0:
        raise ValueError("the bounce must come after the release")
    guess = (np.asarray(bounce) - np.asarray(release)) / seconds - 0.5 * GRAVITY * seconds
    fit = least_squares(lambda V: fly(release, V, seconds)[0] - bounce, guess)
    return fit.x
