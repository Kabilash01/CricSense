"""Bounce from the ball's track in ground coordinates.

Each detection is placed where its sight line meets the ground (the pitch calibration),
which removes camera pans and zooms. That point is not the ball: while the ball is in the
air it lands ahead of it, by roughly (height x camera distance / camera height). So as the
ball descends toward the pitch the ground point moves back toward the ball (rising in x),
meets it at the bounce, and jumps ahead again as the ball climbs off the pitch (falling in
x). The bounce is the apex of that turn - the only moment the ground point IS the ball -
found where a line through the rising side meets one through the falling side. If the ball
is hit or hidden right after pitching, the last rising point stands in and the bounce is
marked unconfirmed.

Image y increases downward; that convention is not used here - everything is in pitch
metres (x from the striker's stumps toward the bowler).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_RISE = 2  # rising points needed before the apex
_FIT_FRAMES = 4  # frames on each side used for the two lines
_MIN_RISE_RATE = 1.0  # m/s: the ground point must genuinely climb toward the ball
_TREND_TOLERANCE_M = 0.35  # m per frame above the rise's own trend before a point is rejected
_NOISE_DROP_PER_FRAME = 0.35  # m: a smaller dip is detection noise, a larger one is the bounce
_FPS_GUESS = 25.0


@dataclass(frozen=True)
class Bounce:
    t: float  # seconds
    x: float  # metres from the striker's stumps (the length)
    y: float  # metres across the pitch (the line)
    confirmed: bool  # False: the ball was not seen after pitching


@dataclass(frozen=True)
class TrackPoint:
    t: float
    gx: float  # ground point of the detection's sight line, metres
    gy: float


def find_bounce(track: list[TrackPoint]) -> Bounce | None:
    """Bounce of one delivery, from its track (release onward, one point per frame)."""
    pts = sorted(track, key=lambda p: p.t)
    if len(pts) < _MIN_RISE + 2:
        return None
    gx = np.array([p.gx for p in pts])
    t = np.array([p.t for p in pts])
    # The dip before the rise: the lowest point of the initial descent. Not the lowest point of
    # the whole track - after the bounce the ground point drops further still.
    i_min = 0
    for i in range(1, len(pts)):
        if gx[i] < gx[i_min]:
            i_min = i
        elif gx[i] > gx[i_min] + 0.25:  # a pace ball rises only ~0.5 m before pitching
            break
    if i_min >= len(pts) - _MIN_RISE:
        return None
    # Follow the rise after the dip along its own trend. The rise accelerates toward the
    # bounce (0.3-0.8 m per frame measured), so there is no fixed step limit: a point well
    # above the trend is a false detection and is skipped (after the bounce the ball can
    # only move back toward the batter, never further up the pitch); a clear drop below the
    # last accepted point means the ball has bounced, and ends the rise.
    rise = [i_min]
    for i in range(i_min + 1, len(pts)):
        last = rise[-1]
        frames = max((t[i] - t[last]) * _FPS_GUESS, 1.0)
        slope = (gx[last] - gx[rise[-2]]) / max((t[last] - t[rise[-2]]) * _FPS_GUESS, 1.0) if len(rise) >= 2 else 0.4
        predicted = gx[last] + max(slope, 0.0) * frames
        if gx[i] > predicted + _TREND_TOLERANCE_M * frames + 0.2:
            continue  # far above the trend: not the ball
        if gx[i] < gx[last] - _NOISE_DROP_PER_FRAME * frames:
            break  # the ball has pitched and is climbing again
        rise.append(i)
    i_apex = max(rise, key=lambda i: gx[i])
    if len(rise) - 1 < _MIN_RISE or gx[i_apex] - gx[i_min] < 0.3:
        return None
    keep = np.zeros(len(pts), bool)
    keep[rise] = True
    keep[i_apex + 1 :] = True  # the right side of the apex is judged by the drop test below
    dt_frame = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
    left = keep & (t > t[i_apex] - _FIT_FRAMES * dt_frame - 1e-9) & (t <= t[i_apex] + 1e-9) & (np.arange(len(t)) >= i_min)
    right = (t > t[i_apex] + 1e-9) & (t <= t[i_apex] + _FIT_FRAMES * dt_frame + 1e-9)
    if left.sum() < 2:
        return None
    a_l, b_l = np.polyfit(t[left], gx[left], 1)
    if a_l < _MIN_RISE_RATE:
        return None
    confirmed = right.sum() >= 1 and gx[right].min() < gx[i_apex] - 0.3
    t_b = t[i_apex]
    if right.sum() >= 2:
        a_r, b_r = np.polyfit(t[right], gx[right], 1)
        if a_r < a_l - 1e-6:
            cross = (b_r - b_l) / (a_l - a_r)
            # the true bounce can fall between frames, or on a frame the detector missed
            if t[i_apex] - dt_frame <= cross <= t[i_apex] + 1.6 * dt_frame:
                t_b = float(cross)
    x_b = float(a_l * t_b + b_l) if t_b != t[i_apex] else float(gx[i_apex])
    ys = [p.gy for p in pts[max(i_apex - 1, 0) : i_apex + 2]]
    return Bounce(t=float(t_b), x=x_b, y=float(np.mean(ys)), confirmed=bool(confirmed))
