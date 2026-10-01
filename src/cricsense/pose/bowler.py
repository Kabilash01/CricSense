"""The release, timed from the bowler's arm.

The bowling hand rises to about a third of a body height above the shoulders and, within a
frame or two of letting the ball go, whips down below them. The release is at the top of
that swing, just before the drop - visible even when the ball itself is blurred against the
bowler, or lost in the first frames of its flight. On the dev clip it agrees with the
ball-track release to within one frame on 7 of 8 deliveries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10
HIGH = 0.18  # hand this far above the shoulders (body heights): the arm at the top
LOW = -0.15  # and this far below them right after: the arm has come over
DROP_FRAMES = 3


@dataclass(frozen=True)
class ArmSample:
    frame: int
    height: float  # the higher hand above the shoulders, in body heights
    hand_uv: tuple[float, float]


def arm_sample(frame: int, kp: np.ndarray, box_height: float) -> ArmSample | None:
    wrists = [kp[j] for j in (L_WRIST, R_WRIST) if kp[j, 2] > 0.4]
    shoulders = [kp[j] for j in (L_SHOULDER, R_SHOULDER) if kp[j, 2] > 0.4]
    if not wrists or not shoulders or box_height <= 0:
        return None
    top = min(wrists, key=lambda w: w[1])  # image y grows downward
    height = (float(np.mean([s[1] for s in shoulders])) - float(top[1])) / box_height
    return ArmSample(frame, height, (float(top[0]), float(top[1])))


def swing_tops(samples: list[ArmSample]) -> list[ArmSample]:
    """Every top-of-swing: a hand high above the shoulders, below them within DROP_FRAMES."""
    samples = sorted(samples, key=lambda s: s.frame)
    tops = []
    for i, s in enumerate(samples):
        if s.height < HIGH:
            continue
        nxt = [t for t in samples[i + 1 :] if t.frame - s.frame <= DROP_FRAMES]
        if nxt and max(t.height for t in nxt) <= s.height and any(t.height <= LOW for t in nxt):
            if not tops or s.frame - tops[-1].frame > DROP_FRAMES:
                tops.append(s)
    return tops


def release_from_arm(samples: list[ArmSample], before: int | None = None) -> tuple[float, tuple[float, float]] | None:
    """(release frame, fractional; the hand's image position then) or None.

    The bound before the delivery stride swings an arm up and down too, so the release is the
    LAST top-of-swing - before `before` (the first sighting of the ball in flight) when given.
    The ball leaves the hand between that frame and the next."""
    tops = [t for t in swing_tops(samples) if before is None or t.frame <= before]
    if not tops:
        return None
    top = tops[-1]
    return top.frame + 0.5, top.hand_uv
