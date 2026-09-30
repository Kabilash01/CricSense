"""What the striker's body did on a delivery, from COCO-17 keypoints.

footwork   the feet on the calibrated pitch at the shot against the stance before release:
           the front foot (the one nearer the bowler) forward -> "front foot"; a foot back
           towards the stumps -> "back foot"; neither -> "in the crease"
swing      the wrists' speed relative to the hips (so a panning, tilting camera cancels):
           fast -> the ball was played, slow -> left alone

Foot positions use the ankle keypoint moved down to the ground; only frames where the
calibration exists are used, so every distance is in metres on the pitch.

The broadcast zooms in on the batter after release, and in the close-up the feet often leave
the picture or hide behind the pads; the pose model then puts "ankles" on the shins. So a
foot is only trusted when its ankle is confident, inside the picture above the score bar,
and clearly below its knee. A stride longer than MAX_STRIDE_M is a misread foot, not a
batter charging. (Knee-on-the-ground was tried and dropped: from in front, a bent knee over
a planted foot looks the same as a knee on the ground - see docs/decisions.md.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16, 9, 10
MIN_CONF = 0.5
FOOT_CONF = 0.7
FOOT_DROP = 0.02  # the ground contact is a little below the ankle keypoint, in body heights
KNEE_OF = {L_ANKLE: L_KNEE, R_ANKLE: R_KNEE}
BOTTOM_MARGIN_PX = 8
MAX_STRIDE_M = 1.5
MIN_FOOT_FRAMES = 3


@dataclass(frozen=True)
class Thresholds:
    stride_front_m: float = 0.30
    stride_back_m: float = 0.20
    swing_speed: float = 1.6


@dataclass(frozen=True)
class Observation:
    frame: int
    kp: np.ndarray  # (17, 3)
    box: tuple[float, float, float, float]
    image_to_pitch: np.ndarray | None
    visible_bottom: float = 918.0  # the lowest image row not covered by the score bar

    def ankle_ok(self, j: int) -> bool:
        """A real, visible foot: confident, in the picture, and below its knee."""
        k = KNEE_OF[j]
        return bool(
            self.kp[j, 2] >= FOOT_CONF
            and self.kp[j, 1] < self.visible_bottom - BOTTOM_MARGIN_PX
            and self.kp[k, 2] >= MIN_CONF
            and self.kp[j, 1] - self.kp[k, 1] > 0.12 * self.height
        )

    @property
    def height(self) -> float:
        return max(self.box[3] - self.box[1], 1.0)

    def foot(self, j: int) -> np.ndarray | None:
        """Pitch position (x, y) of ankle j's ground contact, or None."""
        if self.image_to_pitch is None or not self.ankle_ok(j):
            return None
        q = self.image_to_pitch @ np.array([self.kp[j, 0], self.kp[j, 1] + FOOT_DROP * self.height, 1.0])
        return q[:2] / q[2]


def footwork(obs: list[Observation], release: int, shot: int, th: Thresholds) -> tuple[str | None, float | None]:
    before = [o for o in obs if o.frame <= release]
    at_shot = [o for o in obs if shot - 2 <= o.frame <= shot + 6]
    feet: dict[str, dict[int, np.ndarray]] = {"stance": {}, "shot": {}}
    for name, group in (("stance", before), ("shot", at_shot)):
        for j in (L_ANKLE, R_ANKLE):
            pts = [p for o in group if (p := o.foot(j)) is not None]
            if len(pts) >= (2 if name == "stance" else MIN_FOOT_FRAMES):
                feet[name][j] = np.median(pts, axis=0)
    if len(feet["stance"]) < 2 or len(feet["shot"]) < 2:
        return None, None
    front = max(feet["stance"], key=lambda j: feet["stance"][j][0])  # nearer the bowler: larger x
    back = L_ANKLE if front == R_ANKLE else R_ANKLE
    moved_front = float(feet["shot"][front][0] - feet["stance"][front][0])
    moved_back = float(feet["shot"][back][0] - feet["stance"][back][0])
    if max(abs(moved_front), abs(moved_back)) > MAX_STRIDE_M:
        return None, None
    if moved_front >= th.stride_front_m:
        return "front foot", round(moved_front, 2)
    if min(moved_back, moved_front) <= -th.stride_back_m:
        return "back foot", round(min(moved_back, moved_front), 2)
    return "in the crease", round(moved_front, 2)


def swing(obs: list[Observation], release: int, fps: float, th: Thresholds) -> tuple[str | None, int | None, float | None]:
    """(played | left, peak frame, peak wrist speed in body heights per second)."""
    track = []
    for o in obs:
        if o.frame < release:
            continue
        w = [o.kp[j, :2] for j in (L_WRIST, R_WRIST) if o.kp[j, 2] >= 0.3]
        h = [o.kp[j, :2] for j in (L_HIP, R_HIP) if o.kp[j, 2] >= 0.3]
        if w and h:
            track.append((o.frame, (np.mean(w, axis=0) - np.mean(h, axis=0)) / o.height))
    if len(track) < 4:
        return None, None, None
    speeds = [
        (f1, float(np.linalg.norm(p1 - p0)) / (f1 - f0) * fps)
        for (f0, p0), (f1, p1) in zip(track, track[1:])
        if 0 < f1 - f0 <= 2
    ]
    if not speeds:
        return None, None, None
    peak_frame, peak = max(speeds, key=lambda s: s[1])
    return ("played" if peak >= th.swing_speed else "left"), peak_frame, round(peak, 2)


HAND_MARGIN_M = 0.08  # the feet this far either side of middle stump to call it


def batting_hand(obs: list[Observation], release: int) -> str | None:
    """"right" or "left": in the stance a right-hander's feet are on the camera's right of middle
    stump (pitch y > 0 from behind the bowler), a left-hander's on the left."""
    ys = []
    for o in obs:
        if o.frame <= release:
            feet = [p for j in (L_ANKLE, R_ANKLE) if (p := o.foot(j)) is not None]
            if feet:
                ys.append(float(np.mean([p[1] for p in feet])))
    if not ys:
        return None
    y = float(np.median(ys))
    return "right" if y > HAND_MARGIN_M else "left" if y < -HAND_MARGIN_M else None
