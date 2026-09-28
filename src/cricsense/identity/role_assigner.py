"""Cricket roles from pitch position plus detector evidence, smoothed over time.

Replaces `infer_role_from_class` (B3), which took the role straight from each frame's YOLO
label. A detector cannot tell a striker from a non-striker by appearance, and its label
flickers; where a player stands is decisive. Here each player accumulates evidence from
both over time, and every frame the unique roles (striker, non-striker, bowler, keeper)
are handed out jointly, so two players can never both be the striker.

Pitch frame: x along the pitch from the striker's stumps (0) to the bowler's (20.12),
y across it (see calibration/pitch_model.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from cricsense.calibration import pitch_model as pm
from cricsense.schema import Role

ROLES = (Role.STRIKER, Role.NON_STRIKER, Role.BOWLER, Role.KEEPER, Role.UMPIRE, Role.FIELDER)
_UNIQUE = (Role.STRIKER, Role.NON_STRIKER, Role.BOWLER, Role.KEEPER)
_MAX_UMPIRES = 2  # bowler's end and square leg

# Detector class -> which roles it supports. "Batsman" cannot say which end.
CLASS_EVIDENCE: dict[str, dict[Role, float]] = {
    "Batsman": {Role.STRIKER: 1.0, Role.NON_STRIKER: 1.0},
    "Bowler": {Role.BOWLER: 1.0},
    "Umpire": {Role.UMPIRE: 1.0},
    "Wicket_Keeper": {Role.KEEPER: 1.0},
    "Player": {Role.FIELDER: 1.0},
}


def _gauss(value: float, centre: float, width: float) -> float:
    return float(np.exp(-0.5 * ((value - centre) / width) ** 2))


def geometry_evidence(x: float, y: float) -> dict[Role, float]:
    """How well a foot point fits each role's usual position, 0..1 per role."""
    ay = abs(y)
    evidence = {
        # at the striker's popping crease, level with the stumps
        Role.STRIKER: _gauss(x, pm.STRIKER_POPPING_X, 1.2) * _gauss(ay, 0.3, 0.9),
        # behind the striker's stumps: standing up to spin (~1.5 m back) or back to pace
        # (~15-20 m back, among the slips; the keeper is the one nearest the middle line)
        Role.KEEPER: max(_gauss(x, -1.5, 1.5), _gauss(x, -16.0, 5.0)) * _gauss(ay, 0.0, 1.2),
        # beside the bowler's-end crease, off the middle line
        Role.NON_STRIKER: _gauss(x, pm.BOWLER_POPPING_X - 0.5, 1.2) * _gauss(ay, 1.6, 0.8),
        # at the bowler's crease (delivery stride and follow-through); the run-up beyond it only
        # counts when the player is actually running (motion_evidence), since the umpire stands there
        Role.BOWLER: _gauss(x, pm.BOWLER_POPPING_X, 2.0) * _gauss(ay, 0.6, 1.2),
        # 1-5 m behind the bowler's stumps, on the middle line (measured: 21-23 m and 24.5 m)
        Role.UMPIRE: _gauss(x, pm.BOWLER_STUMPS_X + 3.0, 1.8) * _gauss(ay, 0.0, 0.8),
    }
    # anyone else on the field; weak everywhere so a good positional fit wins
    evidence[Role.FIELDER] = 0.25
    return evidence


@dataclass(frozen=True)
class RoleOptions:
    geometry_weight: float = 0.6
    class_weight: float = 0.4
    motion_weight: float = 0.5
    memory: float = 0.9  # per-frame decay of accumulated evidence (~10-frame horizon)
    speed_window_s: float = 0.5  # speed is displacement over this long: frame-to-frame noise
    fps: float = 25.0            # (~0.2 m) times 25 fps would look like 5 m/s


def motion_evidence(x: float, speed_mps: float) -> dict[Role, float]:
    """At the bowler's end the bowler runs in and the umpire stands still: position alone
    cannot separate them (both stand around x = 20-25 m), movement can."""
    if x < pm.BOWLER_POPPING_X - 4.0:
        return {}
    # below ~1 m/s is standing still: measurement noise and the odd shuffle
    moving = float(np.clip((speed_mps - 1.0) / 3.0, 0.0, 1.0))
    return {Role.BOWLER: moving, Role.UMPIRE: -moving}


@dataclass
class RoleAssigner:
    options: RoleOptions = field(default_factory=RoleOptions)

    def __post_init__(self) -> None:
        self._scores: dict[int, np.ndarray] = {}
        self._track: dict[int, deque] = {}  # recent pitch positions per player

    def update(
        self, observations: dict[int, tuple[tuple[float, float] | None, str | None, float]]
    ) -> dict[int, Role]:
        """observations: sid -> (pitch foot point or None, detector class or None, class confidence)."""
        opts = self.options
        for sid, (pitch_xy, cls, conf) in observations.items():
            frame_ev = np.zeros(len(ROLES))
            if pitch_xy is not None:
                geo = geometry_evidence(*pitch_xy)
                frame_ev += opts.geometry_weight * np.array([geo[r] for r in ROLES])
                n = max(2, round(opts.speed_window_s * opts.fps) + 1)
                track = self._track.setdefault(sid, deque(maxlen=n))
                track.append(pitch_xy)
                if len(track) == n:
                    (x0, y0), (x1, y1) = track[0], track[-1]
                    speed = float(np.hypot(x1 - x0, y1 - y0)) / opts.speed_window_s
                    mot = motion_evidence(pitch_xy[0], speed)
                    frame_ev += opts.motion_weight * np.array([mot.get(r, 0.0) for r in ROLES])
            if cls in CLASS_EVIDENCE:
                ev = CLASS_EVIDENCE[cls]
                frame_ev += opts.class_weight * conf * np.array([ev.get(r, 0.0) for r in ROLES])
            prev = self._scores.get(sid, np.zeros(len(ROLES)))
            self._scores[sid] = opts.memory * prev + (1 - opts.memory) * frame_ev
        return self._assign(list(observations))

    def _assign(self, sids: list[int]) -> dict[int, Role]:
        if not sids:
            return {}
        scores = np.stack([self._scores[s] for s in sids])
        roles: dict[int, Role] = {}

        # Unique roles: one player each, chosen jointly to maximise total evidence.
        uniq = [ROLES.index(r) for r in _UNIQUE]
        sub = scores[:, uniq]
        rows, cols = linear_sum_assignment(-sub)
        for i, j in zip(rows, cols):
            role = _UNIQUE[j]
            # Only take a unique role if it beats being a fielder or umpire for this player.
            if sub[i, j] > max(scores[i, ROLES.index(Role.FIELDER)], scores[i, ROLES.index(Role.UMPIRE)]):
                roles[sids[i]] = role

        # Umpires: the strongest remaining umpire candidates, then everyone else fields.
        rest = [i for i, s in enumerate(sids) if s not in roles]
        ump = ROLES.index(Role.UMPIRE)
        fielder = ROLES.index(Role.FIELDER)
        by_umpire = sorted(rest, key=lambda i: -scores[i, ump])
        for rank, i in enumerate(by_umpire):
            is_umpire = rank < _MAX_UMPIRES and scores[i, ump] > scores[i, fielder]
            roles[sids[i]] = Role.UMPIRE if is_umpire else Role.FIELDER
        return roles

    def forget(self, sid: int) -> None:
        self._scores.pop(sid, None)
        self._track.pop(sid, None)
