"""Stable player IDs on top of the tracker's IDs.

Replaces CricSense-AI src/core/player_registry.py, which:
  B1  initialised `jersey_logged = set` (the class, not a set), so the first jersey lock
      crashed with "argument of type 'type' is not iterable";
  B2  only re-linked a new tracker ID to a lost player whose role - frozen at first
      sighting from the YOLO class - was identical, and matched by raw pixel distance.
      Every tracker ID churn next to a class flip minted a new ID: 534 IDs for ~13 people.

Here the detector class plays no part in matching. A new tracker ID is linked to a lost
player by where they stand on the pitch (metres, immune to camera pans and zooms) when the
frame is calibrated, or by box overlap across a short gap when it is not. Candidates are
assigned globally (Hungarian), and nothing is linked across a camera cut in image space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RegistryOptions:
    max_lost_frames: int = 75  # a lost player can be re-linked for this long (3 s at 25 fps)
    image_gap_frames: int = 10  # ...but by image position only across gaps this short
    pitch_gate_m: float = 1.5  # base radius for pitch-space re-links
    run_speed_mps: float = 9.0  # allowance for movement while lost
    min_iou: float = 0.2
    centre_gate_heights: float = 0.75  # image-space gate: centre distance / box height


@dataclass(frozen=True)
class Detection:
    track_id: int  # the tracker's (raw) ID
    box: Box
    pitch_xy: tuple[float, float] | None = None  # foot point in metres, if the frame is calibrated


@dataclass
class _Player:
    sid: int
    box: Box
    frame_id: int
    pitch_xy: tuple[float, float] | None
    pitch_frame: int | None  # when pitch_xy was last measured
    segment: int  # camera segment (between cuts) the player was last seen in


@dataclass
class PlayerRegistry:
    fps: float
    options: RegistryOptions = field(default_factory=RegistryOptions)

    def __post_init__(self) -> None:
        self._players: dict[int, _Player] = {}
        self._raw_to_sid: dict[int, int] = {}
        self._next_sid = 1
        self._segment = 0

    @property
    def n_players(self) -> int:
        return self._next_sid - 1

    def camera_moved(self, prev_to_cur: np.ndarray) -> None:
        """Warp remembered image boxes by the camera's own motion (3x3, previous -> current pixels).

        Without this, a fast zoom makes a player's new box overlap nothing they left behind,
        so a track the tracker loses during the zoom cannot be re-linked in image space.
        """
        for player in self._players.values():
            x1, y1, x2, y2 = player.box
            corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]]) @ prev_to_cur.T
            corners = corners[:, :2] / corners[:, 2:3]
            player.box = (
                float(corners[:, 0].min()),
                float(corners[:, 1].min()),
                float(corners[:, 0].max()),
                float(corners[:, 1].max()),
            )

    def camera_cut(self) -> None:
        """Call when the view changes; image positions before and after are unrelated."""
        self._segment += 1
        self._raw_to_sid.clear()  # the tracker's IDs do not survive a cut either

    def update(self, frame_id: int, detections: list[Detection]) -> dict[int, int]:
        """Stable ID for every detection in this frame, keyed by tracker ID."""
        result: dict[int, int] = {}
        new: list[Detection] = []
        for det in detections:
            sid = self._raw_to_sid.get(det.track_id)
            if sid is not None and sid in self._players:
                result[det.track_id] = sid
            else:
                new.append(det)

        taken = set(result.values())
        lost = [
            p
            for p in self._players.values()
            if p.sid not in taken and 0 < frame_id - p.frame_id <= self.options.max_lost_frames
        ]
        for det, sid in self._link(frame_id, new, lost).items():
            result[det.track_id] = sid
        for det in new:
            if det.track_id not in result:
                result[det.track_id] = self._next_sid
                self._next_sid += 1

        for det in detections:
            self._remember(result[det.track_id], det, frame_id)
            self._raw_to_sid[det.track_id] = result[det.track_id]
        self._forget_stale(frame_id)
        return result

    def _link(self, frame_id: int, new: list[Detection], lost: list[_Player]) -> dict[Detection, int]:
        if not new or not lost:
            return {}
        cost = np.full((len(new), len(lost)), np.inf)
        for i, det in enumerate(new):
            for j, player in enumerate(lost):
                cost[i, j] = self._cost(frame_id, det, player)
        if not np.isfinite(cost).any():
            return {}
        finite = np.where(np.isfinite(cost), cost, 1e9)
        rows, cols = linear_sum_assignment(finite)
        return {new[i]: lost[j].sid for i, j in zip(rows, cols) if np.isfinite(cost[i, j])}

    def _cost(self, frame_id: int, det: Detection, player: _Player) -> float:
        """0..1 for a plausible re-link (lower is better), inf when it must not happen."""
        opts = self.options
        if det.pitch_xy is not None and player.pitch_xy is not None and player.pitch_frame is not None:
            gap_s = (frame_id - player.pitch_frame) / self.fps
            gate = opts.pitch_gate_m + opts.run_speed_mps * gap_s
            dist = float(np.hypot(det.pitch_xy[0] - player.pitch_xy[0], det.pitch_xy[1] - player.pitch_xy[1]))
            return dist / gate if dist <= gate else np.inf
        if player.segment != self._segment or frame_id - player.frame_id > opts.image_gap_frames:
            return np.inf
        overlap = _iou(det.box, player.box)
        height = max(player.box[3] - player.box[1], 1.0)
        centre = float(np.hypot(*(np.array(_centre(det.box)) - _centre(player.box)))) / height
        if overlap < opts.min_iou and centre > opts.centre_gate_heights:
            return np.inf
        return min(1.0 - overlap, centre / opts.centre_gate_heights)

    def _remember(self, sid: int, det: Detection, frame_id: int) -> None:
        player = self._players.get(sid)
        if player is None:
            player = _Player(sid, det.box, frame_id, None, None, self._segment)
            self._players[sid] = player
        player.box, player.frame_id, player.segment = det.box, frame_id, self._segment
        if det.pitch_xy is not None:
            player.pitch_xy, player.pitch_frame = det.pitch_xy, frame_id

    def _forget_stale(self, frame_id: int) -> None:
        stale = {sid for sid, p in self._players.items() if frame_id - p.frame_id > self.options.max_lost_frames}
        self._raw_to_sid = {raw: sid for raw, sid in self._raw_to_sid.items() if sid not in stale}
        for sid in stale:
            del self._players[sid]


def _centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
