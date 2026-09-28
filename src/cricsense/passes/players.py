"""`cricsense players`: detect, track, stable IDs, roles, identities -> players.json.

Per processed frame:
  detector (players_best.pt) -> tracker (BoT-SORT or ByteTrack) -> foot point on the pitch
  (camera.json) -> stable ID (registry) -> role (geometry + detector evidence) -> name
  (score bar: striker, non-striker, bowler).
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from cricsense.config import Config, video_id_from_path
from cricsense.identity.registry import Detection, PlayerRegistry, RegistryOptions
from cricsense.identity.role_assigner import RoleAssigner, RoleOptions
from cricsense.io import VideoReader, read_doc, write_doc
from cricsense.schema import CameraDoc, PlayerFrame, PlayersDoc, Role, RunInfo, ScoreboardDoc, ScoreboardReading
from cricsense.track.player_tracker import PlayerTracker

log = logging.getLogger("cricsense.players")

# The bar keeps showing the pre-ball state until the ball is over, so the latest reading
# applies for a whole ball (~30-45 s between deliveries including replays).
_SCOREBOARD_MAX_AGE_S = 45.0
_SCOREBOARD_CONSENSUS = 5  # names are voted over this many latest readings (OCR slips)
_EDGE_PX = 4  # a box within this of the frame edge or an on-screen graphic may hide the feet


@dataclass
class PlayersResult:
    doc: PlayersDoc
    path: Path
    tracker_ids: int
    frames_skipped: int
    gmc_calibrated: int
    gmc_fallback: int


class IdentityVoter:
    """Names from the score bar, voted per stable ID, then made unique within each shot.

    A player earns a vote for a name on every frame they hold the role the score bar names
    (striker, non-striker, bowler). Role assignment wobbles for a few frames, so at the end
    each name goes only to the player in that shot with the most votes for it; the share of
    that player's frames that voted for it is the confidence.
    """

    def __init__(self) -> None:
        self._votes: dict[int, Counter] = defaultdict(Counter)
        self._frames: Counter = Counter()
        self._shot: dict[int, int] = {}

    def vote(self, sid: int, shot: int, name: str | None) -> None:
        self._frames[sid] += 1
        self._shot.setdefault(sid, shot)
        if name:
            self._votes[sid][name] += 1

    def merge(self, old: int, new: int) -> None:
        """Fold `new`'s votes into `old` (the same person, split by a tracker handover)."""
        self._votes[old].update(self._votes.pop(new, Counter()))
        self._frames[old] += self._frames.pop(new, 0)
        self._shot.pop(new, None)

    def resolve(self) -> dict[int, tuple[str, float]]:
        best: dict[tuple[int, str], tuple[int, int]] = {}  # (shot, name) -> (votes, sid)
        for sid, votes in self._votes.items():
            for name, n in votes.items():
                key = (self._shot[sid], name)
                if n > best.get(key, (0, -1))[0]:
                    best[key] = (n, sid)
        return {sid: (name, round(n / self._frames[sid], 3)) for (_, name), (n, sid) in best.items()}


def _names_by_role(readings: list[ScoreboardReading]) -> dict[Role, str]:
    """Most common name per role over the given (latest) readings."""
    votes: dict[Role, Counter] = defaultdict(Counter)
    for reading in readings:
        for b in reading.batters:
            if b.name:
                votes[Role.STRIKER if b.on_strike else Role.NON_STRIKER][b.name] += 1
        if reading.bowler:
            votes[Role.BOWLER][reading.bowler] += 1
    return {role: c.most_common(1)[0][0] for role, c in votes.items()}


def _feet_visible(box: np.ndarray, frame_size: tuple[int, int], occluders: list[tuple[float, float, float, float]]) -> bool:
    """False when the box's bottom edge is cut by the frame border or an on-screen graphic.

    Then the box ends at the obstruction, not at the feet, and projecting it onto the pitch
    would put the player metres away from where they stand.
    """
    w, h = frame_size
    x = (box[0] + box[2]) / 2
    bottom = box[3]
    if bottom >= h - _EDGE_PX:
        return False
    for x0, y0, x1, y1 in occluders:
        if x0 * w <= x <= x1 * w and bottom >= y0 * h - _EDGE_PX:
            return False
    return True


_HANDOVER_MAX_OVERLAP = 5  # frames an old and a new track of one person may coexist
_HANDOVER_MIN_IOU = 0.4


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def merge_handovers(frames: list[PlayerFrame]) -> tuple[list[PlayerFrame], dict[int, int]]:
    """Merge a track into the one it took over from.

    During a fast zoom the detector briefly sees one person twice (e.g. a box that stops at
    the score bar and a taller one including the legs below it); the tracker keeps the new
    box and drops the old, so for 1-3 frames two IDs cover one person and the registry,
    which only re-links IDs that are lost, cannot join them. Here: if a track starts while
    another is ending (overlap <= 5 frames) and their boxes agree over the overlap, the new
    track takes the old ID. In each shared frame the box with visible feet (else the larger)
    is kept. Returns the frames and the {new: old} merges.
    """
    by_sid: dict[int, dict[int, PlayerFrame]] = defaultdict(dict)
    for f in frames:
        by_sid[f.sid][f.frame_id] = f
    merged: dict[int, int] = {}
    order = sorted(by_sid, key=lambda s: min(by_sid[s]))
    for new in order:
        new_frames = by_sid[new]
        start = min(new_frames)
        best, best_iou = None, _HANDOVER_MIN_IOU
        for old in order:
            if old == new or old in merged or old not in by_sid:
                continue
            old_frames = by_sid[old]
            if not (min(old_frames) < start <= max(old_frames) <= start + _HANDOVER_MAX_OVERLAP):
                continue
            shared = sorted(set(old_frames) & set(new_frames))
            if not shared:
                continue
            iou = float(np.mean([_iou(old_frames[k].box, new_frames[k].box) for k in shared]))
            if iou >= best_iou:
                best, best_iou = old, iou
        if best is None:
            continue
        target = by_sid[best]
        for k, f in new_frames.items():
            keep = f
            if k in target:
                o = target[k]
                o_area = (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])
                f_area = (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])
                keep = o if (o.pitch_xy is not None, o_area) >= (f.pitch_xy is not None, f_area) else f
            target[k] = keep.model_copy(update={"sid": best})
        del by_sid[new]
        merged[new] = best
    out = sorted((f for fs in by_sid.values() for f in fs.values()), key=lambda f: (f.frame_id, f.sid))
    return out, merged


def _foot_point(box: np.ndarray, H: np.ndarray | None) -> tuple[float, float] | None:
    if H is None:
        return None
    x, y = (box[0] + box[2]) / 2, box[3]  # bottom centre: where the player touches the ground
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def run_players(cfg: Config, video: Path, tracker_yaml: Path | None = None, out_name: str = "players.json") -> PlayersResult:
    pcfg = cfg.players
    video_id = video_id_from_path(video)
    out_dir = cfg.output_dir_for(video_id)

    camera_path = out_dir / "camera.json"
    camera = read_doc(camera_path, CameraDoc) if camera_path.is_file() else None
    only_calibrated = bool(pcfg.get("only_calibrated_frames", True))
    if camera is None:
        log.warning("%s missing: no pitch positions, roles fall back to the detector", camera_path)
        only_calibrated = False
    H_by_frame = {
        f.frame_id: np.array(f.image_to_pitch) for f in (camera.frames if camera else []) if f.image_to_pitch
    }
    view_by_frame = {f.frame_id: f.view for f in (camera.frames if camera else [])}

    sb_path = out_dir / "scoreboard.json"
    readings = read_doc(sb_path, ScoreboardDoc).readings if sb_path.is_file() else []
    if not readings:
        log.warning("%s missing or empty: players will have no names", sb_path)

    det_cfg = pcfg["detector"]
    model = YOLO(str(cfg.root / det_cfg["weights"]))
    names = model.names
    tracker_yaml = tracker_yaml or cfg.root / pcfg["tracker"]["config"]

    frames_out: list[PlayerFrame] = []
    tracker_ids: set[tuple[int, int]] = set()  # (shot, id): the tracker restarts its IDs after a cut
    shot = 0
    skipped = processed = 0
    reading_idx = -1
    t0 = time.perf_counter()
    with VideoReader(video, resize=cfg.video.resize) as reader:
        meta = reader.meta(video_id)
        tracker = PlayerTracker.from_config(tracker_yaml, (meta.width, meta.height), cfg.root, cfg.device)
        registry = PlayerRegistry(meta.fps, RegistryOptions(**pcfg.get("registry", {})))
        roles = RoleAssigner(RoleOptions(fps=meta.fps))
        voter = IdentityVoter()
        occluders = list(cfg.camera.overlay_mask)  # the same graphics the camera tracker ignores
        prev_view: str | None = None
        prev_H: np.ndarray | None = None
        for frame in reader:
            view = view_by_frame.get(frame.frame_id)
            if only_calibrated and view is None:
                skipped += 1
                if prev_view is not None:  # left the calibrated view: whatever comes next is a new shot
                    registry.camera_cut()
                    tracker.reset()
                    shot += 1
                    prev_H = None
                prev_view = None
                continue
            prev_view = view
            processed += 1
            H = H_by_frame.get(frame.frame_id)
            if H is not None and prev_H is not None:
                registry.camera_moved(np.linalg.inv(H) @ prev_H)
            prev_H = H

            result = model.predict(
                frame.image,
                conf=det_cfg["conf"],
                imgsz=det_cfg["imgsz"],
                classes=det_cfg.get("classes"),
                device=cfg.device,
                verbose=False,
            )[0]
            tracks = tracker.update(result.boxes.cpu().numpy(), frame.image, H)
            boxes_cls = result.boxes.cls.cpu().numpy() if len(result.boxes) else np.zeros(0)
            boxes_conf = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros(0)

            dets, obs = [], {}
            for row in tracks:
                box, raw_id = row[:4], int(row[4])
                tracker_ids.add((shot, raw_id))
                feet = _foot_point(box, H) if _feet_visible(box, (meta.width, meta.height), occluders) else None
                dets.append(Detection(raw_id, tuple(float(v) for v in box), feet))
            sid_of = registry.update(frame.frame_id, dets)
            for row, det in zip(tracks, dets):
                det_idx = int(row[7])
                cls = names[int(boxes_cls[det_idx])] if det_idx < len(boxes_cls) else None
                conf = float(boxes_conf[det_idx]) if det_idx < len(boxes_conf) else 0.0
                obs[sid_of[det.track_id]] = (det.pitch_xy, cls, conf)
            role_of = roles.update(obs)

            while reading_idx + 1 < len(readings) and readings[reading_idx + 1].frame_id <= frame.frame_id:
                reading_idx += 1
            recent = [
                r
                for r in readings[max(0, reading_idx - _SCOREBOARD_CONSENSUS + 1) : reading_idx + 1]
                if frame.t - r.t <= _SCOREBOARD_MAX_AGE_S
            ]
            named = _names_by_role(recent)
            for sid, role in role_of.items():
                voter.vote(sid, shot, named.get(role))

            for det in dets:
                sid = sid_of[det.track_id]
                x1, y1, x2, y2 = (round(v) for v in det.box)
                if x2 <= x1 or y2 <= y1:
                    continue
                frames_out.append(
                    PlayerFrame(
                        frame_id=frame.frame_id,
                        t=frame.t,
                        sid=sid,
                        box=(x1, y1, x2, y2),
                        role=role_of.get(sid, Role.UNKNOWN),
                        pitch_xy=det.pitch_xy,
                    )
                )

    frames_out, handovers = merge_handovers(frames_out)
    for new, old in handovers.items():
        voter.merge(old, new)
    if handovers:
        log.info("merged %d tracker handovers into existing IDs", len(handovers))

    # Players do not change role in the middle of a delivery shot, and this pass runs after
    # the fact: each stable ID takes its dominant role for all its frames. Early frames (a
    # bowler still standing at his mark) otherwise get whatever the evidence said so far.
    role_votes: dict[int, Counter] = defaultdict(Counter)
    for f in frames_out:
        role_votes[f.sid][f.role] += 1
    dominant = {sid: votes.most_common(1)[0][0] for sid, votes in role_votes.items()}
    names = voter.resolve()
    frames_out = [
        f.model_copy(
            update={"role": dominant[f.sid]}
            | ({"identity": names[f.sid][0], "identity_conf": names[f.sid][1]} if f.sid in names else {})
        )
        for f in frames_out
    ]
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=processed, device=cfg.device)
    doc = PlayersDoc(video=meta, run=run, frames=frames_out)
    path = write_doc(doc, out_dir / out_name)
    gmc = tracker.gmc
    return PlayersResult(
        doc, path, len(tracker_ids), skipped, gmc.used_calibration if gmc else 0, gmc.used_fallback if gmc else 0
    )


def summarize(res: PlayersResult) -> list[str]:
    doc, run = res.doc, res.doc.run
    sids = {f.sid for f in doc.frames}
    per_frame = Counter(f.frame_id for f in doc.frames)
    lines = [
        f"frames: {run.frames_processed} processed, {res.frames_skipped} skipped (not a calibrated view); "
        f"{run.frames_processed / max(run.wall_time_s, 1e-9):.1f} frames/s",
        f"people per frame: median {int(np.median(list(per_frame.values()))) if per_frame else 0}, "
        f"max {max(per_frame.values(), default=0)}",
        f"IDs: {len(sids)} stable (from {res.tracker_ids} tracker IDs)",
    ]
    if res.gmc_calibrated or res.gmc_fallback:
        lines.append(f"camera motion: {res.gmc_calibrated} frames from calibration, {res.gmc_fallback} from optical flow")
    # each stable ID's dominant role and name
    by_sid: dict[int, Counter] = defaultdict(Counter)
    names: dict[int, str | None] = {}
    for f in doc.frames:
        by_sid[f.sid][f.role.value] += 1
        names[f.sid] = f.identity
    for sid in sorted(by_sid, key=lambda s: -sum(by_sid[s].values())):
        role, n = by_sid[sid].most_common(1)[0]
        total = sum(by_sid[sid].values())
        lines.append(f"  sid {sid:3d}: {total:4d} frames, {role} ({100 * n / total:.0f}%)" + (f", {names[sid]}" if names[sid] else ""))
    lines.append(f"players: {res.path}")
    return lines
