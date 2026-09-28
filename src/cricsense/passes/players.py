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

_SCOREBOARD_MAX_AGE_S = 3.0  # a reading older than this does not name anyone


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

    def resolve(self) -> dict[int, tuple[str, float]]:
        best: dict[tuple[int, str], tuple[int, int]] = {}  # (shot, name) -> (votes, sid)
        for sid, votes in self._votes.items():
            for name, n in votes.items():
                key = (self._shot[sid], name)
                if n > best.get(key, (0, -1))[0]:
                    best[key] = (n, sid)
        return {sid: (name, round(n / self._frames[sid], 3)) for (_, name), (n, sid) in best.items()}


def _names_by_role(reading: ScoreboardReading | None) -> dict[Role, str]:
    if reading is None:
        return {}
    names: dict[Role, str] = {}
    for b in reading.batters:
        if b.name:
            names[Role.STRIKER if b.on_strike else Role.NON_STRIKER] = b.name
    if reading.bowler:
        names[Role.BOWLER] = reading.bowler
    return names


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
        prev_view: str | None = None
        for frame in reader:
            view = view_by_frame.get(frame.frame_id)
            if only_calibrated and view is None:
                skipped += 1
                if prev_view is not None:  # left the calibrated view: whatever comes next is a new shot
                    registry.camera_cut()
                    tracker.reset()
                    shot += 1
                prev_view = None
                continue
            prev_view = view
            processed += 1
            H = H_by_frame.get(frame.frame_id)

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
                box, raw_id, det_idx = row[:4], int(row[4]), int(row[7])
                tracker_ids.add((shot, raw_id))
                dets.append(Detection(raw_id, tuple(float(v) for v in box), _foot_point(box, H)))
            sid_of = registry.update(frame.frame_id, dets)
            for row, det in zip(tracks, dets):
                det_idx = int(row[7])
                cls = names[int(boxes_cls[det_idx])] if det_idx < len(boxes_cls) else None
                conf = float(boxes_conf[det_idx]) if det_idx < len(boxes_conf) else 0.0
                obs[sid_of[det.track_id]] = (det.pitch_xy, cls, conf)
            role_of = roles.update(obs)

            while reading_idx + 1 < len(readings) and readings[reading_idx + 1].frame_id <= frame.frame_id:
                reading_idx += 1
            reading = readings[reading_idx] if reading_idx >= 0 else None
            if reading is not None and frame.t - reading.t > _SCOREBOARD_MAX_AGE_S:
                reading = None
            named = _names_by_role(reading)
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

    names = voter.resolve()
    frames_out = [
        f.model_copy(update={"identity": names[f.sid][0], "identity_conf": names[f.sid][1]}) if f.sid in names else f
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
