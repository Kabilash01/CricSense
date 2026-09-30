"""`cricsense ball`: deliveries -> ball.json (release, track, bounce, line and length, speeds).

Per main-view camera shot (camera.json): ball candidates from the detector, the delivery
found among them (cricsense.ball.delivery), the bounce and length/line from it, the radar
speed the broadcaster showed after the ball (scoreboard.json), and our own speed estimate.

With a second, precise model (detector.track_weights) the first model decides whether a shot
has a delivery and the precise one gives the track wherever it finds the delivery too (and a
bounce, unless neither does). Only precise tracks get a speed estimate: the first model's
false detections near the bowler make its release time, and so its speed, unreliable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cricsense.ball.candidates import detect_candidates
from cricsense.ball.delivery import Candidate, FrameGeometry, find_delivery
from cricsense.config import Config, video_id_from_path
from cricsense.io import read_doc, write_doc
from cricsense.schema import BallDoc, BallFrame, CameraDoc, Delivery, LengthClass, RunInfo, ScoreboardDoc

log = logging.getLogger("cricsense.ball")


@dataclass(frozen=True)
class LengthZones:
    yorker: float
    full: float
    good: float

    def classify(self, length_m: float) -> LengthClass:
        if length_m < self.yorker:
            return LengthClass.YORKER
        if length_m < self.full:
            return LengthClass.FULL
        if length_m < self.good:
            return LengthClass.GOOD
        return LengthClass.SHORT


def shots_of(frames: list[int]) -> list[list[int]]:
    """Runs of consecutive frame ids: one camera shot each."""
    if not frames:
        return []
    shots, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            shots.append(cur)
            cur = [b]
    shots.append(cur)
    return shots


def radar_per_shot(readings, shots: list[list[int]], fps: float) -> dict[int, float]:
    """Each new radar value belongs to the last delivery shot that started before it appeared."""
    values: list[tuple[float, float]] = []
    for r in readings:
        if r.speed_kph and (not values or values[-1][0] != r.speed_kph):
            values.append((r.speed_kph, r.t))
    starts = [s[0] / fps for s in shots]
    out: dict[int, float] = {}
    for kph, t in values:
        before = [i for i, s in enumerate(starts) if s < t]
        if before and shots[before[-1]][0] not in out:
            out[shots[before[-1]][0]] = kph
    return out


def prefer_precise(found, precise) -> bool:
    """Use the precise model's track for this shot? Yes when it found the delivery, unless only
    the first model's track has a bounce."""
    return precise is not None and (precise.bounce is not None or found is None or found.bounce is None)


def run_ball(cfg: Config, video: Path, weights: Path | None = None, fresh: bool = False) -> tuple[BallDoc, Path]:
    bcfg = cfg.ball
    det = bcfg["detector"]
    weights = weights or cfg.root / det["weights"]
    zones = LengthZones(**bcfg["length_zones"])
    video_id = video_id_from_path(video)
    out_dir = cfg.output_dir_for(video_id)
    cam = read_doc(out_dir / "camera.json", CameraDoc)
    fps, height = cam.video.fps, cam.video.height
    geo = {
        f.frame_id: FrameGeometry(np.array(f.image_to_pitch), None if f.projection is None else np.array(f.projection))
        for f in cam.frames
        if f.image_to_pitch
    }
    sb_path = out_dir / "scoreboard.json"
    readings = read_doc(sb_path, ScoreboardDoc).readings if sb_path.is_file() else []

    t0 = time.perf_counter()
    cands = detect_candidates(video, set(geo), weights, out_dir, det["imgsz"], det["conf"], cfg.device, fresh)
    track_weights = cfg.root / det["track_weights"] if det.get("track_weights") else None
    if track_weights is not None and not track_weights.is_file():
        log.warning("track model %s not found; using %s alone", track_weights, weights.name)
        track_weights = None
    track_cands = (
        detect_candidates(video, set(geo), track_weights, out_dir, det["imgsz"], det["conf"], cfg.device, fresh)
        if track_weights is not None
        else None
    )
    shots = shots_of(sorted(geo))
    radar = radar_per_shot(readings, shots, fps)

    frames_out: list[BallFrame] = []
    deliveries: list[Delivery] = []
    def candidates(source: dict[int, list[dict]], shot: list[int]) -> list[Candidate]:
        return [
            Candidate(f, f / fps, (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2, d["conf"])
            for f in shot
            for d in source.get(f, [])
            if d["cls"] == 0 and d["conf"] >= det["min_track_conf"] and (d["box"][1] + d["box"][3]) / 2 < 0.85 * height
        ]

    for shot in shots:
        found = find_delivery(candidates(cands, shot), geo, fps)
        model, estimate = weights.name, found.speed_estimate_kph if found is not None else None
        if track_cands is not None:
            precise = find_delivery(candidates(track_cands, shot), geo, fps)
            if prefer_precise(found, precise):
                found, model, estimate = precise, track_weights.name, precise.speed_estimate_kph
            else:
                estimate = None
        if found is None:
            continue
        did = len(deliveries)
        b = found.bounce
        bounce_frame = None if b is None else int(round(b.t * fps))
        for c, _, _ in found.track:
            frames_out.append(
                BallFrame(
                    frame_id=c.frame_id,
                    t=c.frame_id / fps,
                    uv=(c.u, c.v),
                    observed=True,
                    delivery_id=did,
                    pitch_xy=(b.x, b.y) if b is not None and c.frame_id == bounce_frame else None,
                )
            )
        deliveries.append(
            Delivery(
                delivery_id=did,
                shot_frames=(shot[0], shot[-1]),
                release_frame=found.release_frame,
                release_t=max(found.release_t, 0.0),
                release_point=tuple(float(v) for v in found.release_point),
                detections=len(found.track),
                bounce_t=None if b is None else b.t,
                bounce_xy=None if b is None else (b.x, b.y),
                bounce_confirmed=bool(b and b.confirmed),
                length_m=None if b is None else b.x,
                line_m=None if b is None else b.y,
                length_class=None if b is None else zones.classify(b.x),
                radar_speed_kph=radar.get(shot[0]),
                speed_estimate_kph=estimate,
                track_model=model,
            )
        )
    frames_out.sort(key=lambda f: f.frame_id)
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=len(geo), device=cfg.device)
    doc = BallDoc(video=cam.video, run=run, frames=frames_out, deliveries=deliveries)
    path = write_doc(doc, out_dir / "ball.json")
    log.info("wrote %s: %d deliveries", path, len(deliveries))
    return doc, path


def summarize(doc: BallDoc, path: Path) -> list[str]:
    lines = []
    for d in doc.deliveries:
        parts = [f"delivery {d.delivery_id}: release frame {d.release_frame}, {d.detections} detections ({d.track_model})"]
        if d.length_m is not None:
            parts.append(
                f"bounce length {d.length_m:.2f} m ({d.length_class.value}), line {d.line_m:+.2f} m"
                + ("" if d.bounce_confirmed else " (unconfirmed)")
            )
        if d.radar_speed_kph:
            parts.append(f"radar {d.radar_speed_kph:g} km/h")
        if d.speed_estimate_kph:
            parts.append(f"estimate {d.speed_estimate_kph:.0f} km/h")
        lines.append(" | ".join(parts))
    lines.append(f"ball: {path}")
    return lines
