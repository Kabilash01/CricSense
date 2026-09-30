"""`cricsense pose`: the striker's keypoints around each delivery -> pose.json.

The pose model runs on a crop around the striker (players.json), not the whole frame: the
batter is ~250 px tall in a 1920 frame and the crop gives the model the detail it needs.
From the keypoints: footwork on the calibrated pitch and the swing
(cricsense.pose.batter).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from cricsense.config import Config, video_id_from_path
from cricsense.io import VideoReader, read_doc, write_doc
from cricsense.pose.batter import Observation, Thresholds, batting_hand, footwork, swing
from cricsense.schema import BallDoc, BatterAction, CameraDoc, EventsDoc, EventType, PlayersDoc, PoseDoc, PoseFrame, Role, RunInfo

log = logging.getLogger("cricsense.pose")


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def run_pose(cfg: Config, video: Path, device: str | None = None) -> tuple[PoseDoc, Path]:
    import yaml

    pcfg = yaml.safe_load((cfg.root / "configs" / "pose.yaml").read_text())
    th = Thresholds(**pcfg["thresholds"])
    video_id = video_id_from_path(video)
    out_dir = cfg.output_dir_for(video_id)
    ball = read_doc(out_dir / "ball.json", BallDoc)
    players = read_doc(out_dir / "players.json", PlayersDoc)
    camera = read_doc(out_dir / "camera.json", CameraDoc)
    events = read_doc(out_dir / "events.json", EventsDoc) if (out_dir / "events.json").is_file() else None
    fps = ball.video.fps
    H = {c.frame_id: np.array(c.image_to_pitch) for c in camera.frames if c.image_to_pitch}
    striker = {p.frame_id: p for p in players.frames if p.role == Role.STRIKER}
    contact = {e.payload["delivery_id"]: e.frame_id for e in (events.events if events else []) if e.type == EventType.CONTACT}

    w0, w1 = pcfg["window"]
    plan: dict[int, list[int]] = {}
    for d in ball.deliveries:
        shot = contact.get(d.delivery_id) or (int(round(d.bounce_t * fps)) + 4 if d.bounce_t is not None else d.release_frame + 16)
        frames = [f for f in range(d.release_frame + w0, max(d.release_frame + w1, shot + 10) + 1) if f in striker]
        plan[d.delivery_id] = frames
    wanted = {f for fs in plan.values() for f in fs}

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(cfg.root / pcfg["model"]))
    t0 = time.perf_counter()
    kps: dict[int, tuple[np.ndarray, tuple]] = {}
    if wanted:
        with VideoReader(video, start=min(wanted), end=max(wanted) + 1) as reader:
            for fr in reader:
                if fr.frame_id not in wanted:
                    continue
                p = striker[fr.frame_id]
                x1, y1, x2, y2 = p.box
                side = int(pcfg["crop_scale"] * (y2 - y1))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                h, w = fr.image.shape[:2]
                ox, oy = int(np.clip(cx - side // 2, 0, max(w - side, 0))), int(np.clip(cy - side // 2, 0, max(h - side, 0)))
                crop = fr.image[oy : oy + side, ox : ox + side]
                r = model.predict(crop, imgsz=pcfg["imgsz"], conf=pcfg["conf"], device=device or cfg.device, verbose=False)[0]
                if r.keypoints is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy() + [ox, oy, ox, oy]
                best = max(range(len(boxes)), key=lambda i: _iou(boxes[i], p.box))
                if _iou(boxes[best], p.box) < 0.3:
                    continue  # the keeper or an umpire, not the striker
                k = r.keypoints.data[best].cpu().numpy().copy()
                k[:, 0] += ox
                k[:, 1] += oy
                kps[fr.frame_id] = (k, tuple(float(v) for v in boxes[best]))

    pose_frames: list[PoseFrame] = []
    batters: list[BatterAction] = []
    for d in ball.deliveries:
        bottom = 0.85 * ball.video.height  # the score bar covers the rest
        obs = [Observation(f, kps[f][0], kps[f][1], H.get(f), bottom) for f in plan[d.delivery_id] if f in kps]
        for o in obs:
            pose_frames.append(
                PoseFrame(
                    frame_id=o.frame, t=o.frame / fps, sid=striker[o.frame].sid, role=Role.STRIKER,
                    keypoints=[tuple(round(float(v), 2) for v in row) for row in o.kp],
                )
            )
        shot = contact.get(d.delivery_id) or (int(round(d.bounce_t * fps)) + 4 if d.bounce_t is not None else d.release_frame + 16)
        fw, stride = footwork(obs, d.release_frame, shot, th)
        sw, peak_f, speed = swing(obs, d.release_frame, fps, th)
        sids = [striker[o.frame].sid for o in obs]
        batters.append(
            BatterAction(
                delivery_id=d.delivery_id,
                sid=max(set(sids), key=sids.count) if sids else None,
                footwork=fw, stride_m=stride, swing=sw, batting_hand=batting_hand(obs, d.release_frame), swing_peak_frame=peak_f, hand_speed=speed,
                frames=len(obs),
            )
        )
    pose_frames.sort(key=lambda p: p.frame_id)
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=len(wanted), device=device or cfg.device)
    doc = PoseDoc(video=ball.video, run=run, frames=pose_frames, batters=batters)
    path = write_doc(doc, out_dir / "pose.json")
    return doc, path


def summarize(doc: PoseDoc, path: Path) -> list[str]:
    out = []
    for b in doc.batters:
        parts = [f"delivery {b.delivery_id}: {b.frames} frames"]
        if b.footwork:
            parts.append(f"{b.footwork} ({b.stride_m:+.2f} m)")
        if b.swing:
            parts.append(f"{b.swing} (hands {b.hand_speed} body-heights/s)")
        if b.batting_hand:
            parts.append(f"{b.batting_hand}-handed")
        out.append(" | ".join(parts))
    out.append(f"pose: {path}")
    return out
