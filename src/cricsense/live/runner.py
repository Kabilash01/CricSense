"""Live mode: the pipeline on a stream, at the rate each part can sustain.

    cricsense live --video data/final_0051_6min.mp4 --out outputs/live/overlay.mp4
    cricsense live --url rtsp://camera/stream --out live.mp4 --seconds 600

A reader thread takes frames at the source's pace (a file is played back in real time; a stream
arrives as it comes) and hands each worker only the frames it can keep up with - a worker that
falls behind gets the newest frame, never a backlog:
  score  every 0.5 s: the score bar -> trusted states -> each ball's outcome as soon as the bar
         settles (runs, boundary, wicket, wide / no-ball) and the radar speed;
  view   every Nth frame: is this the calibrated main view, and its homography (camera tracker
         against the match's reference views);
  ball   main-view frames: the fine-tuned ball model; when a main-view shot ends, the delivery
         in it -> where it pitched (length, line);
  draw   the newest frame with the latest results, written to a file at the source's rate.
Everything degrades rather than lags: the report gives each worker's throughput and how long
after a frame each event was emitted (latency).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from cricsense.config import Config

log = logging.getLogger("cricsense.live")


@dataclass
class LiveEvent:
    kind: str  # BALL | SPEED | BOUNCE
    frame_id: int  # the frame the evidence came from
    t: float  # source time of that frame
    emitted_wall: float  # wall-clock seconds since the start
    payload: dict

    @property
    def latency_s(self) -> float:
        return self.emitted_wall - self.t


@dataclass
class LiveState:
    score: str | None = None  # "93-3 (12.0)"
    batters: list = field(default_factory=list)
    bowler: str | None = None
    last_ball: dict | None = None
    last_ball_wall: float = -1e9
    last_delivery: dict | None = None
    radar: float | None = None
    bounces: list = field(default_factory=list)  # (length, line, class)
    events: list[LiveEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, ev: LiveEvent) -> None:
        with self.lock:
            self.events.append(ev)
        log.info("%s at %.1f s (latency %.1f s): %s", ev.kind, ev.t, ev.latency_s, ev.payload)


class Worker(threading.Thread):
    """Takes the newest frame offered; frames arriving while it is busy are dropped."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self.inbox: queue.Queue = queue.Queue(maxsize=1)
        self.processed = 0
        self.dropped = 0
        self.busy_s = 0.0
        self.done = threading.Event()

    def offer(self, item) -> None:
        try:
            self.inbox.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def run(self) -> None:
        while True:
            item = self.inbox.get()
            if item is None:
                break
            t0 = time.perf_counter()
            try:
                self.handle(*item)
            except Exception:  # a bad frame must not stop the live loop
                log.exception("%s failed on a frame", self.name)
            self.busy_s += time.perf_counter() - t0
            self.processed += 1
        self.finish()
        self.done.set()

    def handle(self, frame_id: int, t: float, image: np.ndarray) -> None: ...

    def finish(self) -> None: ...


class ScoreWorker(Worker):
    def __init__(self, cfg: Config, state: LiveState, start_wall: float, video_id: str) -> None:
        super().__init__("score")
        from cricsense.fuse.outcome import trusted_states
        from cricsense.identity.match_sheet import layout_path, load_sheet
        from cricsense.identity.scoreboard_ocr import ScoreboardLayout, ScoreboardReader

        sheet = load_sheet(cfg.root, video_id)  # the match of the clip the calibration came from
        self.squad = set(sheet.all_names) if sheet else set()
        self.team_of = {n: t.code for t in sheet.teams for n in t.squad} if sheet else None
        self.reader = ScoreboardReader(ScoreboardLayout.from_yaml(layout_path(cfg.root, video_id)), tuple(self.squad))
        self.trusted = trusted_states
        self.readings: deque = deque(maxlen=400)  # the last few minutes is all the rules need
        self.state, self.t0 = state, start_wall
        self.last = None  # the last settled state already reported

    def handle(self, frame_id: int, t: float, image: np.ndarray) -> None:
        r = self.reader.read(image, frame_id, t)
        if r is not None:
            self.update(r)

    def update(self, r) -> None:
        """One score-bar reading: the live score, a new radar speed, each newly settled ball."""
        frame_id, t = r.frame_id, r.t
        self.readings.append(r)
        now = time.perf_counter() - self.t0
        if r.speed_kph and r.speed_kph != self.state.radar:
            self.state.radar = r.speed_kph
            self.state.emit(LiveEvent("SPEED", frame_id, t, now, {"radar_kph": r.speed_kph}))
        states = self.trusted(list(self.readings), self.squad, self.team_of)
        if not states:
            return
        last = states[-1]
        with self.state.lock:
            self.state.score = f"{last.runs}-{last.wickets} ({last.legal_balls // 6}.{last.legal_balls % 6})"
            self.state.batters = [(b[0], b[1], b[2], b[3]) for b in last.batters]
            self.state.bowler = last.bowler
        # each settled state after the last one reported: that ball's outcome, now
        fresh = [st for st in states if self.last is None or st.t > self.last.t]
        prev = self.last
        for b in fresh:
            a, prev = prev, b
            if a is None:
                continue
            d_runs, d_balls = b.runs - a.runs, b.legal_balls - a.legal_balls
            if d_balls < 0 or d_runs < 0 or (d_balls, d_runs, b.wickets - a.wickets) == (0, 0, 0):
                continue
            striker = a.striker
            now_b = b.batter(striker[0]) if striker else None
            off_bat = now_b[1] - striker[1] if (striker and now_b) else None
            if d_balls == 1:
                kind = "legal"
            elif striker and now_b:
                kind = "wide" if now_b[2] == striker[2] else "no_ball"
            else:
                kind = "extra"
            ball = {"over": f"{a.legal_balls // 6}.{a.legal_balls % 6 + 1}", "runs": d_runs, "kind": kind,
                    "boundary": {4: "FOUR", 6: "SIX"}.get(off_bat) if off_bat is not None else None,
                    "wicket": b.wickets > a.wickets, "striker": striker[0] if striker else None, "bowler": a.bowler,
                    "score": f"{b.runs}-{b.wickets}"}
            with self.state.lock:
                self.state.last_ball, self.state.last_ball_wall = ball, time.perf_counter() - self.t0
            self.state.emit(LiveEvent("BALL", frame_id, b.t, time.perf_counter() - self.t0, ball))
        if fresh:
            self.last = fresh[-1]


class ViewBallWorker(Worker):
    """Main-view test and homography, then the ball model on main-view frames; a delivery per shot."""

    def __init__(
        self, cfg: Config, state: LiveState, start_wall: float, calibration_yaml: Path, fps: float, device: str, buffer: deque
    ) -> None:
        super().__init__("view+ball")
        self.buffer = buffer  # the reader's last few seconds of frames: (frame_id, t, image)
        import os

        from cricsense.calibration.camera_motion import CameraTracker, _shows_pitch, reference_images
        from cricsense.calibration.homography import PitchHomography
        from cricsense.io import read_yaml_doc
        from cricsense.schema import CalibrationDoc

        cal = read_yaml_doc(calibration_yaml, CalibrationDoc)
        refs = reference_images(None, cal, cfg.root, cfg.video.resize)
        self.refs = {v.name: (refs[v.name], PitchHomography(np.array(v.image_to_pitch))) for v in cal.views}
        self.camera_opts = cfg.camera
        self.new_tracker = lambda: CameraTracker(self.refs, self.camera_opts)
        self.tracker = self.new_tracker()
        self.shows_pitch = lambda img: _shows_pitch(img, cfg.camera.gate_min_pitch or 0.03)
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO

        det = cfg.ball["detector"]
        # as offline: the original model finds the delivery, the fine-tuned one tracks it where it can
        self.finder = YOLO(str(cfg.root / det["weights"]))
        self.model = YOLO(str(cfg.root / det["track_weights"])) if det.get("track_weights") else self.finder
        self.imgsz = det["imgsz"]  # 1920, as offline: at 1280 the 6-15 px ball is mostly lost
        self.device, self.fps = device, fps
        self.state, self.t0 = state, start_wall
        self.shot_start: int | None = None  # first frame of the current main-view shot
        self.shots: queue.Queue = queue.Queue(maxsize=3)  # finished shots waiting to be processed
        self.shot_thread = threading.Thread(target=self._shot_loop, name="shots", daemon=True)
        self.shot_thread.start()
        self.prev_thumb = None
        self.counts = {"main_view_frames": 0, "shots": 0, "short_shot": 0, "not_buffered": 0, "no_delivery": 0, "no_bounce": 0, "bounces": 0}
        self.misses = 0

    def handle(self, frame_id: int, t: float, image: np.ndarray) -> None:
        """Every few frames: has a main-view shot started (one direct match to a reference view
        is enough), and has it ended (a hard cut, or the pitch gone)? At its end the shot is
        processed from the buffer (_close_shot)."""
        thumb = cv2.cvtColor(cv2.resize(image, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = self.prev_thumb is not None and float(np.abs(thumb - self.prev_thumb).mean()) > CUT_DIFF
        self.prev_thumb = thumb
        pitch = self.shows_pitch(image)
        if self.shot_start is not None:
            if cut or not pitch:
                self._queue_shot(self.shot_start, frame_id - 1)
                self.shot_start = None
            return
        if pitch:
            self.tracker.reset()
            est = self.tracker.estimate(image)
            if est.homography is not None:
                self.counts["main_view_frames"] += 1
                self.shot_start = frame_id

    def _queue_shot(self, f0: int, f1: int) -> None:
        """Copy the shot's frames out of the buffer (it moves on) and hand them to the shot
        thread, so noticing the next shot never waits for processing this one."""
        self.counts["shots"] += 1
        # the shot began a little before it was noticed: take the buffered frames since then
        frames = [(f, img) for f, _, img in list(self.buffer) if f0 - LOOKBACK_FRAMES <= f <= f1]
        if len(frames) < 8:
            self.counts["not_buffered"] += 1
            return
        try:
            self.shots.put_nowait(frames)
        except queue.Full:
            self.counts["shot_queue_full"] = self.counts.get("shot_queue_full", 0) + 1

    def _shot_loop(self) -> None:
        while True:
            frames = self.shots.get()
            if frames is None:
                break
            try:
                self._close_shot(frames)
            except Exception:
                log.exception("shot processing failed")
            self.shots.task_done()

    def _close_shot(self, frames: list) -> None:
        """The shot's frames tracked in order with a fresh tracker (as offline: consecutive
        frames follow the zoom), the ball models on the calibrated ones in GPU batches, then
        the delivery."""
        from cricsense.ball.delivery import Candidate, FrameGeometry, find_delivery
        from cricsense.calibration.camera_pose import CameraPose

        f0, f1 = frames[0][0], frames[-1][0]
        try:
            import torch

            torch.cuda.empty_cache()  # whatever the last shot left behind
        except Exception:
            pass
        tracker = self.new_tracker()
        calibrated = []
        for f, img in frames:
            est = tracker.estimate(img)
            if est.homography is not None:
                calibrated.append((f, img, est.homography.H))
        if len(calibrated) < 8:
            self.counts["short_shot"] += 1
            return
        self.counts["main_view_frames"] += len(calibrated)
        geo, cands = {}, []
        size = calibrated[0][1].shape[1::-1]
        for f, _, H in calibrated:
            try:
                P = CameraPose.from_homography(H, size).P
            except Exception:
                P = None
            geo[f] = FrameGeometry(H, P)
        found = {}
        for name, model in (("finder", self.finder), ("tracker", self.model)):
            cands = []
            for k in range(0, len(calibrated), BATCH):
                batch = calibrated[k : k + BATCH]
                for (f, img, _), r in zip(batch, self._detect(model, [img for _, img, _ in batch])):
                    cands += [
                        Candidate(f, f / self.fps, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(c))
                        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
                        if (b[1] + b[3]) / 2 < 0.85 * img.shape[0] and float(c) >= 0.15
                    ]
            found[name] = find_delivery(cands, geo, self.fps)
            if self.model is self.finder:
                found["tracker"] = found["finder"]
                break
        from cricsense.passes.ball import prefer_precise

        d = found["tracker"] if prefer_precise(found["finder"], found["tracker"]) else found["finder"]
        frames = calibrated
        log.info("shot %d-%d: %d calibrated frames, delivery %s", f0, f1, len(calibrated), "found" if d else "not found")
        if d is None or d.bounce is None:
            self.counts["no_delivery" if d is None else "no_bounce"] += 1
            return
        self.counts["bounces"] += 1
        zones = self.state_zones
        length = d.bounce.x
        cls = "yorker" if length < zones[0] else "full" if length < zones[1] else "good" if length < zones[2] else "short"
        info = {"length_m": round(length, 2), "line_m": round(d.bounce.y, 2), "length_class": cls,
                "confirmed": bool(d.bounce.confirmed), "detections": len(d.track), "frames_in_shot": len(frames)}
        with self.state.lock:
            self.state.last_delivery = info
            self.state.bounces.append((length, d.bounce.y, cls, bool(d.bounce.confirmed)))
        self.state.emit(LiveEvent("BOUNCE", int(round(d.bounce.t * self.fps)), d.bounce.t, time.perf_counter() - self.t0, info))

    state_zones = (2.0, 6.0, 8.0)  # yorker | full | good | short, metres from the striker's stumps

    def _detect(self, model, images: list) -> list:
        """A ball model on a batch. On a GPU short of memory (another job sharing it): one frame
        at a time, then the CPU for that batch - slower, but the delivery is still found."""
        import torch

        kw = {"imgsz": self.imgsz, "conf": 0.1, "classes": [0], "verbose": False}
        try:
            return model.predict(images, device=self.device, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        try:
            return [model.predict(img, device=self.device, **kw)[0] for img in images]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.counts["cpu_fallback_batches"] = self.counts.get("cpu_fallback_batches", 0) + 1
            return model.predict(images, device="cpu", **kw)

    def finish(self) -> None:
        if self.shot_start is not None and self.buffer:
            self._queue_shot(self.shot_start, self.buffer[-1][0])
        self.shots.put(None)
        self.shot_thread.join(timeout=300)


def draw(image: np.ndarray, state: LiveState, wall: float, t: float, stats: str) -> np.ndarray:
    from cricsense.render.draw import blend, chip, panel, paste
    from cricsense.render.pitch_map import MapPoint, MapView, draw_pitch_map

    h, w = image.shape[:2]
    with state.lock:
        score, batters, bowler = state.score, list(state.batters), state.bowler
        last_ball, age = state.last_ball, wall - state.last_ball_wall
        delivery, radar, bounces = state.last_delivery, state.radar, list(state.bounces)
    rows = [f"LIVE  {score or '-'}"]
    if batters:
        rows.append("  ".join(f"{n.title()}{'*' if s else ''} {r} ({b})" for n, r, b, s in batters))
    if bowler:
        rows.append(f"bowling: {bowler.title()}")
    if delivery:
        rows.append(f"last ball pitched {delivery['length_class']} {delivery['length_m']:.1f} m" + ("" if delivery["confirmed"] else " (inferred)"))
    if radar:
        rows.append(f"speed {radar:g} km/h (radar)")
    card = panel(470, 30 + 30 * len(rows), (18, 18, 18), 200, 12)
    for i, row in enumerate(rows):
        paste(card, chip(row, 20 if i else 22, (255, 255, 255), None, i == 0, (0, 0)), 16, 12 + 30 * i)
    blend(image, card, w - 498, 28)
    if last_ball and age < 4.0:
        text = "WICKET" if last_ball["wicket"] else last_ball["boundary"] or (
            f"{last_ball['kind'].upper().replace('_', '-')} +{last_ball['runs']}" if last_ball["kind"] != "legal"
            else ("DOT BALL" if last_ball["runs"] == 0 else f"{last_ball['runs']} RUN{'S' if last_ball['runs'] > 1 else ''}"))
        tag = chip(text, 40, (255, 255, 255), (200, 60, 230) if text in ("SIX", "FOUR", "WICKET") else (60, 60, 60), True, (24, 8), 230, 10)
        blend(image, tag, w // 2 - tag.shape[1] // 2, int(0.12 * h))
    if bounces:
        pts = [MapPoint(str(i + 1), x, y, c, conf) for i, (x, y, c, conf) in enumerate(bounces[-12:])]
        img = draw_pitch_map(pts, MapView(px_per_m=24.0, x_range=(-1.0, 12.0), margin_m=1.0, gutter_px=16), highlight=len(pts) - 1, label_only_highlight=True)
        blend(image, img, w - img.shape[1] - 28, int(0.85 * h) - img.shape[0] - 18)
    blend(image, chip(stats, 16, (220, 220, 220), (10, 10, 10), False, (8, 3), 170, 4), 20, 20)
    return image


BUFFER_S = 12.0  # a main-view delivery shot lasts 2-5 s, and is processed after it ends
LOOKBACK_FRAMES = 12  # the shot began a few frames before the worker noticed it
CUT_DIFF = 28.0  # mean grey-level change between sampled thumbnails that marks a cut (normal motion ~10-15)
BATCH = 2  # ball model batch: small enough beside another job on the GPU


def run_live(
    cfg: Config, source: str, out: Path | None, calibration_yaml: Path, seconds: float | None = None,
    view_every: int = 2, realtime: bool = True,
) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    state = LiveState()
    buffer: deque = deque(maxlen=int(BUFFER_S * fps))  # the last seconds of frames, for a shot's delivery
    match_video = yaml.safe_load(Path(calibration_yaml).read_text()).get("video_id") or Path(str(source)).stem
    score = ScoreWorker(cfg, state, 0.0, match_video)
    vb = ViewBallWorker(cfg, state, 0.0, calibration_yaml, fps, cfg.device, buffer)
    workers = [score, vb]
    # the clock starts once the models are loaded (~8 s): before that nothing is live yet
    start = time.perf_counter()
    for wk in workers:
        wk.t0 = start
        wk.start()
    writer = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_id, drawn, late, max_lag, max_lag_t = 0, 0, 0, 0.0, 0.0
    score_every = max(1, int(round(fps / 2)))
    while True:
        ok, image = cap.read()
        if not ok:
            break
        t = frame_id / fps
        if seconds is not None and t > seconds:
            break
        if realtime:  # a file plays at the source's pace, as a stream would arrive
            wait = t - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.2:
                late += 1
            if -wait > max_lag:
                max_lag, max_lag_t = -wait, t
        buffer.append((frame_id, t, image))
        if frame_id % score_every == 0:
            score.offer((frame_id, t, image))
        if frame_id % view_every == 0:
            vb.offer((frame_id, t, image))
        if writer is not None:
            wall = time.perf_counter() - start
            stats = f"live  t {t:6.1f}s  wall {wall:6.1f}s  view+ball {vb.processed}/{frame_id // view_every + 1}"
            writer.write(draw(image.copy(), state, wall, t, stats))
            drawn += 1
        frame_id += 1
    play_wall = time.perf_counter() - start  # the source has been played; shots may still be processing
    for wk in workers:
        wk.inbox.put(None)
    for wk in workers:
        wk.done.wait(timeout=120)
    cap.release()
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - start
    by_kind: dict[str, list[float]] = {}
    for ev in state.events:
        by_kind.setdefault(ev.kind, []).append(ev.latency_s)
    report = {
        "source": source, "frames": frame_id, "source_seconds": round(frame_id / fps, 1), "wall_seconds": round(wall, 1),
        "play_wall_seconds": round(play_wall, 1), "reader_max_lag_s": round(max_lag, 2), "reader_max_lag_at_s": round(max_lag_t, 1),
        "realtime": realtime, "frames_drawn": drawn, "reader_late_frames": late,
        "workers": {wk.name: {"processed": wk.processed, "dropped": wk.dropped, "busy_s": round(wk.busy_s, 1),
                              "rate_per_s": round(wk.processed / max(wall, 1e-6), 2), **getattr(wk, "counts", {})} for wk in workers},
        "latency_s": {k: {"n": len(v), "median": round(float(np.median(v)), 2), "max": round(float(np.max(v)), 2)} for k, v in by_kind.items()},
        "events": [{"kind": e.kind, "t": round(e.t, 2), "latency_s": round(e.latency_s, 2), **e.payload} for e in state.events],
    }
    return report


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
