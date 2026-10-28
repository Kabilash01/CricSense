"""Live plus: live mode with the players named on screen and a delivery card with the speed.

    cricsense live-plus --video "Cricket match/match.mp4" --calibration configs/calibration/<clip>.yaml

The live workers (score bar, view + ball) run as in `cricsense live`, plus one more:
  players  the newest main-view frame: people (the players pass's detector and tracker), their
           feet on the pitch through this frame's calibration, roles from where they stand
           (striker, non-striker, bowler, keeper, umpires), names from the score bar (the
           keeper's from the match sheet).
The picture is shown 1 s behind the source, so each frame is drawn with the players found on it
(or a frame or two before) rather than with boxes that trail a moving camera. The card is the
offline overlay's: over, bowler -> striker, length and line when the bounce is found, speed (the
radar on the score bar, else our estimate, "≈"), and the outcome once the bar settles.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from cricsense.config import Config
from cricsense.live.runner import CUT_DIFF, BUFFER_S, FlowCamera, LiveState, ScoreWorker, ViewBallWorker, Window, Worker
from cricsense.schema import Role

log = logging.getLogger("cricsense.live")

DELAY_S = 1.0  # the picture is this far behind the source: time for the players worker to catch up
MAX_GAP = 5  # frames: players found on a frame are drawn on the next few, no further
CARD_HOLD_S = 25.0  # the finished ball's card stays until the next ball is under way, at most this long
RADAR_LATE_S = 8.0  # a radar reading this soon after a ball's result is still that ball's
NEXT_BALL_AFTER_S = 15.0  # the next delivery comes at least this long after a result (measured: 20 s or more)
BOUNCE_BEFORE_S = 20.0  # a ball's bounce is at most this long before its result (measured: 2-14 s)
YIELD_S = 0.6  # the ball model waits while the main view was on screen this recently...
YIELD_MAX_S = 20.0  # ...but never longer than this per batch (a long main-view spell)


@dataclass
class Person:  # what render.overlay draws for a player
    box: tuple[int, int, int, int]
    role: Role
    identity: str | None
    team: str | None


class PlayersWorker(Worker):
    """People on the newest main-view frame, with roles and names."""

    def __init__(self, cfg: Config, state: LiveState, calibration_yaml: Path, fps: float, match_video: str) -> None:
        super().__init__("players")
        from ultralytics import YOLO

        from cricsense.calibration.camera_motion import CameraTracker, _shows_pitch, reference_images
        from cricsense.calibration.homography import PitchHomography
        from cricsense.identity.match_sheet import load_sheet
        from cricsense.io import read_yaml_doc
        from cricsense.schema import CalibrationDoc

        cal = read_yaml_doc(calibration_yaml, CalibrationDoc)
        refs = reference_images(None, cal, cfg.root, cfg.video.resize)
        refs = {v.name: (refs[v.name], PitchHomography(np.array(v.image_to_pitch))) for v in cal.views}
        self.camera = FlowCamera(CameraTracker(refs, cfg.camera), cfg.camera.overlay_mask)
        self.shows_pitch = lambda img: _shows_pitch(img, cfg.camera.gate_min_pitch or 0.03)
        self.occluders = list(cfg.camera.overlay_mask)
        det = cfg.players["detector"]
        self.model = YOLO(str(cfg.root / det["weights"]))
        self.det = det
        self.half = {"quantize": "fp16"} if str(cfg.device).startswith("cuda") else {}  # as the ball model
        self.tracker_yaml = cfg.root / cfg.players["tracker"]["config"]
        self.cfg, self.state, self.fps = cfg, state, fps
        self.sheet = load_sheet(cfg.root, match_video)
        self.tracker = None
        self.roles = None
        self.prev_thumb = None
        self.people: dict[int, list[Person]] = {}  # frame -> people; [] on frames with no main view
        self.lock = threading.Lock()
        self.counts = {"main_view_frames": 0, "people": 0, "detect_s": 0.0}
        self.main_view_wall = -1e9  # when the main view was last on screen (perf_counter)

    def _new_shot(self) -> None:
        from cricsense.identity.role_assigner import RoleAssigner, RoleOptions

        self.camera.reset()
        if self.tracker is not None:
            self.tracker.reset()
        # the worker sees about every other frame: the role assigner's speed window is in frames
        self.roles = RoleAssigner(RoleOptions(fps=self.fps / 2))

    def _store(self, frame_id: int, people: list[Person]) -> None:
        with self.lock:
            self.people[frame_id] = people
            for f in [f for f in self.people if f < frame_id - 10 * self.fps]:
                del self.people[f]

    def handle(self, frame_id: int, t: float, image: np.ndarray) -> None:
        from cricsense.passes.players import _feet_visible, _foot_point
        from cricsense.track.player_tracker import PlayerTracker

        h, w = image.shape[:2]
        if self.tracker is None:
            self.tracker = PlayerTracker.from_config(self.tracker_yaml, (w, h), self.cfg.root, self.cfg.device)
            self._new_shot()
        thumb = cv2.cvtColor(cv2.resize(image, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = self.prev_thumb is not None and float(np.abs(thumb - self.prev_thumb).mean()) > CUT_DIFF
        self.prev_thumb = thumb
        if cut:
            self._new_shot()
        H = self.camera(image, self.shows_pitch(image))
        if H is None:  # not the calibrated main view: nobody is drawn
            self._store(frame_id, [])
            return
        self.counts["main_view_frames"] += 1
        self.main_view_wall = time.perf_counter()
        t_det = time.perf_counter()
        result = self.model.predict(image, conf=self.det["conf"], imgsz=self.det["imgsz"], classes=self.det.get("classes"),
                                    device=self.cfg.device, verbose=False, **self.half)[0]
        self.counts["detect_s"] += time.perf_counter() - t_det
        tracks = self.tracker.update(result.boxes.cpu().numpy(), image, H)
        boxes = {int(row[4]): row[:4] for row in tracks}
        obs = {tid: (_foot_point(box, H) if _feet_visible(box, (w, h), self.occluders) else None, None, 0.0) for tid, box in boxes.items()}
        role_of = self.roles.update(obs)
        name, team = self._names()
        people = []
        for tid, box in boxes.items():
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            if x2 > x1 and y2 > y1:
                role = role_of.get(tid, Role.UNKNOWN)
                people.append(Person((x1, y1, x2, y2), role, name.get(role), None if role == Role.UMPIRE else team.get(role)))
        self.counts["people"] += len(people)
        self._store(frame_id, people)

    def _names(self) -> tuple[dict[Role, str], dict[Role, str]]:
        """Name and team per role, from the latest score-bar state; the keeper from the match sheet."""
        with self.state.lock:
            batters, bowler = list(self.state.batters), self.state.bowler
        name: dict[Role, str] = {}
        for n, _, _, on_strike in batters:
            name[Role.STRIKER if on_strike else Role.NON_STRIKER] = n
        if bowler:
            name[Role.BOWLER] = bowler
        team: dict[Role, str] = {}
        if self.sheet is not None:
            batting = next((t for n, *_ in batters if (t := self.sheet.team_of(n))), None)
            if batting is None and bowler and self.sheet.team_of(bowler):
                batting = self.sheet.other(self.sheet.team_of(bowler))
            if batting is not None:
                fielding = self.sheet.other(batting)
                if fielding.keeper:
                    name[Role.KEEPER] = fielding.keeper
                team = {Role.STRIKER: batting.code, Role.NON_STRIKER: batting.code}
                team |= {r: fielding.code for r in (Role.BOWLER, Role.KEEPER, Role.FIELDER, Role.UNKNOWN)}
        return name, team

    def yield_gpu(self) -> None:
        """For the ball worker: wait while the main view is on screen, when the players need the
        GPU every frame; its shot processing (after the shot) catches up between deliveries."""
        t0 = time.perf_counter()
        while time.perf_counter() - self.main_view_wall < YIELD_S and time.perf_counter() - t0 < YIELD_MAX_S:
            time.sleep(0.05)

    def at(self, frame_id: int) -> list[Person]:
        """The people found on this frame, or on the nearest frame before it within MAX_GAP."""
        with self.lock:
            for f in range(frame_id, frame_id - MAX_GAP - 1, -1):
                if f in self.people:
                    return self.people[f]
        return []


def _outcome_text(p: dict) -> str:
    if p["wicket"]:
        return "WICKET"
    if p["boundary"]:
        return p["boundary"]
    if p["kind"] != "legal":
        return f"{p['kind'].upper().replace('_', '-')} +{p['runs']}"
    return "DOT BALL" if p["runs"] == 0 else f"{p['runs']} RUN{'S' if p['runs'] > 1 else ''}"


def _bounce(events: list, lo: float, hi: float) -> dict | None:
    """The delivery's bounce among those found in (lo, hi]: a confirmed one first, then the
    earliest - a replay of the ball comes after it and is often unconfirmed."""
    found = [e for e in events if e.kind == "BOUNCE" and lo < e.t <= hi]
    found.sort(key=lambda e: (not e.payload["confirmed"], e.t))
    return found[0].payload if found else None


def ball_card(state: LiveState, t: float) -> dict | None:
    """The card for source time t: the ball under way, or the last one until the next is.

    Only what was known by t is used - the picture is behind the source, so a result shows when
    the frame it came from is on screen. The score bar shows a ball's radar speed from 4 s before
    its result to 4 s after it (Bumrah's over), and the next ball comes 20 s or more later: a
    reading within RADAR_LATE_S of a result is that ball's, a later one the next ball's. Length
    and line come 12-23 s after the ball, so the last ball's card stays until the next is under way
    (at most CARD_HOLD_S)."""
    with state.lock:
        events = [e for e in state.events if e.t <= t]
        next_ball, bowler = state.next_ball, state.bowler
        striker = next((n for n, *_, s in state.batters if s), None)
    balls = [e for e in events if e.kind == "BALL"]
    last = balls[-1] if balls else None
    last_t = last.t if last else -1e9
    before = balls[-2].t if len(balls) > 1 else -1e9
    speeds = [e for e in events if e.kind == "SPEED"]
    # the ball under way: a radar reading well after the last result, or a confirmed bounce after it
    next_radar = next((e for e in speeds if e.t > last_t + RADAR_LATE_S), None)
    # (a replay of the last ball comes within seconds of its result; the next ball 20 s or more after)
    next_bounce = next((e for e in events if e.kind == "BOUNCE" and e.t > last_t + NEXT_BALL_AFTER_S and e.payload["confirmed"]), None)
    if next_radar or next_bounce or last is None or t - last_t > CARD_HOLD_S:
        # the ball under way, or (nothing known of it yet) the one about to be bowled: its title
        title = f"{next_ball or ''}  {(bowler or '').title()}" + (f" \u2192 {striker.title()}" if striker else "")
        radar = next_radar.payload["radar_kph"] if next_radar else None
        delivery, outcome = _bounce(events, last_t + NEXT_BALL_AFTER_S, t), None
    else:
        p = last.payload
        title = f"{p['over']}  {(p['bowler'] or '').title()}" + (f" \u2192 {p['striker'].title()}" if p.get("striker") else "")
        radar = next((e.payload["radar_kph"] for e in speeds if before + RADAR_LATE_S < e.t <= last_t + RADAR_LATE_S), None)
        delivery, outcome = _bounce(events, max(before, last_t - BOUNCE_BEFORE_S), last_t), _outcome_text(p)
    if not title.strip() or (last is None and not bowler):
        return None
    if radar:
        speed = f"{radar:g} km/h  radar"
    elif delivery and delivery.get("speed_estimate_kph"):
        speed = f"\u2248 {delivery['speed_estimate_kph']} km/h  estimate"
    else:
        speed = None
    return {"title": title.strip(), "delivery": delivery, "speed": speed, "outcome": outcome}


def draw_plus(image: np.ndarray, state: LiveState, people: list[Person], t: float, stats: str) -> np.ndarray:
    from cricsense.render.draw import blend, chip, panel, paste
    from cricsense.render.overlay import CARD_W, OUTCOME_COLORS, _players
    from cricsense.render.pitch_map import LENGTH_COLORS, MapPoint, MapView, draw_pitch_map

    h, w = image.shape[:2]
    _players(image, people)
    with state.lock:
        score, batters = state.score, list(state.batters)
        bounces = [(e.payload, e.t) for e in state.events if e.kind == "BOUNCE" and e.t <= t]
    # the score, top left, under the broadcaster's logo
    rows = [f"LIVE  {score or '-'}"] + (["  ".join(f"{n.title()}{'*' if s else ''} {r} ({b})" for n, r, b, s in batters)] if batters else [])
    box = panel(440, 22 + 30 * len(rows), (18, 18, 18), 200, 12)
    for i, row in enumerate(rows):
        paste(box, chip(row, 20 if i else 22, (255, 255, 255), None, i == 0, (0, 0)), 16, 10 + 30 * i)
    blend(image, box, 28, int(0.17 * h))
    card = ball_card(state, t)
    if card is not None:
        d = card["delivery"]
        lines = []
        if d:
            extra = "" if d["confirmed"] else "  (inferred)"
            lines.append(("LENGTH", f"{d['length_class'].upper()}  {d['length_m']:.1f} m{extra}", LENGTH_COLORS[d["length_class"]]))
            side = "right" if d["line_m"] > 0 else "left"
            lines.append(("LINE", f"{abs(d['line_m']) * 100:.0f} cm {side} of middle" if abs(d["line_m"]) >= 0.05 else "on middle stump", (235, 235, 235)))
        if card["speed"]:
            lines.append(("SPEED", card["speed"], (235, 235, 235)))
        out = card["outcome"]
        panel_img = panel(CARD_W, 58 + 34 * len(lines) + (46 if out else 0), (18, 18, 18), 200, 12)
        paste(panel_img, chip(card["title"], 22, (255, 255, 255), None, True, (0, 0)), 18, 14)
        y = 54
        for label, value, color in lines:
            paste(panel_img, chip(label, 15, (150, 150, 150), None, True, (0, 0)), 18, y + 4)
            paste(panel_img, chip(value, 19, color, None, False, (0, 0)), 110, y)
            y += 34
        if out:
            paste(panel_img, chip(out, 20, (255, 255, 255), OUTCOME_COLORS.get(out, (70, 70, 70)), True, (12, 5), 235, 6), 18, y + 4)
        blend(image, panel_img, w - CARD_W - 28, 28)
    if bounces:
        pts = [MapPoint(str(i + 1), p["length_m"], p["line_m"], p["length_class"], p["confirmed"]) for i, (p, _) in enumerate(bounces[-12:])]
        img = draw_pitch_map(pts, MapView(px_per_m=24.0, x_range=(-1.0, 12.0), margin_m=1.0, gutter_px=16), highlight=len(pts) - 1, label_only_highlight=True)
        blend(image, img, w - img.shape[1] - 28, int(0.85 * h) - img.shape[0] - 18)
    blend(image, chip(stats, 16, (220, 220, 220), (10, 10, 10), False, (8, 3), 170, 4), 20, 20)
    return image


def run_live_plus(
    cfg: Config, source: str, out: Path | None, calibration_yaml: Path, seconds: float | None = None,
    realtime: bool = True, show: bool = True,
) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    state = LiveState()
    buffer: deque = deque(maxlen=int(BUFFER_S * fps))
    match_video = yaml.safe_load(Path(calibration_yaml).read_text()).get("video_id") or Path(str(source)).stem
    score = ScoreWorker(cfg, state, 0.0, match_video)
    vb = ViewBallWorker(cfg, state, 0.0, calibration_yaml, fps, cfg.device, buffer)
    players = PlayersWorker(cfg, state, calibration_yaml, fps, match_video)
    vb.gate = players.yield_gpu
    workers = [score, vb, players]
    start = time.perf_counter()  # the clock starts once the models are loaded
    for wk in workers:
        wk.t0 = start
        wk.start()
    writer = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    window = Window(w, h, fps) if show else None
    delayed: deque = deque()  # frames read but not yet shown
    delay_frames = int(round(DELAY_S * fps))
    frame_id, drawn, late = 0, 0, 0
    score_every = max(1, int(round(fps / 2)))

    def show_one() -> None:
        nonlocal drawn
        f, ft, image = delayed.popleft()
        stats = f"live  {ft:6.1f}s  ({DELAY_S:g} s behind)  players on {players.processed} frames"
        frame = draw_plus(image.copy(), state, players.at(f), ft, stats)  # the buffer's frame is the ball worker's
        if writer is not None:
            writer.write(frame)
        if window is not None:
            window.show(frame)
        drawn += 1

    while True:
        ok, image = cap.read()
        if not ok:
            break
        t = frame_id / fps
        if seconds is not None and t > seconds:
            break
        if realtime:
            wait = t - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.2:
                late += 1
        buffer.append((frame_id, t, image))
        if frame_id % score_every == 0:
            score.offer((frame_id, t, image))
        if frame_id % 2 == 0:
            vb.offer((frame_id, t, image))
        players.offer((frame_id, t, image))
        delayed.append((frame_id, t, image))
        if len(delayed) > delay_frames:
            show_one()
        frame_id += 1
    while delayed:
        show_one()
    log.info("the source has ended: finishing the last deliveries (up to 2 min; Ctrl+C to skip)")
    for wk in workers:
        wk.inbox.put(None)
    for wk in workers:
        wk.done.wait(timeout=120)
    unfinished = [wk.name for wk in workers if not wk.done.is_set()]
    cap.release()
    if writer is not None:
        writer.release()
    if window is not None:
        window.close()
    wall = time.perf_counter() - start
    return {
        "source": source, "frames": frame_id, "source_seconds": round(frame_id / fps, 1), "wall_seconds": round(wall, 1),
        "frames_drawn": drawn, "reader_late_frames": late, "delay_s": DELAY_S, "unfinished": unfinished,
        "workers": {wk.name: {"processed": wk.processed, "dropped": wk.dropped, "rate_per_s": round(wk.processed / max(wall, 1e-6), 2),
                              **getattr(wk, "counts", {}), **getattr(getattr(wk, "camera", None), "counts", {})} for wk in workers},
        "events": [{"kind": e.kind, "t": round(e.t, 2), "latency_s": round(e.latency_s, 2), **e.payload} for e in state.events],
    }
