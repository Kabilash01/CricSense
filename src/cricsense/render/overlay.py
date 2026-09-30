"""Per-frame overlay: players, the ball's flight, the bounce, a delivery card, event banners
and the pitch-map inset, all from the pass outputs (no model runs here).

Measured values and inferred ones are drawn differently, because the event log says which is
which: an unconfirmed bounce is a ring and "inferred"; our speed is "≈ ... estimate"; the
broadcaster's radar is "radar".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from cricsense.render.draw import blend, chip, panel, paste
from cricsense.render.pitch_map import LENGTH_COLORS, MapPoint, MapView, draw_pitch_map
from cricsense.render.wagon_wheel import Shot, draw_wagon_wheel
from cricsense.schema import BallDoc, CameraDoc, EventsDoc, EventType, PlayersDoc, Role

TEAM_COLORS = {"IND": (230, 130, 30), "SA": (70, 165, 40)}  # BGR
UMPIRE_COLOR = (170, 170, 170)
OTHER_COLOR = (220, 220, 220)
ROLE_NAMES = {
    Role.STRIKER: "striker",
    Role.NON_STRIKER: "non-striker",
    Role.BOWLER: "bowler",
    Role.KEEPER: "keeper",
    Role.UMPIRE: "umpire",
    Role.FIELDER: "fielder",
}
OUTCOME_COLORS = {"SIX": (200, 60, 230), "FOUR": (235, 150, 30), "WICKET": (50, 50, 230)}
TRAIL_FRAMES = 14
CARD_HOLD_S = 4.0  # the card stays this long after the ball's outcome
BANNER_S = 3.0


@dataclass
class DeliveryView:
    did: int
    release_frame: int
    end_frame: int
    title: str
    bounce_frame: int | None
    bounce_xy: tuple[float, float] | None
    length_m: float | None
    line_m: float | None
    length_class: str | None
    confirmed: bool
    speed: str | None
    speed_source: str | None
    outcome: str | None = None
    outcome_frame: int | None = None
    track: dict[int, tuple[float, float]] = field(default_factory=dict)
    release_y: float = 0.0  # sideways position of the release, metres
    path: BallPath | None = None


@dataclass
class BallPath:
    """The delivery's flight as a smooth curve, in one reference view (see ball_path)."""

    ref_frame: int
    before: np.ndarray  # (n, 3): t, u, v - release to bounce
    after: np.ndarray  # bounce onwards (may be empty)
    seen_from: float = 0.0  # before this time the curve is extrapolated to the hand, not seen


class Scene:
    """Everything the overlay needs, indexed by frame."""

    def __init__(
        self,
        ball: BallDoc,
        events: EventsDoc,
        players: PlayersDoc | None,
        camera: CameraDoc | None,
        size: tuple[int, int],
    ) -> None:
        self.fps = ball.video.fps
        self.w, self.h = size
        self.captions: list[tuple[float, float, str]] = []  # (start s, end s, text) from voice.json
        self.shots: dict[int, str] = {}  # delivery -> "slog sweep, leg side" (commentary.json, read by the VLM)
        self.skeletons: dict[int, np.ndarray] = {}  # frame -> striker COCO-17 keypoints (pose.json)
        self.batter: dict[int, str] = {}  # delivery -> "back foot 0.9 m, played" (pose.json)
        self.wagon: list[tuple[int, Shot]] = []  # (delivery, scoring shot) for the wagon wheel
        self.players: dict[int, list] = defaultdict(list)
        if players:
            for p in players.frames:
                self.players[p.frame_id].append(p)
        self.pitch_to_image: dict[int, np.ndarray] = {}
        self.image_to_pitch: dict[int, np.ndarray] = {}
        self.projection: dict[int, np.ndarray] = {}
        if camera:
            for c in camera.frames:
                if c.projection is not None:
                    self.projection[c.frame_id] = np.array(c.projection)
                if c.image_to_pitch:
                    self.image_to_pitch[c.frame_id] = np.array(c.image_to_pitch)
                    self.pitch_to_image[c.frame_id] = np.linalg.inv(self.image_to_pitch[c.frame_id])
        by_delivery: dict[int, dict[EventType, object]] = defaultdict(dict)
        for e in events.events:
            if "delivery_id" in e.payload:
                by_delivery[e.payload["delivery_id"]][e.type] = e
        self.banners: list[tuple[int, str, tuple[int, int, int]]] = []
        self.deliveries: list[DeliveryView] = []
        tracks: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        for f in ball.frames:
            if f.delivery_id is not None:
                tracks[f.delivery_id][f.frame_id] = f.uv
        n = len(ball.deliveries)
        for i, d in enumerate(ball.deliveries):
            evs = by_delivery.get(d.delivery_id, {})
            rel = evs.get(EventType.RELEASE)
            p = rel.payload if rel else {}
            title = f"{p['over']}  {p['bowler']} → {p['striker']}" if "bowler" in p else f"Delivery {d.delivery_id + 1}"
            if p.get("radar_speed_kph"):
                speed, source = f"{p['radar_speed_kph']:g} km/h", "radar"
            elif p.get("speed_estimate_kph"):
                speed, source = f"≈{p['speed_estimate_kph']} km/h", "estimate"
            else:
                speed, source = None, None
            runs = evs.get(EventType.RUNS)
            outcome, outcome_frame = None, None
            if runs:
                q = runs.payload
                outcome_frame = runs.frame_id
                if EventType.WICKET in evs:
                    outcome = "WICKET"
                elif EventType.BOUNDARY in evs:
                    outcome = evs[EventType.BOUNDARY].payload["kind"]
                elif q["kind"] == "wide":
                    outcome = f"WIDE +{q['runs']}"
                elif q["kind"] == "no_ball":
                    outcome = f"NO BALL +{q['runs']}"
                else:
                    outcome = "DOT BALL" if q["runs"] == 0 else f"{q['runs']} RUN" + ("S" if q["runs"] > 1 else "")
                if outcome in OUTCOME_COLORS:
                    who = q.get("striker") if outcome != "WICKET" else q.get("striker")
                    self.banners.append((outcome_frame, f"{outcome}  {who}" if who else outcome, OUTCOME_COLORS[outcome]))
            next_release = ball.deliveries[i + 1].release_frame if i + 1 < n else None
            end = (outcome_frame + int(CARD_HOLD_S * self.fps)) if outcome_frame else d.shot_frames[1] + int(3 * self.fps)
            if next_release is not None:
                end = min(end, next_release - 1)
            self.deliveries.append(
                DeliveryView(
                    did=d.delivery_id,
                    release_frame=d.release_frame,
                    end_frame=end,
                    title=title,
                    bounce_frame=None if d.bounce_t is None else int(round(d.bounce_t * self.fps)),
                    bounce_xy=d.bounce_xy,
                    length_m=d.length_m,
                    line_m=d.line_m,
                    length_class=d.length_class.value if d.length_class else None,
                    confirmed=d.bounce_confirmed,
                    speed=speed,
                    speed_source=source,
                    outcome=outcome,
                    outcome_frame=outcome_frame,
                    track=tracks.get(d.delivery_id, {}),
                    release_y=d.release_point[1],
                )
            )
        for dv in self.deliveries:
            dv.path = ball_path(dv, self)
        self.map_points = [
            MapPoint(dv.title.split()[0] if "→" in dv.title else str(dv.did + 1), dv.length_m, dv.line_m, dv.length_class, dv.confirmed)
            for dv in self.deliveries
            if dv.length_m is not None
        ]
        self._map_index = {dv.did: k for k, dv in enumerate(dv for dv in self.deliveries if dv.length_m is not None)}

    def active(self, f: int) -> DeliveryView | None:
        for dv in reversed(self.deliveries):
            if dv.release_frame - int(0.3 * self.fps) <= f <= dv.end_frame:
                return dv
        return None


SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def _skeleton(frame: np.ndarray, kp: np.ndarray) -> None:
    for a, b in SKELETON:
        if kp[a, 2] > 0.4 and kp[b, 2] > 0.4:
            cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), (255, 255, 255), 2, cv2.LINE_AA)
    for j in range(5, 17):
        if kp[j, 2] > 0.4:
            cv2.circle(frame, (int(kp[j, 0]), int(kp[j, 1])), 3, (0, 220, 255), -1, cv2.LINE_AA)


def compose(frame: np.ndarray, f: int, scene: Scene) -> np.ndarray:
    _players(frame, scene.players.get(f, []))
    if f in scene.skeletons:
        _skeleton(frame, scene.skeletons[f])
    dv = scene.active(f)
    if dv is not None:
        _ball(frame, f, dv, scene)
        _card(frame, f, dv, scene)
        _inset(frame, f, dv, scene)
    for start, text, color in scene.banners:
        if start <= f < start + int(BANNER_S * scene.fps):
            _banner(frame, text, color, (f - start) / scene.fps)
    t = f / scene.fps
    for start, end, text in scene.captions:
        if start <= t < end + 0.4:
            _caption(frame, text)
            break
    return frame


def _caption(frame: np.ndarray, text: str) -> None:
    """The line being spoken, above the score bar; wrapped to two lines at most."""
    h, w = frame.shape[:2]
    words, rows, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > 70 and cur:
            rows.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    rows.append(cur)
    y = int(0.85 * h) - 14 - 38 * len(rows[:2])
    for row in rows[:2]:
        tag = chip(row, 24, (255, 255, 255), (10, 10, 10), False, (12, 4), 175, 6)
        blend(frame, tag, w // 2 - tag.shape[1] // 2, y)
        y += 38


def _players(frame: np.ndarray, people: list) -> None:
    for p in people:
        x1, y1, x2, y2 = p.box
        color = UMPIRE_COLOR if p.role == Role.UMPIRE else TEAM_COLORS.get(p.team or "", OTHER_COLOR)
        L = max(8, (x2 - x1) // 4)
        for (ax, ay), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)), ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
            cv2.line(frame, (ax, ay), (ax + dx * L, ay), color, 2, cv2.LINE_AA)
            cv2.line(frame, (ax, ay), (ax, ay + dy * L), color, 2, cv2.LINE_AA)
        role = ROLE_NAMES.get(p.role)
        if p.identity:
            text = p.identity.title() + (f" · {role}" if role and p.role != Role.FIELDER else "")
        elif role and p.role != Role.FIELDER:
            text = role
        else:
            continue
        tag = chip(text, 17, (255, 255, 255), color, True, (6, 2), 215, 5)
        blend(frame, tag, (x1 + x2) // 2 - tag.shape[1] // 2, y1 - tag.shape[0] - 4)


PATH_STEPS = 60
RELEASE_X, RELEASE_Z = 18.6, 2.1  # the hand at release: just past the popping crease, ~2.1 m up


def ball_path(dv: DeliveryView, scene: Scene) -> BallPath | None:
    """Fit the flight in one reference view.

    The broadcast camera turns and zooms but stays put, so one homography maps any point
    between two of its views, whatever its height: every detection is carried into the
    reference view, a smooth curve is fitted before and after the bounce (they meet at the
    measured bounce point), and at draw time the curve is carried into the current view.
    """
    pts = [(g, uv) for g, uv in sorted(dv.track.items()) if g in scene.image_to_pitch]
    if len(pts) < 3:
        return None
    ref = pts[0][0]
    to_ref = scene.pitch_to_image[ref]
    obs = []
    for g, (u, v) in pts:
        p = to_ref @ scene.image_to_pitch[g] @ np.array([u, v, 1.0])
        obs.append((g / scene.fps, p[0] / p[2], p[1] / p[2]))
    obs = np.array(obs)
    tb = None if dv.bounce_frame is None else dv.bounce_frame / scene.fps
    seen_from = obs[0, 0]
    # The detector first finds the ball a few frames after it leaves the hand (it is blurred
    # against the bowler). Join the curve to the release point, placed by the 3D camera, at
    # the time the ball's image speed puts it there; drawn dashed, as it was not seen.
    Pr = scene.projection.get(ref)
    if Pr is not None and len(obs) >= 3:
        x = Pr @ np.array([RELEASE_X, dv.release_y, RELEASE_Z, 1.0])
        rel_uv = x[:2] / x[2]
        speed = np.linalg.norm(obs[2, 1:] - obs[0, 1:]) / max(obs[2, 0] - obs[0, 0], 1e-3)
        gap = np.linalg.norm(rel_uv - obs[0, 1:]) / max(speed, 1e-3)
        if 0.02 < gap < 0.4 and rel_uv[1] < obs[0, 2] + 40:  # plausible: close in time, not below the flight
            obs = np.vstack([[obs[0, 0] - gap, *rel_uv], obs])
    anchor = None
    if dv.bounce_xy is not None and tb is not None:
        q = to_ref @ np.array([dv.bounce_xy[0], dv.bounce_xy[1], 1.0])
        anchor = np.array([tb, q[0] / q[2], q[1] / q[2]])

    def fit(seg: np.ndarray, t0: float, t1: float) -> np.ndarray:
        if len(seg) < 2 or t1 <= t0:
            return np.empty((0, 3))
        deg = 2 if len(seg) >= 4 else 1
        w = np.ones(len(seg))
        if anchor is not None:
            w[np.isclose(seg[:, 0], anchor[0])] = 25.0  # pass through the measured bounce
        ts = np.linspace(t0, t1, PATH_STEPS)
        cu = np.polyfit(seg[:, 0], seg[:, 1], deg, w=w)
        cv = np.polyfit(seg[:, 0], seg[:, 2], deg, w=w)
        return np.stack([ts, np.polyval(cu, ts), np.polyval(cv, ts)], axis=1)

    if tb is None:
        return BallPath(ref, fit(obs, obs[0, 0], obs[-1, 0]), np.empty((0, 3)), seen_from)
    pre = obs[obs[:, 0] <= tb]
    post = obs[obs[:, 0] > tb]
    if anchor is not None:
        pre = np.vstack([pre, anchor])
        post = np.vstack([anchor, post])
    return BallPath(ref, fit(pre, obs[0, 0], tb), fit(post, tb, obs[-1, 0]) if len(post) >= 2 else np.empty((0, 3)), seen_from)


def _draw_path(frame: np.ndarray, f: int, dv: DeliveryView, scene: Scene) -> None:
    path = dv.path
    P = scene.image_to_pitch.get(f)
    if path is None or P is None:
        return  # another camera: the path cannot be placed in it
    H = scene.pitch_to_image[f] @ scene.image_to_pitch[path.ref_frame]
    t_now = f / scene.fps
    color_after = LENGTH_COLORS.get(dv.length_class or "", (255, 255, 255))
    for seg, color in ((path.before, (40, 235, 255)), (path.after, color_after)):
        seg = seg[seg[:, 0] <= t_now] if len(seg) else seg
        if len(seg) < 2:
            continue
        xy = H @ np.stack([seg[:, 1], seg[:, 2], np.ones(len(seg))])
        pts = (xy[:2] / xy[2]).T.astype(np.int32)
        unseen = seg[:, 0] < path.seen_from
        solid = pts[~unseen | np.roll(~unseen, -1)]  # from the first seen point on
        if len(solid) >= 2:
            cv2.polylines(frame, [solid.reshape(-1, 1, 2)], False, (20, 20, 20), 9, cv2.LINE_AA)
            cv2.polylines(frame, [solid.reshape(-1, 1, 2)], False, color, 4, cv2.LINE_AA)
        dashed = pts[unseen]
        for k in range(0, len(dashed) - 1, 2):  # extrapolated to the hand: dashed
            cv2.line(frame, tuple(dashed[k]), tuple(dashed[k + 1]), color, 2, cv2.LINE_AA)


def _ball(frame: np.ndarray, f: int, dv: DeliveryView, scene: Scene) -> None:
    _draw_path(frame, f, dv, scene)
    pts = [(g, uv) for g, uv in sorted(dv.track.items()) if f - TRAIL_FRAMES <= g <= f]
    for (g0, a), (g1, b) in zip(pts, pts[1:]):
        if g1 - g0 > 4:
            continue
        k = 1.0 - (f - g1) / TRAIL_FRAMES
        color = (int(60 + 180 * k), int(230 * k + 25), 255)
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, max(1, int(4 * k)), cv2.LINE_AA)
    if f in dv.track:
        u, v = dv.track[f]
        cv2.circle(frame, (int(u), int(v)), 9, (0, 240, 255), 2, cv2.LINE_AA)
    if dv.bounce_frame is not None and dv.bounce_xy is not None and dv.bounce_frame <= f <= dv.bounce_frame + int(2.5 * scene.fps):
        H = scene.pitch_to_image.get(f)
        if H is None:
            return
        # a ground ellipse: project a 0.25 m circle on the pitch into this frame
        ang = np.linspace(0, 2 * np.pi, 40)
        ring = np.stack([dv.bounce_xy[0] + 0.25 * np.cos(ang), dv.bounce_xy[1] + 0.25 * np.sin(ang), np.ones_like(ang)])
        img = H @ ring
        img = (img[:2] / img[2]).T.astype(np.int32)
        color = LENGTH_COLORS.get(dv.length_class or "", (255, 255, 255))
        if dv.confirmed:
            cv2.polylines(frame, [img], True, color, 3, cv2.LINE_AA)
        else:  # inferred: dashed
            for k in range(0, len(img) - 1, 4):
                cv2.line(frame, tuple(img[k]), tuple(img[k + 1]), color, 3, cv2.LINE_AA)


CARD_W = 440


def _card(frame: np.ndarray, f: int, dv: DeliveryView, scene: Scene) -> None:
    rows: list[tuple[str, str, tuple[int, int, int]]] = []
    bounced = dv.bounce_frame is not None and f >= dv.bounce_frame
    if bounced and dv.length_class:
        extra = "" if dv.confirmed else "  (inferred)"
        rows.append(("LENGTH", f"{dv.length_class.upper()}  {dv.length_m:.1f} m{extra}", LENGTH_COLORS[dv.length_class]))
        side = "right" if dv.line_m > 0 else "left"
        rows.append(("LINE", f"{abs(dv.line_m) * 100:.0f} cm {side} of middle" if abs(dv.line_m) >= 0.05 else "on middle stump", (235, 235, 235)))
    if dv.speed and (bounced or dv.bounce_frame is None and f > dv.release_frame + int(0.8 * scene.fps)):
        rows.append(("SPEED", f"{dv.speed}  {dv.speed_source}", (235, 235, 235)))
    shot = scene.shots.get(dv.did)
    if shot and f > dv.release_frame + int(1.0 * scene.fps):
        rows.append(("SHOT", f"{shot}  (AI read)", (235, 235, 235)))
    batter = scene.batter.get(dv.did)
    if batter and f > dv.release_frame + int(1.0 * scene.fps):
        rows.append(("BATTER", batter, (235, 235, 235)))
    show_outcome = dv.outcome and dv.outcome_frame is not None and f >= dv.outcome_frame
    height = 58 + 34 * len(rows) + (46 if show_outcome else 0)
    card = panel(CARD_W, height, (18, 18, 18), 200, 12)
    paste(card, chip(dv.title, 22, (255, 255, 255), None, True, (0, 0)), 18, 14)
    y = 54
    for label, value, color in rows:
        paste(card, chip(label, 15, (150, 150, 150), None, True, (0, 0)), 18, y + 4)
        paste(card, chip(value, 19, color, None, False, (0, 0)), 110, y)
        y += 34
    if show_outcome:
        color = OUTCOME_COLORS.get(dv.outcome, (70, 70, 70))
        paste(card, chip(dv.outcome, 20, (255, 255, 255), color, True, (12, 5), 235, 6), 18, y + 4)
    blend(frame, card, scene.w - CARD_W - 28, 28)


INSET = MapView(px_per_m=28.0, x_range=(-1.0, 12.0), margin_m=1.0, gutter_px=18)


WAGON_SIZE = 330


def _inset(frame: np.ndarray, f: int, dv: DeliveryView, scene: Scene) -> None:
    scored = [k for k, (did, _) in enumerate(scene.wagon) if did == dv.did]
    if scored and dv.outcome_frame is not None and f >= dv.outcome_frame:
        # after a scoring shot: where it went, with the shots before it
        upto = [shot for did, shot in scene.wagon if did <= dv.did]
        img = draw_wagon_wheel(upto, WAGON_SIZE, highlight=len(upto) - 1)
        blend(frame, img, scene.w - WAGON_SIZE - 20, int(0.85 * scene.h) - WAGON_SIZE - 14)
        return
    shown = [
        k
        for k, d in enumerate(d for d in scene.deliveries if d.length_m is not None)
        if d.bounce_frame is not None and d.bounce_frame <= f
    ]
    if not shown:
        return
    pts = [scene.map_points[k] for k in shown]
    idx = scene._map_index.get(dv.did)
    highlight = shown.index(idx) if idx in shown else None
    img = draw_pitch_map(pts, INSET, highlight=highlight, labels=True, bg_alpha=190, label_only_highlight=True)
    h, w = img.shape[:2]
    blend(frame, img, scene.w - w - 28, int(0.85 * scene.h) - h - 18)


def _banner(frame: np.ndarray, text: str, color: tuple[int, int, int], age_s: float) -> None:
    tag = chip(text, 46, (255, 255, 255), color, True, (28, 10), 230, 12)
    if age_s < 0.25:  # quick fade in
        tag = tag.copy()
        tag[..., 3] = (tag[..., 3] * (age_s / 0.25)).astype(np.uint8)
    h, w = frame.shape[:2]
    blend(frame, tag, w // 2 - tag.shape[1] // 2, int(0.12 * h))
