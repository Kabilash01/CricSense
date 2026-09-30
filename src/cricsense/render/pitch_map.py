"""Top-down pitch map: where each ball pitched, coloured by length.

Seen from the bowler's end, like the broadcast: the striker's stumps at the top, +line to the
right (as on screen). A filled marker is a confirmed bounce (the ball was seen after
pitching); a ring is unconfirmed. The same drawing serves the in-video inset and the
summary pitch_map.png.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cricsense.calibration.pitch_model import PITCH_HALF_WIDTH, POPPING_CREASE, RETURN_CREASE_Y, STUMP_HALF_SPAN
from cricsense.render.draw import chip, paste

LENGTH_COLORS = {  # BGR
    "yorker": (220, 110, 240),
    "full": (40, 190, 255),
    "good": (90, 210, 90),
    "short": (80, 80, 235),
}
SS = 3  # supersampling for anti-aliased shapes


@dataclass(frozen=True)
class MapPoint:
    label: str  # "11.5"
    length_m: float
    line_m: float
    length_class: str
    confirmed: bool


@dataclass(frozen=True)
class MapView:
    px_per_m: float = 28.0
    x_range: tuple[float, float] = (-1.0, 13.0)  # metres from the striker's stumps
    margin_m: float = 1.1  # either side of the pitch
    zones: tuple[float, float, float] = (2.0, 6.0, 8.0)  # yorker | full | good | short
    gutter_px: int = 0  # room on the left for the zone names

    @property
    def size(self) -> tuple[int, int]:
        w = (2 * PITCH_HALF_WIDTH + 2 * self.margin_m) * self.px_per_m + self.gutter_px
        h = (self.x_range[1] - self.x_range[0]) * self.px_per_m
        return int(round(w)), int(round(h))

    def to_px(self, length_m: float, line_m: float, scale: float = 1.0) -> tuple[int, int]:
        u = (line_m + PITCH_HALF_WIDTH + self.margin_m) * self.px_per_m + self.gutter_px
        v = (length_m - self.x_range[0]) * self.px_per_m
        return int(round(u * scale)), int(round(v * scale))


def draw_pitch_map(
    points: list[MapPoint],
    view: MapView = MapView(),
    highlight: int | None = None,
    labels: bool = True,
    bg_alpha: int = 200,
    label_only_highlight: bool = False,
) -> np.ndarray:
    """The map as a BGRA image. highlight: index into points drawn larger, on top."""
    w, h = view.size
    big = np.zeros((h * SS, w * SS, 4), np.uint8)
    big[..., :3] = (28, 32, 30)
    big[..., 3] = bg_alpha
    s = view.px_per_m * SS

    def P(x: float, y: float) -> tuple[int, int]:
        return view.to_px(x, y, SS)

    # the pitch and its length bands
    edges = [view.x_range[0], 0.0, *view.zones, view.x_range[1]]
    tints = [(78, 88, 94), (92, 80, 98), (72, 96, 110), (78, 100, 82), (78, 78, 104)]  # muted, so markers stand out
    for (a, b), tint in zip(zip(edges, edges[1:]), tints):
        cv2.rectangle(big, (*P(a, -PITCH_HALF_WIDTH),), (*P(b, PITCH_HALF_WIDTH),), (*tint, 255), -1)
    for z in view.zones:
        cv2.line(big, P(z, -PITCH_HALF_WIDTH), P(z, PITCH_HALF_WIDTH), (200, 200, 200, 255), max(1, SS // 2), cv2.LINE_AA)
    white = (245, 245, 245, 255)
    lw = max(2, int(0.05 * s))
    cv2.line(big, P(POPPING_CREASE, -PITCH_HALF_WIDTH - 0.3), P(POPPING_CREASE, PITCH_HALF_WIDTH + 0.3), white, lw, cv2.LINE_AA)
    cv2.line(big, P(0.0, -RETURN_CREASE_Y - 0.4), P(0.0, RETURN_CREASE_Y + 0.4), white, max(1, lw // 2), cv2.LINE_AA)
    for y in (-RETURN_CREASE_Y, RETURN_CREASE_Y):
        cv2.line(big, P(view.x_range[0], y), P(POPPING_CREASE, y), white, max(1, lw // 2), cv2.LINE_AA)
    for y in (-STUMP_HALF_SPAN, 0.0, STUMP_HALF_SPAN):
        cv2.circle(big, P(0.0, y), max(2, int(0.045 * s)), (60, 200, 250, 255), -1, cv2.LINE_AA)

    order = [i for i in range(len(points)) if i != highlight] + ([highlight] if highlight is not None else [])
    r = max(4 * SS, int(0.16 * s))
    for i in order:
        p = points[i]
        if not (view.x_range[0] <= p.length_m <= view.x_range[1]):
            continue
        c = (*LENGTH_COLORS.get(p.length_class, (200, 200, 200)), 255)
        rr = int(r * 1.45) if i == highlight else r
        centre = P(p.length_m, p.line_m)
        if p.confirmed:
            cv2.circle(big, centre, rr, c, -1, cv2.LINE_AA)
            cv2.circle(big, centre, rr, (20, 20, 20, 255), max(1, SS), cv2.LINE_AA)
        else:
            cv2.circle(big, centre, rr, c, max(2, int(0.35 * rr)), cv2.LINE_AA)
        if i == highlight:
            cv2.circle(big, centre, rr + 2 * SS, (255, 255, 255, 255), max(2, SS), cv2.LINE_AA)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)

    fs = max(10, int(view.px_per_m * 0.42))
    names = ("YORKER", "FULL", "GOOD", "SHORT") if view.gutter_px >= 4 * fs else ("Y", "F", "G", "S")
    for name, (a, b) in zip(names, zip([0.0, *view.zones], [*view.zones, view.x_range[1]])):
        paste(img, chip(name, fs, (225, 225, 225), None, True, (2, 0)), 3, int(((a + b) / 2 - view.x_range[0]) * view.px_per_m) - fs // 2)
    if labels:
        taken: list[tuple[int, int, int, int]] = []
        for i in reversed(order):  # the highlighted ball gets the first pick
            p = points[i]
            if label_only_highlight and i != highlight:
                continue
            if not (view.x_range[0] <= p.length_m <= view.x_range[1]):
                continue
            u, v = view.to_px(p.length_m, p.line_m)
            tag = chip(p.label, fs, (255, 255, 255), (15, 15, 15), i == highlight, (3, 1), 170, 3)
            th, tw = tag.shape[:2]
            gap = r // SS + 3
            right, left = (u + gap, v - th // 2), (u - gap - tw, v - th // 2)
            first, second = (right, left) if p.line_m <= 0 else (left, right)  # away from the middle stump first
            spots = [first, second] + [(x, y + dy) for dy in (-th, th, -2 * th, 2 * th) for x, y in (first, second)]
            for x, y in spots:
                box = (x, y, x + tw, y + th)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                    break
            taken.append(box)
            paste(img, tag, x, y)
    return img


COLUMNS = ((0, "BALL"), (70, "BOWLER \u2192 BATTER"), (300, "LENGTH"), (440, "LINE"), (560, "SPEED"), (760, "RESULT"))


def summary_png(points: list[MapPoint], rows: list[tuple[str, ...]], title: str) -> np.ndarray:
    """pitch_map.png: the map with every delivery, a legend and a table, one row per delivery.

    rows: (ball, bowler -> batter, length, line, speed, result) strings.
    """
    view = MapView(px_per_m=46.0, gutter_px=96)
    m = draw_pitch_map(points, view, labels=True, bg_alpha=255)
    mh, mw = m.shape[:2]
    side = 900
    canvas = np.zeros((max(mh, 150 + 34 * len(rows)) + 90, mw + side + 60, 4), np.uint8)
    canvas[..., :3] = (22, 24, 26)
    canvas[..., 3] = 255
    paste(canvas, chip(title, 28, (255, 255, 255), None, True, (0, 0)), 30, 22)
    paste(canvas, m, 30, 70)
    x0, y = mw + 60, 76
    x = x0
    for name, c in LENGTH_COLORS.items():
        cv2.circle(canvas, (x + 9, y + 12), 8, (*c, 255), -1, cv2.LINE_AA)
        paste(canvas, chip(name.capitalize(), 18, (230, 230, 230), None, False, (0, 0)), x + 24, y)
        x += 130
    y += 34
    paste(canvas, chip("filled: bounce seen    ring / *: ball not seen after pitching    \u2248: our estimate", 16, (170, 170, 170), None, False, (0, 0)), x0, y)
    y += 44
    for dx, name in COLUMNS:
        paste(canvas, chip(name, 15, (150, 150, 150), None, True, (0, 0)), x0 + dx, y)
    y += 30
    for row in rows:
        for (dx, _), cell in zip(COLUMNS, row):
            paste(canvas, chip(cell, 18, (235, 235, 235), None, False, (0, 0)), x0 + dx, y)
        y += 34
    return np.ascontiguousarray(canvas[..., :3])
