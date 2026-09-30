"""Wagon wheel: where each scoring shot went, on a top-down field.

The direction comes from the shot (named from the posture the VLM saw, with measured
footwork - commentary/vlm.py) at its textbook angle, mirrored for a left-hander (the batting
hand is measured from the stance - pose/batter.py). The length comes from the runs: a four or
a six reaches the rope. The broadcast does not show where the ball went precisely, so this
is an approximation and is labelled as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from cricsense.render.draw import chip, paste

# degrees from straight back past the bowler; + to the batter's off side, - to the leg side
SHOT_ANGLE = {
    "straight drive": 0, "cover drive": 45, "drive": 35, "back-foot punch": 55, "cut": 90, "reverse sweep": 120,
    "flick": -45, "clip to leg": -55, "slog sweep": -60, "pull": -85, "sweep": -100, "hook": -100, "glance": -150,
}
REACH = {1: 0.35, 2: 0.55, 3: 0.75, 4: 1.0, 5: 1.0, 6: 1.12}  # of the boundary radius
RUN_COLORS = {1: (230, 230, 230), 2: (235, 190, 60), 3: (235, 190, 60), 4: (235, 150, 30), 6: (200, 60, 230)}  # BGR
BOUNDARY_M = 65.0


@dataclass(frozen=True)
class Shot:
    label: str  # "11.5"
    shot: str
    runs: int
    hand: str  # right | left


def endpoint(s: Shot) -> tuple[float, float] | None:
    """(x, y) metres from the striker's stumps: x towards the bowler, y to the right as seen from behind him."""
    if s.shot not in SHOT_ANGLE or s.runs <= 0:
        return None
    a = math.radians(SHOT_ANGLE[s.shot])
    off = -1.0 if s.hand == "right" else 1.0  # seen from behind the bowler a right-hander's off side is on the left
    r = BOUNDARY_M * REACH.get(s.runs, 1.0)
    return r * math.cos(a), off * r * math.sin(a)


def draw_wagon_wheel(shots: list[Shot], size: int = 520, highlight: int | None = None, bg_alpha: int = 215) -> np.ndarray:
    """BGRA image, viewed like the broadcast: the batter's end at the top, the bowler's end below."""
    SS = 3
    S = size * SS
    img = np.zeros((S, S, 4), np.uint8)
    c = S // 2
    scale = (S / 2 - 14 * SS) / (BOUNDARY_M * 1.15)
    cv2.circle(img, (c, c), int(BOUNDARY_M * 1.15 * scale), (40, 60, 38, bg_alpha), -1, cv2.LINE_AA)  # the outfield
    cv2.circle(img, (c, c), int(BOUNDARY_M * scale), (235, 235, 235, 255), max(2, SS), cv2.LINE_AA)  # the rope
    cv2.circle(img, (c, c), int(27 * scale), (90, 110, 90, 255), max(1, SS // 2), cv2.LINE_AA)  # 30-yard circle
    top, bottom = c - int(10.06 * scale), c + int(10.06 * scale)  # the pitch, striker at the top
    cv2.rectangle(img, (c - int(1.5 * scale), top), (c + int(1.5 * scale), bottom), (120, 170, 200, 255), -1)
    origin = (c, top)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(origin[0] + y * scale), int(origin[1] + x * scale)

    order = [i for i in range(len(shots)) if i != highlight] + ([highlight] if highlight is not None else [])
    ends_origin = (origin[0] // SS, origin[1] // SS)
    ends = {}
    for i in order:
        e = endpoint(shots[i])
        if e is None:
            continue
        color = (*RUN_COLORS.get(shots[i].runs, (230, 230, 230)), 255)
        width = (5 if i == highlight else 3) * SS
        end = P(*e)
        cv2.line(img, origin, end, color, width, cv2.LINE_AA)
        if shots[i].runs == 6:  # over the rope: mark the landing
            cv2.circle(img, end, 5 * SS, color, -1, cv2.LINE_AA)
        ends[i] = end
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    taken: list[tuple[int, int, int, int]] = []
    for i in sorted(ends, key=lambda i: i != highlight):  # the highlighted shot is labelled first
        u, v = ends[i][0] // SS, ends[i][1] // SS
        tag = chip(f"{shots[i].label} ({shots[i].runs})", 13, (255, 255, 255), (15, 15, 15), i == highlight, (3, 1), 180, 3)
        th, tw = tag.shape[:2]
        # outwards from the batter first, then around the end of the line
        dx, dy = u - ends_origin[0], v - ends_origin[1]
        n = max(1.0, (dx * dx + dy * dy) ** 0.5)
        base = (int(u + 10 * dx / n - tw / 2), int(v + 10 * dy / n - th / 2))
        rows = [0] + [k * sign * (th + 2) for k in range(1, 6) for sign in (-1, 1)]
        spots = [base] + [(base[0] + ox, base[1] + oy) for oy in rows for ox in (0, tw + 6, -tw - 6)]
        for x, y in spots:
            box = (x, y, x + tw, y + th)
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in taken):
                break
        taken.append(box)
        paste(out, tag, x, y)
    paste(out, chip("direction from the shot type (AI read) - approximate", 12, (190, 190, 190), None, False, (0, 0)), 8, size - 18)
    return out
