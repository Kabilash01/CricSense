"""Bat on ball: a bat box touching a ball detection near the striker, just after the bounce."""

from __future__ import annotations

from dataclasses import dataclass

CONTACT_WINDOW = (-3, 14)  # frames around the bounce to search (the bat meets the ball ~0.1-0.4 s later)
MARGIN_PX = 12  # a fast ball and a swinging bat are both blurred
MIN_BAT_CONF = 0.4
MIN_BALL_CONF = 0.15


@dataclass(frozen=True)
class Contact:
    frame_id: int
    bat_conf: float
    ball_conf: float


def find_contact(
    bats: dict[int, list[dict]],
    balls: dict[int, list[dict]],
    around_frame: int,
    near_box: tuple[float, float, float, float] | None,
    frame_size: tuple[int, int],
    overlays: tuple[tuple[float, float, float, float], ...],
) -> Contact | None:
    """First frame in the window where a ball candidate lies inside a bat box.

    near_box: the striker's box (if known) - bats elsewhere (the non-striker's, the score
    bar's bat icon) do not count. Returns None when no such frame exists.
    """
    w, h = frame_size
    for f in range(around_frame + CONTACT_WINDOW[0], around_frame + CONTACT_WINDOW[1] + 1):
        for bat in (d for d in bats.get(f, []) if d["cls"] == 1 and d["conf"] >= MIN_BAT_CONF):
            x1, y1, x2, y2 = bat["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(ox0 * w <= cx <= ox1 * w and oy0 * h <= cy <= oy1 * h for ox0, oy0, ox1, oy1 in overlays):
                continue  # an on-screen graphic, e.g. the striker's bat icon
            if near_box is not None and not _overlaps((x1, y1, x2, y2), near_box, pad=40):
                continue
            for ball in (d for d in balls.get(f, []) if d["cls"] == 0 and d["conf"] >= MIN_BALL_CONF):
                bx, by = (ball["box"][0] + ball["box"][2]) / 2, (ball["box"][1] + ball["box"][3]) / 2
                if x1 - MARGIN_PX <= bx <= x2 + MARGIN_PX and y1 - MARGIN_PX <= by <= y2 + MARGIN_PX:
                    return Contact(f, bat["conf"], ball["conf"])
    return None


def _overlaps(a, b, pad: float) -> bool:
    return not (a[2] < b[0] - pad or a[0] > b[2] + pad or a[3] < b[1] - pad or a[1] > b[3] + pad)
