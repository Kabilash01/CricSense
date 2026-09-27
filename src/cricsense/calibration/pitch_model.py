"""Pitch geometry in metres (ICC Law 6 and Law 7 dimensions).

Pitch frame used everywhere in the package:
    origin  base of the middle stump at the STRIKER's end (the far end in the main
            broadcast view, which always shoots from behind the bowler)
    x       along the pitch toward the bowler's stumps, 0 .. 20.12
    y       across the pitch, positive toward the right of the broadcast image

Length is therefore simply x: a ball pitching at x = 2 m landed 2 m from the batter's stumps.
Whether +y is off side or leg side depends on the batter's handedness, which this frame
deliberately does not encode.
"""

from __future__ import annotations

STUMPS_TO_STUMPS = 20.12  # between the two sets of stumps
POPPING_CREASE = 1.22  # popping crease in front of each bowling crease
RETURN_CREASE_Y = 1.32  # return creases either side of the middle-stump line
RETURN_CREASE_BACK = 2.44  # return creases extend at least this far behind the popping crease
POPPING_CREASE_HALF = 1.83  # popping crease extends at least this far either side
PITCH_HALF_WIDTH = 1.525  # the 3.05 m wide pitch
STUMP_HALF_SPAN = 0.1143  # outer stumps, 22.86 cm apart overall

STRIKER_STUMPS_X = 0.0
BOWLER_STUMPS_X = STUMPS_TO_STUMPS
STRIKER_POPPING_X = STRIKER_STUMPS_X + POPPING_CREASE
BOWLER_POPPING_X = BOWLER_STUMPS_X - POPPING_CREASE

# Named ground points a user can click. "left"/"right" mean as seen in the broadcast image.
LANDMARKS: dict[str, tuple[float, float]] = {
    # The four crease corners: the default calibration set (a 17.68 x 2.64 m rectangle).
    "striker_popping_left": (STRIKER_POPPING_X, -RETURN_CREASE_Y),
    "striker_popping_right": (STRIKER_POPPING_X, RETURN_CREASE_Y),
    "bowler_popping_left": (BOWLER_POPPING_X, -RETURN_CREASE_Y),
    "bowler_popping_right": (BOWLER_POPPING_X, RETURN_CREASE_Y),
    # Where each bowling crease meets the return creases: useful held-out check points.
    "striker_bowling_left": (STRIKER_STUMPS_X, -RETURN_CREASE_Y),
    "striker_bowling_right": (STRIKER_STUMPS_X, RETURN_CREASE_Y),
    "bowler_bowling_left": (BOWLER_STUMPS_X, -RETURN_CREASE_Y),
    "bowler_bowling_right": (BOWLER_STUMPS_X, RETURN_CREASE_Y),
    # Middle stump bases, often hidden by the batter, keeper or umpire.
    "striker_middle_stump": (STRIKER_STUMPS_X, 0.0),
    "bowler_middle_stump": (BOWLER_STUMPS_X, 0.0),
}

DEFAULT_FIT_LANDMARKS = (
    "striker_popping_left",
    "striker_popping_right",
    "bowler_popping_left",
    "bowler_popping_right",
)

Segment = tuple[tuple[float, float], tuple[float, float]]


def crease_segments() -> list[Segment]:
    """Painted lines and the pitch outline, for drawing a projected pitch over the image."""
    segs: list[Segment] = []
    for stumps_x, toward_middle in ((STRIKER_STUMPS_X, 1.0), (BOWLER_STUMPS_X, -1.0)):
        popping_x = stumps_x + toward_middle * POPPING_CREASE
        back_x = popping_x - toward_middle * RETURN_CREASE_BACK
        segs.append(((stumps_x, -RETURN_CREASE_Y), (stumps_x, RETURN_CREASE_Y)))  # bowling crease
        segs.append(((popping_x, -POPPING_CREASE_HALF), (popping_x, POPPING_CREASE_HALF)))  # popping crease
        for y in (-RETURN_CREASE_Y, RETURN_CREASE_Y):
            segs.append(((popping_x, y), (back_x, y)))  # return creases
    for y in (-PITCH_HALF_WIDTH, PITCH_HALF_WIDTH):
        segs.append(((STRIKER_STUMPS_X, y), (BOWLER_STUMPS_X, y)))  # pitch edges
    return segs


def stump_points() -> list[tuple[float, float]]:
    return [
        (x, y)
        for x in (STRIKER_STUMPS_X, BOWLER_STUMPS_X)
        for y in (-STUMP_HALF_SPAN, 0.0, STUMP_HALF_SPAN)
    ]
