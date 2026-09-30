"""Wagon wheel: shot angles, mirrored for left-handers, lengths from runs; batting hand from the stance."""

import numpy as np

from cricsense.pose.batter import L_ANKLE, L_KNEE, R_ANKLE, R_KNEE, Observation, batting_hand
from cricsense.render.wagon_wheel import BOUNDARY_M, Shot, draw_wagon_wheel, endpoint


def test_off_side_flips_with_the_batting_hand():
    rh = endpoint(Shot("1.1", "cover drive", 4, "right"))
    lh = endpoint(Shot("1.2", "cover drive", 4, "left"))
    assert rh[0] > 0 and lh[0] > 0  # forward of square, towards the bowler's end
    assert rh[1] < 0 < lh[1]  # from behind the bowler a right-hander's off side is on the left
    assert abs(np.hypot(*rh) - BOUNDARY_M) < 1e-6  # a four reaches the rope


def test_lengths_follow_the_runs_and_unknown_shots_are_not_drawn():
    one, six = endpoint(Shot("a", "pull", 1, "right")), endpoint(Shot("b", "pull", 6, "right"))
    assert np.hypot(*one) < np.hypot(*six) and np.hypot(*six) > BOUNDARY_M
    assert endpoint(Shot("c", "mystery shot", 4, "right")) is None
    assert endpoint(Shot("d", "pull", 0, "right")) is None


def test_behind_square_goes_back_past_the_keeper():
    assert endpoint(Shot("e", "glance", 1, "right"))[0] < 0


def test_draw_returns_a_square_bgra_image():
    img = draw_wagon_wheel([Shot("1.1", "pull", 4, "right"), Shot("1.2", "cut", 1, "left")], 300, highlight=0)
    assert img.shape == (300, 300, 4)


def _stance(y_m: float) -> Observation:
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    for a, k, du in ((L_ANKLE, L_KNEE, -10), (R_ANKLE, R_KNEE, 10)):
        kp[a] = (100 * y_m + du + 500, 500, 0.95)
        kp[k] = (100 * y_m + du + 500, 440, 0.9)
    H = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, -5.0], [0.0, 0.0, 1.0]])  # pitch y = (u - 500) / 100
    return Observation(0, kp, (400.0, 300.0, 600.0, 520.0), H, 918.0)


def test_batting_hand_from_where_the_feet_stand():
    assert batting_hand([_stance(0.3)], release=0) == "right"
    assert batting_hand([_stance(-0.25)], release=0) == "left"
    assert batting_hand([_stance(0.02)], release=0) is None  # on middle stump: do not guess
