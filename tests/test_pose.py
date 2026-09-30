"""Batter pose: footwork on the pitch, the swing, and refusing to guess when feet are hidden."""

import numpy as np

from cricsense.pose.batter import L_ANKLE, L_HIP, L_KNEE, L_WRIST, R_ANKLE, R_HIP, R_KNEE, R_WRIST, Observation, Thresholds, footwork, swing

H = np.diag([0.01, 0.01, 1.0])  # image px -> metres: x = u / 100
TH = Thresholds()


def body(front_u: float, back_u: float, wrist=(0.0, 0.0), ankle_v: float = 500.0) -> np.ndarray:
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    kp[R_ANKLE] = (front_u, ankle_v, 0.95)  # the front foot: larger u -> larger pitch x, towards the bowler
    kp[L_ANKLE] = (back_u, ankle_v, 0.95)
    kp[R_KNEE] = (front_u, ankle_v - 60, 0.9)
    kp[L_KNEE] = (back_u, ankle_v - 60, 0.9)
    kp[L_HIP] = kp[R_HIP] = ((front_u + back_u) / 2, ankle_v - 120, 0.9)
    kp[L_WRIST] = kp[R_WRIST] = ((front_u + back_u) / 2 + wrist[0], ankle_v - 150 + wrist[1], 0.9)
    return kp


def obs(frame, kp, bottom=918.0):
    return Observation(frame, kp, (0.0, kp[:, 1].min() - 20, 1.0, kp[:, 1].min() + 180), H, bottom)


def test_front_foot_stride_is_measured_on_the_pitch():
    seq = [obs(f, body(80, 50)) for f in range(0, 11)] + [obs(f, body(125, 55)) for f in range(18, 26)]
    assert footwork(seq, release=10, shot=20, th=TH) == ("front foot", 0.45)


def test_back_foot_and_in_the_crease():
    back = [obs(f, body(80, 50)) for f in range(0, 11)] + [obs(f, body(78, 20)) for f in range(18, 26)]
    assert footwork(back, 10, 20, TH) == ("back foot", -0.3)
    still = [obs(f, body(80, 50)) for f in range(0, 11)] + [obs(f, body(85, 52)) for f in range(18, 26)]
    assert footwork(still, 10, 20, TH)[0] == "in the crease"


def test_hidden_or_impossible_feet_give_no_answer():
    cut_off = [obs(f, body(80, 50)) for f in range(0, 11)] + [obs(f, body(125, 55, ankle_v=915)) for f in range(18, 26)]
    assert footwork(cut_off, 10, 20, TH) == (None, None)  # ankles under the score bar
    charge = [obs(f, body(80, 50)) for f in range(0, 11)] + [obs(f, body(300, 250)) for f in range(18, 26)]
    assert footwork(charge, 10, 20, TH) == (None, None)  # 2.2 m: a misread foot


def test_swing_played_or_left_whatever_the_camera_does():
    # the whole body drifts 20 px a frame (a tilting camera): only the hands relative to the hips count
    played = [obs(f, body(80, 50, wrist=(0, -40 * max(0, f - 12))) + [0, 20 * f, 0]) for f in range(8, 20)]
    left = [obs(f, body(80, 50) + [0, 20 * f, 0]) for f in range(8, 20)]
    assert swing(played, 10, 25.0, TH)[0] == "played"
    assert swing(left, 10, 25.0, TH)[0] == "left"
