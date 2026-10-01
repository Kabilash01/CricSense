"""Automatic calibration: finds the regulation creases on a synthetic broadcast frame."""

import cv2
import numpy as np

from cricsense.calibration import pitch_model as pm
from cricsense.calibration.auto import CORNER_NAMES, calibrate, find_views

# a real main-view homography (the hand-clicked reference of the dev clip)
TRUE = np.array([
    [-5.8157299257276686e-18, 0.07876973841728391, -32.494038924506526],
    [0.011095262794426611, 8.405502116989851e-05, -10.865456366548107],
    [-4.6620790228424265e-19, 0.0008914926133469184, 1.0],
])


def frame(occlude: bool = False, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((1080, 1920, 3), (60, 140, 70), np.uint8)  # grass
    P = np.linalg.inv(TRUE)

    def px(x, y):
        q = P @ [x, y, 1.0]
        return int(round(q[0] / q[2])), int(round(q[1] / q[2]))

    strip = np.array([px(-3, -1.525), px(-3, 1.525), px(23, 1.525), px(23, -1.525)], np.int32)
    cv2.fillPoly(img, [strip], (120, 170, 200))  # the pitch
    for (x0, y0), (x1, y1) in pm.crease_segments():
        if abs(abs(y0) - pm.PITCH_HALF_WIDTH) < 1e-6:
            continue  # pitch edges are not painted
        cv2.line(img, px(x0, y0), px(x1, y1), (245, 245, 245), 3, cv2.LINE_AA)
    u, v = px(0, 0)
    cv2.line(img, (u, v), (u, v - 60), (240, 240, 240), 6)  # the stumps
    if occlude:  # a player in white pads standing on the near-left return crease
        nu, nv = px(18.9, -1.32)
        cv2.rectangle(img, (nu - 25, nv - 160), (nu + 25, nv + 20), (40, 40, 200), -1)
        cv2.rectangle(img, (nu - 20, nv - 40), (nu + 20, nv + 15), (235, 235, 235), -1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def corner_errors(H: np.ndarray) -> np.ndarray:
    A, B = np.linalg.inv(H), np.linalg.inv(TRUE)
    errs = []
    for n in CORNER_NAMES:
        p = np.array([*pm.LANDMARKS[n], 1.0])
        a, b = A @ p, B @ p
        errs.append(np.hypot(*(a[:2] / a[2] - b[:2] / b[2])))
    return np.array(errs)


def test_finds_the_creases_on_a_clean_frame():
    r = calibrate(frame())
    assert r is not None and r.score > 0.8
    assert corner_errors(r.image_to_pitch).max() < 3.0


def test_a_player_on_a_crease_does_not_pull_it_off():
    r = calibrate(frame(occlude=True))
    assert r is not None
    assert np.median(corner_errors(r.image_to_pitch)) < 3.0


def test_no_pitch_no_calibration():
    grass = np.full((1080, 1920, 3), (60, 140, 70), np.uint8)
    assert calibrate(grass) is None


def test_views_need_two_agreeing_frames():
    one = find_views({0: frame(seed=1)})
    two = find_views({0: frame(seed=1), 50: frame(seed=2)})
    assert one == [] and len(two) == 1 and two[0][2] == 2


def test_a_camera_keeps_references_from_different_moments():
    views = find_views({0: frame(seed=1), 50: frame(seed=2), 400: frame(seed=3)})
    # one camera: its best frame, plus the frame far enough from it in time (not the one 50 frames on)
    assert len(views) == 2 and {f for f, _, _ in views} in ({0, 400}, {50, 400})
    assert all(n == 3 for _, _, n in views)
