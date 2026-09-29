"""3D camera from the ground calibration (places the ball and stump tops above the ground)."""

import numpy as np
import pytest

from cricsense.calibration import pitch_model as pm
from cricsense.calibration.camera_pose import CameraPose
from cricsense.calibration.homography import PitchHomography
from tests.synthetic import CAMERA_CENTRE, FRAME_SIZE, camera_homography, camera_projection, pitch_to_world, project

PROBES = np.array([(x, y, z) for x in (0.0, 10.0, 20.12) for y in (-1.5, 1.5) for z in (0.0, 0.711, 2.0)])


def truth(pan=0.0, tilt=0.0, zoom=1.0):
    P = camera_projection(pan, tilt, zoom)
    H = np.linalg.inv(P[:, [0, 1, 3]])
    p = np.c_[PROBES, np.ones(len(PROBES))] @ P.T
    return H / H[2, 2], p[:, :2] / p[:, 2:3]


@pytest.mark.parametrize("pan, tilt, zoom", [(0, 0, 1.0), (60, -40, 1.25), (-120, 30, 1.6)])
def test_exact_ground_calibration_gives_the_true_camera(pan, tilt, zoom):
    H, true_px = truth(pan, tilt, zoom)
    cam = CameraPose.from_homography(H, FRAME_SIZE)
    assert np.allclose(cam.centre, CAMERA_CENTRE, atol=0.05)
    assert np.abs(cam.project(PROBES) - true_px).max() < 0.05  # includes points 2 m above the ground


def test_click_noise_on_a_telephoto_view_stays_near_the_homography_error():
    # The closed form alone was 44-357 px off with 0.3-0.7 px of click noise; refinement fixes it.
    W2F = camera_homography(60, -40, 1.25) @ pitch_to_world()
    _, true_px = truth(60, -40, 1.25)
    pitch = np.array([pm.LANDMARKS[n] for n in pm.DEFAULT_FIT_LANDMARKS])
    worst = []
    for seed in range(10):
        clicks = project(W2F, pitch) + np.random.default_rng(seed).normal(0, 0.5, (4, 2))
        cam = CameraPose.from_homography(PitchHomography.fit(clicks, pitch).H, FRAME_SIZE)
        worst.append(np.abs(cam.project(PROBES) - true_px).max())
    assert np.median(worst) < 5.0


def test_rays_pass_through_the_point():
    H, _ = truth(30, 10, 1.2)
    cam = CameraPose.from_homography(H, FRAME_SIZE)
    point = np.array([12.0, 0.5, 1.8])  # a ball in flight
    origin, d = cam.ray(tuple(cam.project(point[None])[0]))
    closest = origin + d * ((point - origin) @ d)
    assert np.linalg.norm(closest - point) < 1e-3
