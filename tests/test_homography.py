import numpy as np
import pytest

from cricsense.calibration import pitch_model
from cricsense.calibration.homography import CalibrationError, PitchHomography, reprojection_errors
from tests.synthetic import camera_homography, pitch_to_world, project, true_image_to_pitch

FIT = pitch_model.DEFAULT_FIT_LANDMARKS
CHECK = ("striker_bowling_left", "striker_bowling_right", "bowler_bowling_left", "bowler_bowling_right")


def truth() -> np.ndarray:
    return true_image_to_pitch(camera_homography())


def clicks(names, H_true=None, noise_px=0.0, seed=0):
    H_true = truth() if H_true is None else H_true
    pitch = np.array([pitch_model.LANDMARKS[n] for n in names])
    image = project(np.linalg.inv(H_true), pitch)
    return image + np.random.default_rng(seed).normal(0, noise_px, image.shape), pitch


def probe_points() -> np.ndarray:
    return np.array([(x, y) for x in np.linspace(0, 20.12, 11) for y in (-1.5, 0.0, 1.5)])


def test_exact_corners_recover_the_true_mapping():
    image, pitch = clicks(FIT)
    h = PitchHomography.fit(image, pitch)
    probes = probe_points()
    probe_px = project(np.linalg.inv(truth()), probes)
    assert np.abs(h.image_to_pitch(probe_px) - probes).max() < 1e-4
    assert np.abs(h.pitch_to_image(probes) - probe_px).max() < 1e-3


def test_one_pixel_click_noise_stays_within_centimetres_on_the_pitch():
    image, pitch = clicks(FIT, noise_px=1.0)
    h = PitchHomography.fit(image, pitch)
    check_img, check_pitch = clicks(CHECK)
    _, err_m = reprojection_errors(h, check_img, check_pitch)
    assert err_m.max() < 0.15  # 1 px of noise in a 960x540 frame: well under a ball's width of length


def test_four_points_always_fit_exactly_so_fit_error_says_nothing():
    image, pitch = clicks(FIT, noise_px=3.0)
    h = PitchHomography.fit(image, pitch)
    err_px, _ = reprojection_errors(h, image, pitch)
    assert err_px.max() < 1e-3  # hence the held-out check points


def test_swapped_labels_are_rejected():
    image, pitch = clicks(FIT)
    image[[2, 3]] = image[[3, 2]]  # bowler-end left/right swapped
    with pytest.raises(CalibrationError, match="swapped"):
        PitchHomography.fit(image, pitch)


def test_degenerate_points_are_rejected():
    line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    with pytest.raises(CalibrationError, match="collinear"):
        PitchHomography.fit(line, np.array([pitch_model.LANDMARKS[n] for n in FIT]))
    image, pitch = clicks(FIT)
    with pytest.raises(CalibrationError, match="at least 4"):
        PitchHomography.fit(image[:3], pitch[:3])


def test_extra_points_on_a_shared_line_are_fine():
    names = (*FIT, "striker_bowling_left", "striker_middle_stump", "striker_bowling_right")
    image, pitch = clicks(names, noise_px=0.5)
    PitchHomography.fit(image, pitch)  # three of these lie on x = 0, which is legitimate


def test_sky_maps_to_nan():
    h = PitchHomography.fit(*clicks(FIT))
    horizon_y = project(np.linalg.inv(truth()), np.array([[-1e7, 0.0]]))[0, 1]  # vanishing point
    above = h.image_to_pitch(np.array([[480.0, horizon_y - 50]]))
    below = h.image_to_pitch(np.array([[480.0, 300.0]]))
    assert np.all(np.isnan(above)) and np.all(np.isfinite(below))


def test_for_frame_composes_camera_motion():
    ref = PitchHomography(truth())
    moved = camera_homography(pan_px=40, tilt_px=-25, zoom=1.2)
    frame_to_ref = camera_homography() @ np.linalg.inv(moved)  # frame pixels -> reference pixels
    composed = ref.for_frame(frame_to_ref)
    probes = probe_points()
    probe_px = project(moved @ pitch_to_world(), probes)
    assert np.abs(composed.image_to_pitch(probe_px) - probes).max() < 1e-6


def test_invalid_matrix_rejected():
    with pytest.raises(CalibrationError):
        PitchHomography(np.zeros((3, 3)))
