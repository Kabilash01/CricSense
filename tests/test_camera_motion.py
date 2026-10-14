import dataclasses

import numpy as np
import pytest

from cricsense.calibration.camera_motion import CameraTracker
from cricsense.calibration.homography import PitchHomography
from cricsense.config import CameraOptions
from tests.synthetic import (
    FRAME_SIZE,
    camera_homography,
    camera_path,
    pitch_to_world,
    project,
    render_frame,
    render_world,
    true_image_to_pitch,
)

OPTS = CameraOptions(
    match_scale=1.0,
    orb_features=2000,
    ransac_px=2.0,
    min_inliers=40,
    max_chain_frames=25,
    smooth_window=11,
    outlier_px=6.0,
    overlay_mask=((0.0, 0.85, 1.0, 1.0),),
)
PROBES = np.array([(x, y) for x in np.linspace(0, 20.12, 6) for y in (-1.5, 0.0, 1.5)])


@pytest.fixture(scope="module")
def world():
    return render_world()


@pytest.fixture(scope="module")
def reference(world):
    H0 = camera_homography()
    return render_frame(world, H0), PitchHomography(true_image_to_pitch(H0))


def pixel_error(estimate, world_to_frame) -> float:
    """Worst error in pixels over the probes the frame actually shows (0 if it shows none).

    Pixels measure the tracker itself, independent of foreshortening; probes outside the
    frame would only measure extrapolation.
    """
    true_px = project(world_to_frame @ pitch_to_world(), PROBES)
    w, h = FRAME_SIZE
    visible = (true_px[:, 0] >= 0) & (true_px[:, 0] < w) & (true_px[:, 1] >= 0) & (true_px[:, 1] < h)
    if not visible.any():
        return 0.0
    est_px = estimate.homography.pitch_to_image(PROBES[visible])
    return float(np.linalg.norm(est_px - true_px[visible], axis=1).max())


def test_tracks_a_delivery_pan_and_zoom(world, reference):
    tracker = CameraTracker({"main": reference}, OPTS)
    errors = []
    for pan, tilt, zoom in camera_path(15):
        H = camera_homography(pan, tilt, zoom)
        est = tracker.estimate(render_frame(world, H))
        assert est.view == "main" and est.inliers >= OPTS.min_inliers
        errors.append(pixel_error(est, H))
    assert max(errors) < 5.0 and np.median(errors) < 1.0, errors


def test_cut_to_unrelated_footage_is_not_calibrated(world, reference):
    tracker = CameraTracker({"main": reference}, OPTS)
    other = render_frame(render_world(seed=99, creases=False), camera_homography(), score_bar=True)
    est = tracker.estimate(other)
    assert est.view is None and est.homography is None
    # ...and tracking resumes when the main camera returns.
    assert tracker.estimate(render_frame(world, camera_homography(20, 0, 1.1))).view == "main"


def test_static_score_bar_does_not_pin_the_estimate(world, reference):
    # With the bar masked, a zoomed frame is not mistaken for "no camera motion".
    tracker = CameraTracker({"main": reference}, OPTS)
    H = camera_homography(50, -30, 1.3)
    assert pixel_error(tracker.estimate(render_frame(world, H)), H) < 5.0


def test_unmasked_graphic_alone_cannot_produce_a_match(world, reference):
    # Real-footage failure: an unmasked logo matched itself at zero displacement in every
    # close-up. Its inliers are one tight cluster, which the spread check rejects.
    ref_img, ref_hom = reference
    logo = np.random.default_rng(7).integers(0, 255, (60, 120, 3), dtype=np.uint8)
    ref_with_logo = ref_img.copy()
    ref_with_logo[20:80, 20:140] = logo
    close_up = np.full_like(ref_img, 90)  # featureless except for the logo
    close_up[20:80, 20:140] = logo
    tracker = CameraTracker({"main": (ref_with_logo, ref_hom)}, OPTS)
    assert tracker.estimate(close_up).view is None


PAN_AWAY = np.arange(0, 901, 25)  # the camera follows a shot until little of the reference view is left


def test_zoom_up_to_2x_tracks_directly(world, reference):
    tracker = CameraTracker({"main": reference}, OPTS)
    for zoom in 1.015 ** np.arange(0, 48, 4):  # 1.5% per frame up to ~2x; real deliveries peaked near 1%
        H = camera_homography(0, 0, zoom)
        est = tracker.estimate(render_frame(world, H))
        assert est.view == "main" and not est.chained, f"zoom {zoom:.2f}"
        assert pixel_error(est, H) < 5.0


def test_panning_away_from_the_reference_is_bridged_by_chaining(world, reference):
    tracker = CameraTracker({"main": reference}, OPTS)
    results = []
    for pan in PAN_AWAY:
        H = camera_homography(-pan, 0, 1.0)
        est = tracker.estimate(render_frame(world, H))
        assert est.view == "main", f"lost at pan {pan}px"
        results.append((est.chained, pixel_error(est, H)))
    assert any(chained for chained, _ in results), "expected direct matching to give out and chaining to take over"
    assert max(err for _, err in results) < 5.0


def test_chaining_is_capped(world, reference):
    no_chain = CameraTracker({"main": reference}, dataclasses.replace(OPTS, max_chain_frames=0))
    lost = [no_chain.estimate(render_frame(world, camera_homography(-pan, 0, 1.0))).view is None for pan in PAN_AWAY]
    assert any(lost), "without chaining, the far end of the pan should be untrackable"


def test_picks_the_matching_view_among_several(world, reference):
    other_world = render_world(seed=5)
    H0 = camera_homography()
    other_ref = (render_frame(other_world, H0), PitchHomography(true_image_to_pitch(H0)))
    tracker = CameraTracker({"main": reference, "other_end": other_ref}, OPTS)
    assert tracker.estimate(render_frame(other_world, camera_homography(10, 5, 1.05))).view == "other_end"
    assert tracker.estimate(render_frame(world, camera_homography(10, 5, 1.05))).view == "main"


def test_smoothing_removes_jitter_and_outliers():
    from cricsense.calibration.camera_motion import smooth_homographies

    rng = np.random.default_rng(3)
    truths, noisy = [], []
    for k, (pan, tilt, zoom) in enumerate(camera_path(40)):
        H = camera_homography(pan, tilt, zoom)
        truths.append(H)
        jitter = np.array([[1, 0, rng.normal(0, 3)], [0, 1, rng.normal(0, 3)], [0, 0, 1]])  # ~3 px per frame
        if k in (12, 27):
            jitter[0, 2] += 60  # two grossly wrong estimates
        noisy.append(true_image_to_pitch(jitter @ H))
    frames = [(k, H) for k, H in enumerate(noisy)] + [(40, None)]
    smoothed, outliers = smooth_homographies(frames, window=11, outlier_px=6.0)

    def err(Hs):
        """Worst pixel distance between where probes land under the estimate vs the truth."""
        worst = 0.0
        for H, truth in zip(Hs, truths):
            est_px = project(np.linalg.inv(H), PROBES)
            true_px = project(truth @ pitch_to_world(), PROBES)
            worst = max(worst, float(np.linalg.norm(est_px - true_px, axis=1).max()))
        return worst

    assert outliers >= 2 and smoothed[-1] is None
    assert err(smoothed[:40]) < err(noisy) / 3


def test_within_a_shot_only_the_last_reference_is_matched(world, reference):
    """Several references of one scene: a smooth move is followed on the last frame's reference,
    one feature match per frame instead of one per reference (the 6x speed-up on Bumrah's over)."""
    H1 = camera_homography(4, 2, 1.1)
    second = (render_frame(world, H1), PitchHomography(true_image_to_pitch(H1)))
    tracker = CameraTracker({"main": reference, "zoomed": second}, OPTS)
    calls = 0
    match = tracker._match

    def counting(*args):
        nonlocal calls
        calls += 1
        return match(*args)

    tracker._match = counting
    path = camera_path(25)  # ~1 s of a delivery's pan and zoom
    for pan, tilt, zoom in path:
        assert tracker.estimate(render_frame(world, camera_homography(pan, tilt, zoom))).view is not None
    assert calls <= 1.2 * len(path)  # searching every reference: 2 per frame
