"""End to end: `cricsense calibrate --points ...` on a synthetic delivery clip."""

import shutil

import numpy as np
import pytest

from cricsense.calibration import pitch_model
from cricsense.calibration.homography import CalibrationError
from cricsense.cli import main
from cricsense.config import load_config
from cricsense.io import read_doc, read_yaml_doc
from cricsense.passes.calibrate import parse_points, run_calibrate
from cricsense.schema import CalibrationDoc, CameraDoc
from tests.conftest import REPO_ROOT
from tests.synthetic import (
    camera_homography,
    camera_path,
    pitch_to_world,
    project,
    render_frame,
    render_world,
    true_image_to_pitch,
    write_video,
)

N_FRAMES = 30
CUTS = {12, 13}  # frames replaced by unrelated footage, like a cut to a replay
REF_FRAME = 3
CHECK = ("striker_bowling_left", "striker_bowling_right", "bowler_bowling_left", "bowler_bowling_right")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO_ROOT / "configs", root / "configs")
    shutil.rmtree(root / "configs" / "calibration")
    world, cut_world = render_world(), render_world(seed=42, creases=False)
    truths, frames = [], []
    for i, (pan, tilt, zoom) in enumerate(camera_path(N_FRAMES)):
        H = camera_homography(pan, tilt, zoom)
        truths.append(H)
        frames.append(render_frame(cut_world if i in CUTS else world, H))
    (root / "data").mkdir()
    video = write_video(root / "data" / "synthetic_delivery.avi", frames)
    return root, video, truths


def points_spec(truth_H, noise_px=0.7, seed=1):
    names = (*pitch_model.DEFAULT_FIT_LANDMARKS, *CHECK)
    pitch = np.array([pitch_model.LANDMARKS[n] for n in names])
    image = project(truth_H @ pitch_to_world(), pitch)
    image += np.random.default_rng(seed).normal(0, noise_px, image.shape)
    return ";".join(f"{n}={x:.2f},{y:.2f}" for n, (x, y) in zip(names, image))


def test_calibrate_end_to_end(clip, monkeypatch, capsys):
    root, video, truths = clip
    monkeypatch.setenv("CRICSENSE_ROOT", str(root))
    argv = ["calibrate", "--video", str(video), "--frame", str(REF_FRAME), "--points", points_spec(truths[REF_FRAME])]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "held-out check points (4)" in out and "frames calibrated" in out

    cal = read_yaml_doc(root / "configs" / "calibration" / "synthetic_delivery.yaml", CalibrationDoc)
    view = cal.view("main")
    assert view.frame_id == REF_FRAME and view.check_max_m < 0.15
    assert (root / "outputs" / "synthetic_delivery" / "calibration_check.png").is_file()

    camera = read_doc(root / "outputs" / "synthetic_delivery" / "camera.json", CameraDoc)
    assert len(camera.frames) == N_FRAMES
    # Score the tracker in pixels: metres depend on how foreshortened the view is.
    probes = np.array([(x, y) for x in np.linspace(0, 20.12, 6) for y in (-1.5, 1.5)])
    errors_px = []
    for f in camera.frames:
        if f.frame_id in CUTS:
            assert f.view is None, f"cut frame {f.frame_id} should not be calibrated"
            continue
        assert f.view == "main", f"frame {f.frame_id} lost tracking"
        pitch_to_image = np.linalg.inv(np.array(f.image_to_pitch))
        true_px = project(truths[f.frame_id] @ pitch_to_world(), probes)
        errors_px.append(np.linalg.norm(project(pitch_to_image, probes) - true_px, axis=1).max())
    # Includes 0.7 px of click noise on the reference frame and MJPG compression.
    assert max(errors_px) < 4.0 and np.median(errors_px) < 2.0, errors_px

    # Re-running for the same view needs --force; --propagate-only reuses the saved view.
    assert main(argv) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["calibrate", "--video", str(video), "--propagate-only"]) == 0


def test_points_parsing_errors():
    with pytest.raises(CalibrationError, match="unknown landmark"):
        parse_points("nowhere=1,2")
    with pytest.raises(CalibrationError, match="bad point"):
        parse_points("striker_popping_left=1")
    with pytest.raises(CalibrationError, match="twice"):
        parse_points("striker_popping_left=1,2;striker_popping_left=3,4")


def test_missing_corner_and_swapped_corners_fail_cleanly(clip, monkeypatch):
    root, video, truths = clip
    monkeypatch.setenv("CRICSENSE_ROOT", str(root))
    cfg = load_config()
    pts = parse_points(points_spec(truths[0], noise_px=0))
    three = {k: v for k, v in pts.items() if k != "bowler_popping_right"}
    with pytest.raises(CalibrationError, match="missing fit landmarks"):
        run_calibrate(cfg, video, points=three, view_name="x", propagate=False)
    swapped = dict(pts)
    swapped["bowler_popping_left"], swapped["bowler_popping_right"] = pts["bowler_popping_right"], pts["bowler_popping_left"]
    with pytest.raises(CalibrationError, match="swapped"):
        run_calibrate(cfg, video, points=swapped, view_name="y", propagate=False)
