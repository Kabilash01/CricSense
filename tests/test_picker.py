import numpy as np
import pytest

from cricsense.calibration import pitch_model
from cricsense.calibration.picker import PickerSession, draw_pitch, draw_session
from tests.synthetic import FRAME_SIZE, camera_homography, project, render_frame, render_world, true_image_to_pitch

H_TRUE = true_image_to_pitch(camera_homography())


def image_of(name: str) -> tuple[float, float]:
    x, y = project(np.linalg.inv(H_TRUE), np.array([pitch_model.LANDMARKS[name]]))[0]
    return float(x), float(y)


def click_all(session: PickerSession, names) -> None:
    for n in names:
        assert session.current() == n
        session.click(*image_of(n))


def test_clicks_follow_the_landmark_order_and_fit():
    s = PickerSession()
    assert not s.fit_complete
    click_all(s, pitch_model.DEFAULT_FIT_LANDMARKS)
    assert s.fit_complete and s.current() == "striker_bowling_left"
    probe = np.array([[10.0, 0.5]])
    assert np.allclose(s.fit().pitch_to_image(probe), project(np.linalg.inv(H_TRUE), probe), atol=1e-6)


def test_fit_points_cannot_be_skipped_but_check_points_can():
    s = PickerSession()
    assert not s.skip()
    click_all(s, pitch_model.DEFAULT_FIT_LANDMARKS)
    assert s.skip() and s.current() == "striker_bowling_right"
    s.click(*image_of("striker_bowling_right"))
    assert set(s.check_points()) == {"striker_bowling_right"}


def test_undo_walks_back_through_clicks_and_skips():
    s = PickerSession()
    click_all(s, pitch_model.DEFAULT_FIT_LANDMARKS)
    s.skip()
    s.undo()  # un-skip
    assert s.current() == "striker_bowling_left"
    s.undo()  # un-click the 4th corner
    assert s.current() == "bowler_popping_right" and not s.fit_complete
    s.reset()
    assert s.current() == "striker_popping_left" and not s.points


def test_extra_clicks_after_the_last_landmark_are_ignored():
    s = PickerSession(check_names=())
    click_all(s, pitch_model.DEFAULT_FIT_LANDMARKS)
    s.click(1, 1)
    assert len(s.points) == 4 and s.current() is None


def test_bad_configuration_rejected():
    with pytest.raises(ValueError, match="unknown"):
        PickerSession(fit_names=("nope", "a", "b", "c"))
    with pytest.raises(ValueError, match="at least 4"):
        PickerSession(fit_names=pitch_model.DEFAULT_FIT_LANDMARKS[:3])


def test_drawing_marks_the_creases():
    frame = render_frame(render_world(), camera_homography(), score_bar=False)
    s = PickerSession()
    click_all(s, pitch_model.DEFAULT_FIT_LANDMARKS)
    drawn = draw_session(frame, s, s.fit())
    assert drawn.shape == frame.shape and (drawn != frame).any()
    # The projected striker popping crease passes through the clicked corner.
    corner = np.round(image_of("striker_popping_left")).astype(int)
    lines_only = draw_pitch(np.zeros_like(frame), s.fit())
    x, y = corner
    assert lines_only[y - 2 : y + 3, x - 2 : x + 3].any()
    assert frame.shape[:2] == FRAME_SIZE[::-1]
