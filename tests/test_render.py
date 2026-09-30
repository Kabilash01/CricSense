"""Overlay and pitch map: geometry, what is shown when, and that drawing never breaks a frame."""

import numpy as np

from cricsense.render.draw import blend, chip
from cricsense.render.overlay import Scene, compose
from cricsense.render.pitch_map import LENGTH_COLORS, MapPoint, MapView, draw_pitch_map, summary_png
from cricsense.schema import BallDoc, BallFrame, Delivery, Event, EventsDoc, EventType, LengthClass, VideoMeta

VIDEO = VideoMeta(video_id="clip", fps=25.0, n_frames=1000, width=640, height=360)


def test_pitch_map_orientation_matches_the_broadcast():
    view = MapView(px_per_m=10, x_range=(-1.0, 13.0), margin_m=1.0)
    stumps = view.to_px(0.0, 0.0)
    full = view.to_px(4.0, 0.0)
    right = view.to_px(4.0, 0.5)
    assert stumps[1] < full[1]  # the striker's stumps at the top
    assert right[0] > full[0]  # +line to the right, as on screen


def test_pitch_map_marks_the_bounce_in_its_length_colour():
    view = MapView(px_per_m=30)
    img = draw_pitch_map([MapPoint("1.1", 7.0, 0.0, "good", True)], view, labels=False)
    assert img.shape == (view.size[1], view.size[0], 4)
    u, v = view.to_px(7.0, 0.0)
    assert np.abs(img[v, u, :3].astype(int) - LENGTH_COLORS["good"]).max() < 30


def test_blend_clips_at_the_frame_edge_and_respects_alpha():
    frame = np.zeros((50, 80, 3), np.uint8)
    patch = np.zeros((20, 20, 4), np.uint8)
    patch[..., :3] = 200
    patch[..., 3] = 255
    blend(frame, patch, 70, 40)  # mostly outside: must not raise
    assert frame[45, 75].tolist() == [200, 200, 200]
    patch[..., 3] = 0
    blend(frame, patch, 0, 0)
    assert frame[5, 5].tolist() == [0, 0, 0]
    assert chip("SIX", 20).shape[2] == 4


def _scene() -> Scene:
    frames = [BallFrame(frame_id=f, t=f / 25, uv=(300 + f % 10, 100 + f % 50), observed=True, delivery_id=0) for f in range(100, 112)]
    deliveries = [
        Delivery(
            delivery_id=0,
            shot_frames=(90, 140),
            release_frame=100,
            release_t=4.0,
            release_point=(18.6, 0.0, 2.0),
            detections=12,
            bounce_t=110 / 25,
            bounce_xy=(4.0, 0.1),
            bounce_confirmed=True,
            length_m=4.0,
            line_m=0.1,
            length_class=LengthClass.FULL,
            radar_speed_kph=140.0,
        ),
        Delivery(delivery_id=1, shot_frames=(600, 640), release_frame=600, release_t=24.0, release_point=(18.6, 0.0, 2.0), detections=6),
    ]
    ball = BallDoc(video=VIDEO, frames=frames, deliveries=deliveries)
    ev = [
        Event(frame_id=100, t=4.0, type=EventType.RELEASE, payload={"delivery_id": 0, "bowler": "NORTJE", "striker": "KOHLI", "over": "12.1", "radar_speed_kph": 140.0}, confidence=0.8, source="t"),
        Event(frame_id=300, t=12.0, type=EventType.RUNS, payload={"delivery_id": 0, "runs": 6, "off_bat": 6, "extras": 0, "kind": "legal", "striker": "KOHLI"}, confidence=0.9, source="t"),
        Event(frame_id=300, t=12.0, type=EventType.BOUNDARY, payload={"delivery_id": 0, "kind": "SIX"}, confidence=0.9, source="t"),
        Event(frame_id=600, t=24.0, type=EventType.RELEASE, payload={"delivery_id": 1}, confidence=0.8, source="t"),
    ]
    return Scene(ball, EventsDoc(video=VIDEO, events=ev), None, None, (VIDEO.width, VIDEO.height))


def test_card_lasts_until_after_the_outcome_but_not_into_the_next_ball():
    s = _scene()
    first = s.deliveries[0]
    assert first.title.startswith("12.1") and first.outcome == "SIX" and first.speed_source == "radar"
    assert s.active(105) is first
    assert s.active(300 + 50) is first  # held after the outcome
    assert s.active(300 + 150) is None  # CARD_HOLD_S is over
    assert s.active(600).did == 1
    assert [b[1] for b in s.banners] == ["SIX  KOHLI"]


def test_compose_draws_without_changing_the_frame_size():
    s = _scene()
    for f in (0, 105, 111, 300, 600):
        frame = np.full((VIDEO.height, VIDEO.width, 3), 90, np.uint8)
        out = compose(frame, f, s)
        assert out.shape == (VIDEO.height, VIDEO.width, 3)
    assert (compose(np.full((360, 640, 3), 90, np.uint8), 300, s) != 90).any()  # the card and banner are drawn


def test_summary_png_lists_every_delivery():
    rows = [("12.1", "Nortje → Kohli", "full 4.0 m", "10 cm R", "140 km/h radar", "SIX")]
    img = summary_png([MapPoint("12.1", 4.0, 0.1, "full", True)], rows, "test")
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[1] > img.shape[0]
