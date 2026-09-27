import pytest
from pydantic import ValidationError

from cricsense.io import read_doc, write_doc
from cricsense.schema import (
    BallDoc,
    BallFrame,
    Event,
    EventsDoc,
    EventType,
    PlayerFrame,
    PlayersDoc,
    Role,
    RunInfo,
    VideoMeta,
)

VIDEO = VideoMeta(video_id="clip", fps=25.0, n_frames=100, width=1920, height=1080)


def player(frame_id=10, **kw):
    fields = dict(frame_id=frame_id, t=frame_id / 25, sid=1, box=(10, 20, 50, 120), role=Role.STRIKER)
    return PlayerFrame(**{**fields, **kw})


def test_round_trip_all_docs(tmp_path):
    players = PlayersDoc(
        video=VIDEO,
        run=RunInfo(wall_time_s=1.5, frames_processed=100, device="cpu"),
        frames=[player(), player(frame_id=11, t=11 / 25, pitch_xy=(1.2, -0.3), identity="Kohli", identity_conf=0.9)],
    )
    ball = BallDoc(
        video=VIDEO,
        frames=[
            BallFrame(frame_id=12, t=12 / 25, uv=(640.5, 400.0), observed=True),
            BallFrame(frame_id=13, t=13 / 25, uv=(641.0, 420.0), observed=False, pitch_xy=(6.1, 0.2)),
        ],
    )
    events = EventsDoc(
        video=VIDEO,
        events=[
            Event(
                frame_id=13, t=13 / 25, type=EventType.BOUNCE, actors={Role.BOWLER: 3},
                payload={"length_m": 6.1, "length_class": "good"}, confidence=0.8, source="ball.bounce",
            )
        ],
    )
    for doc, name in ((players, "players.json"), (ball, "ball.json"), (events, "events.json")):
        path = write_doc(doc, tmp_path / "clip" / name)
        assert read_doc(path, type(doc)) == doc


def test_unknown_fields_rejected():
    # No pass can smuggle in a field such as a km/h speed.
    with pytest.raises(ValidationError, match="speed_kmph"):
        BallFrame(frame_id=1, t=0.04, uv=(1, 1), observed=True, speed_kmph=409.9)


def test_time_must_match_source_frame_index():
    # This is the legacy skip bug: frame 20 read with skip=2 was logged as frame 10.
    with pytest.raises(ValidationError, match="t must come from the source frame index"):
        PlayersDoc(video=VIDEO, frames=[player(frame_id=10, t=20 / 25)])


def test_frame_beyond_video_rejected():
    with pytest.raises(ValidationError, match="n_frames"):
        PlayersDoc(video=VIDEO, frames=[player(frame_id=100, t=4.0)])


@pytest.mark.parametrize(
    "bad",
    [
        {"box": (50, 20, 10, 120)},  # x2 < x1
        {"sid": 0},
        {"identity_conf": 1.5},
        {"role": "Batsman"},  # a YOLO class name is not a role
    ],
)
def test_player_frame_constraints(bad):
    with pytest.raises(ValidationError):
        player(**bad)


def test_event_constraints():
    ok = dict(frame_id=1, t=0.04, type="CONTACT", confidence=0.5, source="fuse.contact")
    assert Event(**ok).type is EventType.CONTACT
    for bad in ({"confidence": -0.1}, {"source": ""}, {"type": "SIX"}, {"actors": {"batsman": 1}}):
        with pytest.raises(ValidationError):
            Event(**{**ok, **bad})


def test_read_doc_missing_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the pass that produces it"):
        read_doc(tmp_path / "ball.json", BallDoc)
