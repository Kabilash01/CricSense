import pytest

from cricsense.io import VideoReader
from tests.conftest import CLIP_FPS, CLIP_FRAMES, CLIP_SIZE, frame_level


def _levels_match(frames) -> bool:
    return all(abs(float(f.image.mean()) - frame_level(f.frame_id)) < 3 for f in frames)


def test_reports_source_properties(synthetic_clip):
    with VideoReader(synthetic_clip) as r:
        assert r.fps == pytest.approx(CLIP_FPS)
        assert r.n_frames == CLIP_FRAMES
        assert (r.width, r.height) == CLIP_SIZE
        meta = r.meta("synthetic_clip")
    assert meta.fps == pytest.approx(CLIP_FPS) and meta.n_frames == CLIP_FRAMES


def test_reads_every_frame_in_order(synthetic_clip):
    with VideoReader(synthetic_clip) as r:
        frames = list(r)
    assert [f.frame_id for f in frames] == list(range(CLIP_FRAMES))
    assert _levels_match(frames)


def test_skip_keeps_source_frame_ids_and_times(synthetic_clip):
    # The legacy reader numbered returned frames 1, 2, 3... so with skip=2 every id was halved.
    with VideoReader(synthetic_clip, skip=2) as r:
        frames = list(r)
    assert [f.frame_id for f in frames] == list(range(0, CLIP_FRAMES, 2))
    assert all(f.t == pytest.approx(f.frame_id / CLIP_FPS) for f in frames)
    assert _levels_match(frames)


def test_start_end_window(synthetic_clip):
    with VideoReader(synthetic_clip, start=10, end=20, skip=3) as r:
        frames = list(r)
    assert [f.frame_id for f in frames] == [10, 13, 16, 19]
    assert _levels_match(frames)


def test_resize(synthetic_clip):
    with VideoReader(synthetic_clip, resize=(80, 60)) as r:
        first = next(iter(r))
        assert (r.width, r.height) == (80, 60)
    assert first.image.shape == (60, 80, 3)


def test_bad_inputs(tmp_path, synthetic_clip):
    with pytest.raises(RuntimeError, match="failed to open"):
        VideoReader(tmp_path / "missing.mp4")
    with pytest.raises(ValueError, match="skip"):
        VideoReader(synthetic_clip, skip=0)
