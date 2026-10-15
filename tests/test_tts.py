"""The commentary audio track: clips placed at their times, never overlapping."""

import numpy as np

from cricsense.commentary.tts import GAP_S, RATE, place, read_wav, style_for, write_wav


def test_clips_are_placed_at_their_times_and_pushed_when_the_last_runs_on():
    one_s = np.full(RATE, 0.5, np.float32)
    track, placed = place([(1.0, one_s), (1.5, one_s), (5.0, one_s)], duration_s=8.0)
    assert placed[0] == (1.0, 2.0)
    assert abs(placed[1][0] - (2.0 + GAP_S)) < 1e-9  # pushed, not overlapped
    assert placed[2] == (5.0, 6.0)
    assert track[int(0.5 * RATE)] == 0 and track[int(1.5 * RATE)] > 0
    assert abs(np.abs(track).max() - 0.89) < 1e-3


def test_wav_round_trip(tmp_path):
    x = np.sin(np.linspace(0, 100, RATE)).astype(np.float32) * 0.5
    write_wav(tmp_path / "a.wav", x)
    y = read_wav((tmp_path / "a.wav").read_bytes())
    assert len(y) == len(x) and np.abs(y - x).max() < 1e-3


def test_big_moments_get_an_excited_delivery():
    assert "excited" in style_for("ball", "SIX! Axar launches it.")
    assert "calm" in style_for("after", "India 91 for 3.")
    assert "brisk" in style_for("ball", "Kohli works it away for one.")


def test_f5_says_what_it_needs():
    """The cloned-voice engine needs a recording and its exact words; without them it says so."""
    import pytest

    from cricsense.commentary.tts import STYLES, TTSConfig, synthesize

    with pytest.raises(RuntimeError, match="ref_audio"):
        synthesize(TTSConfig(engine="f5", ref_audio="voices/missing.wav", ref_text="hello"), None, "Four!", STYLES["big"])
    with pytest.raises(RuntimeError, match="ref_text"):
        synthesize(TTSConfig(engine="f5", ref_audio="README.md", ref_text=""), None, "Four!", STYLES["big"])
