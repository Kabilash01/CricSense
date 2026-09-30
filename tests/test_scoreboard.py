"""Score-bar reading, on two real crops from the SA v IND final (ICC T20 WC 2024 graphics)."""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from cricsense.identity.scoreboard_ocr import (
    ScoreboardLayout,
    ScoreboardReader,
    clean_name,
    parse_overs,
    parse_score,
    parse_speed_kph,
)
from tests.conftest import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"
LAYOUT = ScoreboardLayout.from_yaml(REPO_ROOT / "configs" / "scoreboard.yaml")
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def frame_with(crop_name: str, size=(1920, 1080)) -> np.ndarray:
    """A full frame with the fixture score bar pasted back where it was cut from (y 930-1030)."""
    frame = np.full((1080, 1920, 3), 40, np.uint8)
    frame[930:1030] = cv2.imread(str(FIXTURES / crop_name))
    return frame if size == (1920, 1080) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


@pytest.mark.parametrize(
    "text, expected",
    [("82-3", (82, 3)), ("- 82-3 ]", (82, 3)), ("106 - 3", (106, 3)), ("123/4", (123, 4)), ("82-13", (None, None))],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    "text, balls", [("11.1 OVERS", 67), ("11 OVERS", 66), ("3 oVERS", 18), ("1.1", 7), ("11.7", None), ("", None)]
)
def test_parse_overs(text, balls):
    assert parse_overs(text) == balls


@pytest.mark.parametrize(
    "text, kph",
    [("SPEED 81.4KPH % 50.6 MPH", 81.4), ("SPEED 142 KPH", 142.0), ("SPEED 1.4 KPH", None), ("OCOO0000O", None)],
)
def test_parse_speed(text, kph):
    assert parse_speed_kph(text) == kph


def test_clean_name_snaps_to_squad():
    squad = ("KOHLI", "SHAMSI", "DE KOCK")
    assert clean_name("KOHL1", squad) == "KOHLI"
    assert clean_name("AMSI", ()) == "AMSI"
    assert clean_name("|", squad) is None


@needs_tesseract
def test_reads_the_bar_before_the_ball():
    r = ScoreboardReader(LAYOUT).read(frame_with("scorebar_before_ball.png"), frame_id=100, t=4.0)
    assert r is not None
    assert [(b.name, b.runs, b.balls, b.on_strike) for b in r.batters] == [
        ("AXAR", 29, 23, False),
        ("KOHLI", 39, 32, True),
    ]
    assert (r.runs, r.wickets, r.legal_balls) == (82, 3, 66)
    assert (r.bowler, r.bowler_wickets, r.bowler_runs, r.bowler_legal_balls) == ("SHAMSI", 0, 7, 6)
    assert r.speed_kph is None


@needs_tesseract
def test_reads_the_radar_speed_after_the_ball():
    r = ScoreboardReader(LAYOUT).read(frame_with("scorebar_radar_speed.png"), frame_id=700, t=28.0)
    assert r.speed_kph == 81.4
    assert r.legal_balls == 67 and r.bowler_legal_balls == 7 and r.batters[1].balls == 33


@needs_tesseract
def test_layout_scales_to_720p():
    r = ScoreboardReader(LAYOUT).read(frame_with("scorebar_radar_speed.png", size=(1280, 720)), frame_id=0, t=0.0)
    assert (r.runs, r.wickets) == (82, 3) and r.batters[1].on_strike


def test_hidden_bar_gives_no_reading():
    reader = ScoreboardReader(LAYOUT)
    assert reader.read(np.full((1080, 1920, 3), 40, np.uint8), frame_id=0, t=0.0) is None


@needs_tesseract
def test_unchanged_fields_are_not_re_read():
    reader = ScoreboardReader(LAYOUT)
    frame = frame_with("scorebar_before_ball.png")
    reader.read(frame, 0, 0.0)
    first = reader.ocr_calls
    reader.read(frame, 12, 0.48)
    assert first == len(LAYOUT.fields) and reader.ocr_calls == first
    reader.read(frame_with("scorebar_radar_speed.png"), 24, 0.96)
    assert first < reader.ocr_calls < 2 * first  # only the fields that changed


def test_impossible_ocr_values_become_unknown_not_errors():
    from cricsense.identity.scoreboard_ocr import _within

    assert _within(114, 0, 10) is None and _within(3, 0, 10) == 3 and _within(None, 0, 10) is None


def test_surname_prefixes_dropped_by_ocr_still_match_the_squad():
    from cricsense.identity.scoreboard_ocr import clean_name

    squad = ("DE KOCK", "STUBBS", "VAN DER DUSSEN", "SHARMA", "ROHIT SHARMA")
    assert clean_name("KOCK", squad) == "DE KOCK"
    assert clean_name("RKOCK", squad) == "DE KOCK"
    assert clean_name("DUSSEN", squad) == "VAN DER DUSSEN"
    assert clean_name("SHARMA", squad) == "SHARMA"  # exact full name wins before the surname rule
