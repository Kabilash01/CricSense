"""Commentary: templates say only the facts; LLM lines are checked against them."""

from dataclasses import replace

from cricsense.commentary import llm
from cricsense.commentary.facts import BallFacts
from cricsense.commentary.llm import Cache, LLMConfig, check_line, llm_lines
from cricsense.commentary.templates import after_line, ball_line, ordinal

SQUAD = {"KOHLI", "AXAR", "SHAMSI", "NORTJE", "BUMRAH", "DE KOCK"}

SIX = BallFacts(
    delivery_id=4, over="11.5", bowler="Shamsi", striker="Axar", non_striker="Kohli", batting_team="India",
    release_frame=3067, contact_frame=3080, outcome_frame=3400, next_release_frame=3974,
    length_class="full", length_m=3.9, line_m=0.12, bounce_confirmed=True, speed_kph=89.2, speed_source="radar",
    bat_contact=True, runs=6, off_bat=6, extras=0, kind="legal", boundary="SIX", wicket=False, score="91-3",
    striker_score="36 (25)", milestone=None, over_complete=False, overs_completed=None, over_runs=None,
    bowler_figures="0-16 (1.5)",
)


def test_template_lines_use_only_the_facts():
    text = ball_line(SIX) + " " + after_line(SIX)
    assert "89.2" in text and "SIX" in text and "India 91 for 3" in text
    assert check_line(ball_line(SIX), SIX.as_dict(), SQUAD, 40) is None


def test_hedges_estimates_and_unconfirmed_bounces():
    f = replace(SIX, speed_kph=154.0, speed_source="estimate", bounce_confirmed=False, runs=1, off_bat=1, boundary=None)
    line = ball_line(f)
    assert "around 155 kph" in line and "looks full" in line


def test_unknown_outcome_describes_the_delivery_only():
    f = replace(SIX, runs=None, off_bat=None, boundary=None, score=None)
    assert ball_line(f) == "Shamsi, full, 89.2 kph." or ball_line(f).startswith("Shamsi")
    assert after_line(f) is None


def test_end_of_over_names_the_over_properly():
    f = replace(SIX, over="11.6", over_complete=True, overs_completed=12, over_runs=11, boundary=None, runs=2, off_bat=2, score="93-3", bowler_figures="0-18 (2.0)")
    assert after_line(f).startswith("End of the 12th over, 11 from it. India 93 for 3.")
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]


def test_fact_check_rejects_invented_numbers_people_and_events():
    facts = SIX.as_dict()
    assert check_line("SIX! Axar launches Shamsi at 89.2 kph into the stands!", facts, SQUAD, 30) is None
    assert "number 92" in check_line("Shamsi at 92 kph, SIX!", facts, SQUAD, 30)
    assert "Bumrah" in check_line("Bumrah will be pleased, SIX for Axar!", facts, SQUAD, 30)
    assert "too long" in check_line("word " * 31, facts, SQUAD, 30)
    single = replace(SIX, runs=1, off_bat=1, boundary=None).as_dict()
    assert "wicket" in check_line("Axar is out!", single, SQUAD, 30)
    assert check_line("Axar hits it out of the middle for one.", single, SQUAD, 30) is None


def test_llm_reply_is_parsed_and_cached(tmp_path, monkeypatch):
    calls = []

    def fake(cfg, key, system, user, images=None):
        calls.append(user)
        return '```json\n{"lines": [{"delivery_id": 4, "ball": "Shamsi tosses it up, 89.2 kph, and Axar launches it for SIX!", "after": null}]}\n```'

    monkeypatch.setattr(llm, "complete", fake)
    cache = Cache(tmp_path / "c.json")
    cfg = LLMConfig("anthropic", "test-model")
    for _ in range(2):
        out = llm_lines(cfg, "k", [SIX.as_dict()], {4: (20, 20)}, cache)
    assert out[4]["ball"].startswith("Shamsi tosses") and out[4]["after"] is None
    assert len(calls) == 1  # the second run came from the cache


def test_the_over_counts_only_as_a_whole():
    f = replace(SIX, over="11.6", over_complete=True, overs_completed=12, boundary=None, runs=2, off_bat=2, score="93-3").as_dict()
    assert "number 11" in check_line("India 93-3, 11 overs gone.", f, SQUAD, 30)
    assert check_line("India 93-3 after 12 overs.", f, SQUAD, 30) is None


def test_shot_is_named_from_the_posture_by_the_textbook():
    from cricsense.commentary.vlm import name_shot

    seen = {"knee_on_ground": True, "stance": "kneeling", "foot": "front", "bat_finish": "high", "ball_side": "leg"}
    assert name_shot(seen) == "slog sweep"
    assert name_shot({**seen, "bat_finish": "low"}) == "sweep"
    upright = {"knee_on_ground": False, "stance": "upright", "foot": "front", "bat_finish": "high", "ball_side": "straight"}
    assert name_shot(upright) == "straight drive"
    assert name_shot({**upright, "foot": "back", "ball_side": "leg"}) == "pull"
    assert name_shot({**upright, "ball_side": "unclear"}) is None
    assert name_shot({**upright, "stance": "unclear", "knee_on_ground": False}) is None


def test_a_shot_is_reported_only_when_readings_agree(tmp_path, monkeypatch):
    from cricsense.commentary import vlm

    seen = {"knee_on_ground": True, "stance": "kneeling", "foot": "front", "bat_finish": "high", "ball_side": "leg", "clear": 0.9}
    drive = {"knee_on_ground": False, "stance": "upright", "foot": "front", "bat_finish": "high", "ball_side": "off", "clear": 0.9}
    replies = iter([seen, drive, seen])
    monkeypatch.setattr(vlm, "complete", lambda *a, **k: __import__("json").dumps(next(replies)))
    got = vlm.read_shot(LLMConfig("openai", "m"), "k", [b"jpeg"], Cache(tmp_path / "c.json"))
    assert got["shot"] == "slog sweep" and got["votes"] == "2/3"
    replies = iter([seen, drive, {**drive, "ball_side": "straight"}])
    assert vlm.read_shot(LLMConfig("openai", "m"), "k", [b"other"], Cache(tmp_path / "d.json")) is None  # no two agree


def test_fact_check_rejects_how_a_batter_got_out():
    out = replace(SIX, runs=0, off_bat=0, boundary=None, wicket=True).as_dict()
    assert check_line("Axar is out! What a moment.", out, SQUAD, 30) is None
    for line in ("Axar is bowled! Through the gate!", "Caught! Axar has to go.", "Axar edges it and he's out!", "Plumb lbw, Axar is out!"):
        assert "how the batter was out" in check_line(line, out, SQUAD, 30), line
