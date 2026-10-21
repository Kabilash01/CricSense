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


class _Reply:
    def __init__(self, status: int, text: str = "{}", headers: dict | None = None) -> None:
        self.status_code, self.text, self.headers = status, text, headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise llm.requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_a_busy_free_tier_is_waited_out_but_a_spent_daily_quota_is_not(monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)

    def answers(*replies):
        it = iter(replies)
        monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(it))

    answers(_Reply(429, '{"error": {"details": [{"retryDelay": "7s"}]}}'), _Reply(503), _Reply(200, "ok"))
    assert llm._post("u", headers={}, json={}, timeout=1).text == "ok"
    assert waits == [7.0, 10.0]  # the API's own delay, then the second back-off step

    for final in ('{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}',  # Gemini: the day's quota
                  '{"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted"}}'):  # OpenAI: no credit
        waits.clear()
        answers(_Reply(429, final))
        try:
            llm._post("u", headers={}, json={}, timeout=1)
            raise AssertionError("waiting cannot fix this: it must fail at once")
        except llm.requests.HTTPError:
            assert waits == []


def test_the_non_striker_is_the_strikers_partner_not_the_state_after_a_single():
    from types import SimpleNamespace

    from cricsense.commentary.facts import non_striker

    after_a_single = SimpleNamespace(striker="HEAD", non_striker="MARSH")  # Marsh faced and ran: ends swapped
    assert non_striker(after_a_single, "MARSH") == "HEAD"
    assert non_striker(SimpleNamespace(striker="MARSH", non_striker="HEAD"), "MARSH") == "HEAD"  # a dot ball
    assert non_striker(SimpleNamespace(striker="MAXWELL", non_striker="HEAD"), "MARSH") is None  # Marsh out


def test_a_ball_hit_for_six_is_never_said_to_be_left():
    from types import SimpleNamespace

    from cricsense.commentary.facts import swing_of
    from cricsense.schema import EventType

    left = SimpleNamespace(swing="left")
    six = {EventType.BOUNDARY: SimpleNamespace(payload={"kind": "SIX"})}
    assert swing_of(left, False, None, six) is None  # score only, but a six
    assert swing_of(left, True, 0, {}) is None  # the bat touched it
    assert swing_of(left, False, 1, {}) is None  # runs off the bat
    assert swing_of(left, False, 0, {}) == "left"  # nothing says otherwise
    assert swing_of(SimpleNamespace(swing="played"), False, 0, {}) == "played"


def test_more_frames_than_the_provider_takes_go_as_one_tiled_picture(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from cricsense.commentary import vlm

    frames = [cv2.imencode(".jpg", np.full((h, 300, 3), 40 * i, np.uint8))[1].tobytes() for i, h in enumerate((300, 300, 300, 170))]
    sent = []

    def fake(cfg, key, system, user, images=None):
        sent.append((user, images))
        return '{"stance": "upright", "foot": "front", "bat_finish": "high", "ball_side": "off", "clear": 0.9}'

    monkeypatch.setattr(vlm, "complete", fake)
    vlm.read_shot(LLMConfig("openai", "m", max_images=3), "k", frames, Cache(tmp_path / "c.json"))
    user, images = sent[0]
    assert len(images) == 1 and user.endswith(vlm.TILED)
    grid = cv2.imdecode(np.frombuffer(images[0], np.uint8), cv2.IMREAD_COLOR)
    assert grid.shape[:2] == (2 * 768, 2 * 768)  # 2x2 cells, time order left to right
    sent.clear()
    vlm.read_shot(LLMConfig("openai", "m"), "k", frames, Cache(tmp_path / "d.json"))
    assert len(sent[0][1]) == 4 and sent[0][0] == vlm.PROMPT  # no limit: the frames as they are
