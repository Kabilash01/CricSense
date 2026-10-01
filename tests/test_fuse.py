"""Ball outcomes from score-bar changes, and bat-on-ball contact."""

from cricsense.fuse.contact import find_contact
from cricsense.fuse.outcome import outcome, trusted_states
from cricsense.schema import BatterLine, ScoreboardReading

SQUAD = {"KOHLI", "AXAR", "SHAMSI", "NORTJE"}


def reading(t, runs, wkts, balls, kohli, axar, strike="KOHLI", bowler="SHAMSI"):
    return ScoreboardReading(
        frame_id=int(t * 25),
        t=t,
        batters=[
            BatterLine(name="KOHLI", runs=kohli[0], balls=kohli[1], on_strike=strike == "KOHLI"),
            BatterLine(name="AXAR", runs=axar[0], balls=axar[1], on_strike=strike == "AXAR"),
        ],
        runs=runs,
        wickets=wkts,
        legal_balls=balls,
        bowler=bowler,
        bowler_runs=None,
        bowler_legal_balls=None,
    )


def seen(t, *args, **kw):  # a state held for two readings, as the bar is read every 0.5 s
    return [reading(t, *args, **kw), reading(t + 0.5, *args, **kw)]


def test_single_misreads_and_backward_scores_are_not_trusted():
    rs = [
        *seen(0, 84, 3, 69, (40, 30), (10, 8), strike="AXAR"),
        reading(3, 48, 3, 69, (40, 30), (10, 8), strike="AXAR"),  # one-off misread
        *seen(5, 24, 3, 69, (40, 30), (10, 8), strike="AXAR"),  # stable but backwards
        *seen(8, 90, 3, 70, (40, 30), (16, 9), strike="AXAR"),
    ]
    states = trusted_states(rs, SQUAD)
    assert [s.runs for s in states] == [84, 90]


def test_six_off_the_bat():
    states = trusted_states(
        [*seen(0, 85, 3, 69, (40, 30), (10, 8), strike="AXAR"), *seen(9, 91, 3, 70, (40, 30), (16, 9), strike="AXAR")], SQUAD
    )
    o = outcome(states, release_t=2.0, next_release_t=30.0)
    assert (o.striker, o.bowler, o.runs, o.off_bat, o.kind, o.boundary, o.wicket) == ("AXAR", "SHAMSI", 6, 6, "legal", "SIX", False)


def test_wide_no_ball_and_wicket():
    base = seen(0, 85, 3, 69, (40, 30), (10, 8))
    wide = trusted_states([*base, *seen(9, 86, 3, 69, (40, 30), (10, 8))], SQUAD)
    assert outcome(wide, 2.0, 30.0).kind == "wide"
    nb = trusted_states([*base, *seen(9, 86, 3, 69, (40, 31), (10, 8))], SQUAD)
    assert outcome(nb, 2.0, 30.0).kind == "no_ball"
    out = ScoreboardReading(**{**reading(9, 85, 4, 70, (0, 0), (10, 8)).model_dump(), "batters": [
        BatterLine(name="AXAR", runs=10, balls=8, on_strike=False), BatterLine(name="NORTJE", runs=0, balls=0, on_strike=True)]})
    wk = trusted_states([*base, out, out.model_copy(update={"t": 9.5})], SQUAD)
    o = outcome(wk, 2.0, 30.0)
    assert o.wicket and o.kind == "legal" and o.runs == 0


def test_no_clean_after_state_gives_no_outcome():
    states = trusted_states(seen(0, 85, 3, 69, (40, 30), (10, 8)), SQUAD)
    assert outcome(states, 2.0, 30.0) is None


def test_contact_needs_the_ball_in_a_bat_box_near_the_striker():
    bats = {105: [{"cls": 1, "conf": 0.8, "box": [900, 500, 960, 600]}], 104: [{"cls": 1, "conf": 0.8, "box": [100, 990, 130, 1010]}]}
    balls = {104: [{"cls": 0, "conf": 0.5, "box": [110, 995, 118, 1003]}], 105: [{"cls": 0, "conf": 0.4, "box": [930, 540, 940, 550]}]}
    overlays = ((0.0, 0.85, 1.0, 1.0),)
    c = find_contact(bats, balls, 100, (880, 450, 1000, 700), (1920, 1080), overlays)
    assert c is not None and c.frame_id == 105  # frame 104's "bat" is the score-bar icon
    assert find_contact(bats, balls, 100, (100, 100, 200, 300), (1920, 1080), overlays) is None  # not the striker's bat
    assert find_contact(bats, {}, 100, None, (1920, 1080), overlays) is None


def test_a_new_innings_is_not_a_score_going_backwards():
    team_of = {"KOHLI": "IND", "AXAR": "IND", "STUBBS": "SA", "DE KOCK": "SA", "SHAMSI": "SA", "BUMRAH": "IND"}
    first = seen(0, 176, 7, 120, (76, 59), (47, 31))
    second = [
        ScoreboardReading(**{**r.model_dump(), "runs": 4, "wickets": 0, "legal_balls": 2, "bowler": "BUMRAH", "t": r.t + 3000,
                             "frame_id": r.frame_id + 75000,
                             "batters": [BatterLine(name="DE KOCK", runs=4, balls=1, on_strike=True),
                                         BatterLine(name="STUBBS", runs=0, balls=1, on_strike=False)]})
        for r in seen(0, 0, 0, 0, (0, 0), (0, 0))
    ]
    squad = SQUAD | {"STUBBS", "DE KOCK", "BUMRAH"}
    assert [s.runs for s in trusted_states([*first, *second], squad)] == [176]  # without teams: rejected
    assert [s.runs for s in trusted_states([*first, *second], squad, team_of)] == [176, 4]


def test_a_score_jumping_faster_than_the_game_allows_is_a_misread():
    rs = [
        *seen(0, 6, 0, 6, (4, 4), (1, 2)),
        *seen(38, 6, 0, 60, (4, 4), (1, 2)),  # "1.0" read as "10.0" 38 s later
        *seen(80, 7, 0, 7, (5, 5), (1, 2)),  # the genuine next ball
    ]
    assert [(s.runs, s.legal_balls) for s in trusted_states(rs, SQUAD)] == [(6, 6), (7, 7)]


def test_score_only_readings_give_runs_and_legal_or_extra():
    from cricsense.fuse.outcome import score_track

    def score_only(t, runs, wkts, balls):
        return [ScoreboardReading(frame_id=int(x * 25), t=x, batters=[BatterLine(name=None, runs=None, balls=None, on_strike=False)] * 2,
                                  runs=runs, wickets=wkts, legal_balls=balls, bowler=None, bowler_runs=None, bowler_legal_balls=None)
                for x in (t, t + 0.5)]
    rs = [*score_only(0, 120, 4, 84), *score_only(30, 126, 4, 85), *score_only(60, 127, 4, 85)]
    track = score_track(rs)
    assert [(s.runs, s.legal_balls) for s in track] == [(120, 84), (126, 85), (127, 85)]
    six = outcome([], 10.0, 40.0, track)
    assert (six.kind, six.runs, six.boundary, six.striker, six.off_bat) == ("legal", 6, "SIX", None, None)
    extra = outcome([], 45.0, 90.0, track)
    assert (extra.kind, extra.runs) == ("extra", 1)
