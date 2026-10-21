"""Score engine: the laws of the game, and validation / re-sync against the score bar."""

from cricsense.fuse.outcome import Outcome, ScoreState
from cricsense.fuse.score_engine import ScoreEngine, score_innings


def bar(t, runs, wkts, balls, a, b, strike="A", bowler="X", fig=None, names=("A", "B")):
    return ScoreState(
        t, runs, wkts, balls, bowler,
        ((names[0], *a, strike == names[0]), (names[1], *b, strike == names[1])),
        *(fig or (None, None, None)),
    )


def ball(before, after, striker, runs, off_bat, kind="legal", wicket=False, bowler="X"):
    return Outcome(before, after, striker, bowler, runs, off_bat, runs - off_bat, wicket, kind, {4: "FOUR", 6: "SIX"}.get(off_bat))


START = bar(0, 50, 2, 60, (20, 15), (10, 9), fig=(12, 6, 0))  # 10.0 overs, A on strike, X 0-12 (1.0)


def test_single_rotates_strike_and_updates_figures():
    e = ScoreEngine(START)
    after = bar(5, 51, 2, 61, (21, 16), (10, 9), strike="B", fig=(13, 7, 0))
    checks = e.apply(ball(START, after, "A", 1, 1))
    checks.update(e.compare(after))
    assert all(checks.values()), checks
    assert e.s.striker == "B" and e.s.batters["A"].runs == 21 and e.s.bowlers["X"].balls == 7


def test_wide_is_not_a_ball_faced_and_counts_against_the_bowler():
    e = ScoreEngine(START)
    e.apply(ball(START, START, "A", 1, 0, kind="wide"))
    assert (e.s.legal_balls, e.s.batters["A"].balls, e.s.bowlers["X"].runs, e.s.striker) == (60, 15, 13, "A")


def test_byes_are_not_the_bowlers_and_a_no_ball_is_faced():
    e = ScoreEngine(START)
    e.apply(ball(START, START, "A", 1, 0))  # a bye: striker changes, bowler not charged
    assert (e.s.bowlers["X"].runs, e.s.striker) == (12, "B")
    e.apply(ball(START, START, "B", 1, 0, kind="no_ball"))  # no-ball, no runs off the bat
    assert (e.s.legal_balls, e.s.batters["B"].balls, e.s.bowlers["X"].runs) == (61, 10, 13)


def test_end_of_over_changes_strike_and_expects_a_new_bowler():
    s = bar(0, 50, 2, 65, (20, 15), (10, 9))  # 10.5
    e = ScoreEngine(s)
    e.apply(ball(s, s, "A", 0, 0))
    assert e.s.over == "11.0" and e.s.striker == "B" and e.awaiting_bowler
    checks = e.apply(ball(s, s, "B", 0, 0, bowler="Y"))
    assert checks["new bowler at the over"] and e.s.bowler == "Y"


def test_wicket_takes_the_incoming_batter_from_the_bar():
    after = bar(5, 50, 3, 61, (0, 0), (10, 9), strike="C", names=("C", "B"))
    states = [START, after]
    out = score_innings([(0, 10, 0.4, ball(START, after, "A", 0, 0, wicket=True))], states, 25.0)
    st = out[0]
    assert st.wickets == 3 and st.striker == "C" and st.checks["score"] and st.checks["on strike"]
    assert any("incoming batter" in n for n in st.notes)


def test_a_missed_ball_is_resynced_from_the_bar_and_flagged():
    b1 = bar(10, 51, 2, 61, (21, 16), (10, 9), strike="B")
    b2 = bar(20, 55, 2, 62, (21, 16), (14, 10), strike="B")  # the engine never saw ball 2
    b3 = bar(30, 56, 2, 63, (21, 16), (15, 11), strike="A")
    deliveries = [
        (0, 0, 1.0, ball(START, b1, "A", 1, 1)),
        (1, 300, 12.0, None),
        (2, 550, 22.0, ball(b2, b3, "B", 1, 1)),
    ]
    out = score_innings(deliveries, [START, b1, b2, b3], 25.0)
    assert [s.source for s in out] == ["engine", "no_evidence", "resync"]
    assert out[2].runs == 56 and all(out[2].checks[k] for k in ("score", "wickets", "overs"))


def test_a_score_only_dismissal_that_names_who_faced_is_still_not_applied_by_the_laws():
    # the score after the wicket was read without the batters: runs off the bat unknown
    after = bar(5, 50, 3, 61, (0, 0), (10, 9), strike="C", names=("C", "B"))
    o = Outcome(START, after, "A", "X", 0, None, None, True, "legal", None)
    out = score_innings([(0, 10, 0.4, o)], [START], 25.0)
    assert out[0].source == "no_evidence" and "score only" in out[0].notes[0]
