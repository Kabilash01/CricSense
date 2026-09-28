import numpy as np

from cricsense.passes.players import _feet_visible, _names_by_role
from cricsense.schema import BatterLine, Role, ScoreboardReading

SCORE_BAR = [(0.0, 0.85, 1.0, 1.0)]


def test_feet_hidden_by_score_bar_or_frame_edge():
    size = (1920, 1080)
    assert _feet_visible(np.array([900, 500, 960, 700]), size, SCORE_BAR)
    assert not _feet_visible(np.array([900, 700, 960, 918]), size, SCORE_BAR)  # stops at the bar's top edge
    assert not _feet_visible(np.array([900, 900, 960, 1080]), size, [])  # cut by the frame border


def reading(striker: str, other: str, bowler: str | None) -> ScoreboardReading:
    return ScoreboardReading(
        frame_id=0, t=0.0,
        batters=[BatterLine(name=striker, runs=1, balls=1, on_strike=True), BatterLine(name=other, runs=1, balls=1, on_strike=False)],
        runs=10, wickets=1, legal_balls=6, bowler=bowler, bowler_wickets=0, bowler_runs=5, bowler_legal_balls=6,
    )


def test_names_are_voted_over_recent_readings():
    names = _names_by_role([reading("KOHLI", "AXAR", "NORTJE"), reading("KOHLI", "AXAR", "OJOI"), reading("KOHLI", "AXAR", "NORTJE")])
    assert names == {Role.STRIKER: "KOHLI", Role.NON_STRIKER: "AXAR", Role.BOWLER: "NORTJE"}


def test_tracker_handover_is_merged_into_the_original_id():
    from cricsense.passes.players import merge_handovers
    from cricsense.schema import PlayerFrame

    def pf(fid, sid, box, feet=True):
        return PlayerFrame(frame_id=fid, t=fid / 25, sid=sid, box=box, role=Role.NON_STRIKER, pitch_xy=(19.0, 1.5) if feet else None)

    old = [pf(k, 3, (456, 673, 678, 947), feet=False) for k in range(0, 11)]           # stops at the score bar
    new = [pf(k, 6, (470, 650, 690, 1080), feet=False) for k in range(9, 20)]          # legs below the bar too
    other = [pf(k, 7, (1500, 300, 1540, 420)) for k in range(9, 20)]                     # someone else entirely
    frames, merges = merge_handovers(old + new + other)
    assert merges == {6: 3}
    assert {f.sid for f in frames} == {3, 7}
    assert len([f for f in frames if f.sid == 3]) == 20  # frames 0-19, one box per frame
