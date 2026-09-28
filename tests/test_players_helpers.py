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


def test_teams_and_keeper_name_from_the_match_sheet(tmp_path):
    from cricsense.identity.match_sheet import MatchSheet
    from cricsense.passes.players import assign_teams
    from cricsense.schema import PlayerFrame

    sheet_file = tmp_path / "match.yaml"
    sheet_file.write_text(
        "teams:\n  IND: {squad: [KOHLI, AXAR, PANT], keeper: PANT}\n  SA: {squad: [DE KOCK, SHAMSI], keeper: DE KOCK}\n"
    )
    sheet = MatchSheet.from_yaml(sheet_file)

    def pf(sid, role, name=None):
        return PlayerFrame(
            frame_id=0, t=0.0, sid=sid, box=(0, 0, 10, 20), role=role, identity=name,
            identity_source="scoreboard" if name else None,
        )

    frames = [
        pf(1, Role.STRIKER, "KOHLI"), pf(2, Role.NON_STRIKER, "AXAR"), pf(3, Role.BOWLER, "SHAMSI"),
        pf(4, Role.KEEPER), pf(5, Role.UMPIRE), pf(6, Role.FIELDER),
    ]
    out = {f.sid: f for f in assign_teams(frames, sheet, {s: 0 for s in range(1, 7)}, {4: 0.97})}
    assert [out[s].team for s in range(1, 7)] == ["IND", "IND", "SA", "SA", None, "SA"]
    assert (out[4].identity, out[4].identity_source, out[4].identity_conf) == ("DE KOCK", "match_sheet", 0.97)


def test_match_sheet_rejects_a_keeper_outside_the_squad(tmp_path):
    import pytest

    from cricsense.identity.match_sheet import MatchSheet

    f = tmp_path / "match.yaml"
    f.write_text("teams:\n  A: {squad: [X], keeper: Y}\n  B: {squad: [Z]}\n")
    with pytest.raises(ValueError, match="not in the squad"):
        MatchSheet.from_yaml(f)
