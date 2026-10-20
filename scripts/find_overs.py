"""When each over was bowled, from a whole-match score scan - to choose what to cut.

    python scripts/find_overs.py outputs/replay-super-8s-aus-vs-ind/scoreboard.json
    python scripts/find_overs.py <scan> --innings 2 --min-runs 15     # the big overs of the chase

Run `make scan MATCH=...` first. For every over: the time the score bar first showed it (seconds
into the video), the bowler, the score before and after, the runs and wickets in it, and a
`make clip` line that cuts it with a little margin. Times come from the score bar, so an over the
bar did not show cleanly is missing or approximate - check the clip before relying on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cricsense.fuse.outcome import batting_side, score_track, trusted_states
from cricsense.identity.match_sheet import MatchSheet
from cricsense.io import read_doc
from cricsense.schema import ScoreboardDoc

ROOT = Path(__file__).resolve().parents[1]
BEFORE_S = 20.0  # the clip starts this long before the over's first ball is shown on the bar
AFTER_S = 15.0  # ... and ends this long after the next over appears


def hms(t: float) -> str:
    return f"{int(t // 3600):d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scan", type=Path, help="outputs/<match>/scoreboard.json")
    ap.add_argument("--innings", type=int, choices=(1, 2))
    ap.add_argument("--min-runs", type=int, default=0, help="only overs with at least this many runs")
    ap.add_argument("--video", help="the match video, for the make clip lines (default: Cricket match/<match>.mp4)")
    args = ap.parse_args(argv)

    match = args.scan.resolve().parent.name
    sheet = MatchSheet.from_yaml(ROOT / "configs" / "matches" / f"{match}.yaml")
    squad = set(sheet.all_names)
    team_of = {n: t.code for t in sheet.teams for n in t.squad}
    board = read_doc(args.scan, ScoreboardDoc)
    states = trusted_states(board.readings, squad, team_of)
    starts = [b.t for a, b in zip(states, states[1:]) if batting_side(a, team_of) != batting_side(b, team_of)]
    cut = starts[0] if starts else float("inf")
    track = score_track(board.readings, starts)
    # the bowler of each over: the full states name him; the score-only track does not
    bowler_at = [(s.t, s.legal_balls, s.bowler) for s in states]
    video = args.video or f"Cricket match/{match}.mp4"

    for k, (lo, hi) in enumerate(((0.0, cut), (cut, float("inf")))):
        if args.innings and args.innings != k + 1:
            continue
        inn = [s for s in track if lo <= s.t < hi]
        if not inn:
            continue
        side = next((batting_side(s, team_of) for s in states if lo <= s.t < hi and batting_side(s, team_of)), "?")
        print(f"\ninnings {k + 1} ({side} batting)")
        print(f"  {'over':>4}  {'starts':>8}  {'video s':>7}  {'bowler':<11} {'score':>11}  {'runs':>4} {'wkts':>4}")
        for over in range(20):
            first = next((s for s in inn if s.legal_balls >= over * 6 and s.legal_balls < over * 6 + 6), None)
            end = next((s for s in inn if s.legal_balls >= (over + 1) * 6), None)
            if first is None:
                print(f"  {over + 1:>4}  (not read)")
                continue
            before = max((s for s in inn if s.legal_balls <= over * 6), key=lambda s: s.t, default=None)
            bowler = next((b for t, lb, b in bowler_at if lo <= t < hi and over * 6 < lb <= over * 6 + 6), "?")
            runs = (end.runs - before.runs) if end and before else None
            wkts = (end.wickets - before.wickets) if end and before else None
            if args.min_runs and (runs is None or runs < args.min_runs):
                continue
            t0 = max(0.0, (before.t if before else first.t) - BEFORE_S)
            t1 = (end.t if end else first.t + 300) + AFTER_S
            score = f"{before.runs}-{before.wickets}" if before else "?"
            score += f" > {end.runs}-{end.wickets}" if end else ""
            print(f"  {over + 1:>4}  {hms(t0):>8}  {t0:7.0f}  {bowler.title():<11} {score:>11}  {'' if runs is None else runs:>4} {'' if wkts is None else wkts:>4}"
                  f"   make clip MATCH=\"{video}\" START={t0:.0f} END={t1:.0f} CLIP={match[:12]}_i{k + 1}_o{over + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
