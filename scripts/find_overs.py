"""When each over was bowled, from a whole-match score scan - to choose what to cut.

    python scripts/find_overs.py outputs/replay-super-8s-aus-vs-ind/scoreboard.json
    python scripts/find_overs.py <scan> --innings 2 --min-runs 15     # the big overs of the chase

Run `make scan MATCH=...` first. For every over: the time the score bar first showed it (seconds
into the video), the bowler, the score before and after, the runs and wickets in it, and a
`make clip` line that cuts it with a little margin. Times come from the score bar, so an over the
bar did not show cleanly is missing or approximate - check the clip before relying on it.
(`python main.py <match video> --overs 1:3` cuts and analyses overs in one go.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cricsense.fuse.overs import overs
from cricsense.identity.match_sheet import MatchSheet
from cricsense.io import read_doc
from cricsense.schema import ScoreboardDoc

ROOT = Path(__file__).resolve().parents[1]


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
    video = args.video or f"Cricket match/{match}.mp4"
    innings = None
    for o in overs(read_doc(args.scan, ScoreboardDoc), sheet):
        if (args.innings and o.innings != args.innings) or (args.min_runs and (o.runs is None or o.runs < args.min_runs)):
            continue
        if o.innings != innings:
            innings = o.innings
            print(f"\ninnings {o.innings} ({o.batting or '?'} batting)")
            print(f"  {'over':>4}  {'starts':>8}  {'video s':>7}  {'bowler':<11} {'score':>13}  {'runs':>4} {'wkts':>4}")
        score = (f"{o.before[0]}-{o.before[1]}" if o.before else "?") + (f" > {o.after[0]}-{o.after[1]}" if o.after else "")
        print(f"  {o.over:>4}  {hms(o.start_s):>8}  {o.start_s:7.0f}  {(o.bowler or '?').title():<11} {score:>13}  "
              f"{'' if o.runs is None else o.runs:>4} {'' if o.wickets is None else o.wickets:>4}"
              f"   make clip MATCH=\"{video}\" START={o.start_s:.0f} END={o.end_s:.0f} CLIP={match[:16].rstrip('-_')}_i{o.innings}_o{o.over}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
