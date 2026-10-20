"""How much of a whole match the score bar was read for: ball counts seen per innings.

    python scripts/match_coverage.py outputs/replay-final-sa-vs-ind/scoreboard.json

Run `cricsense scoreboard --video <whole match>` first. For each innings: how many of the 121
ball counts (0 to 120 legal balls) appear in a trusted score-bar state, how many steps were
exactly one ball (each such step is one ball's outcome read), and the final score read. Compare
the final scores with the official scorecard: they are the check that nothing went wrong.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scan", type=Path, help="scoreboard.json of a whole match")
    ap.add_argument("--sheet", type=Path, help="default: configs/matches/<the scan's folder name>.yaml")
    args = ap.parse_args(argv)

    sheet = MatchSheet.from_yaml(args.sheet or ROOT / "configs" / "matches" / f"{args.scan.resolve().parent.name}.yaml")
    squad = set(sheet.all_names)
    team_of = {n: t.code for t in sheet.teams for n in t.squad}
    board = read_doc(args.scan, ScoreboardDoc)
    # full states (names and figures) and score-only states (compact or moved bars)
    states = trusted_states(board.readings, squad, team_of)
    starts = [b.t for a, b in zip(states, states[1:]) if batting_side(a, team_of) != batting_side(b, team_of)]
    track = score_track(board.readings, starts)
    cut = starts[0] if starts else float("inf")
    for k, (lo, hi) in enumerate(((0.0, cut), (cut, float("inf")))):
        both = sorted((s for s in states + track if lo <= s.t < hi), key=lambda s: s.t)
        if not both:
            print(f"innings {k + 1}: nothing read")
            continue
        seen = {s.legal_balls for s in both}
        inn_track = [s for s in track if lo <= s.t < hi]
        steps = sum(1 for a, b in zip(inn_track, inn_track[1:]) if b.legal_balls == a.legal_balls + 1)
        last = max(both, key=lambda s: (s.legal_balls, s.runs, s.t))
        side = next((batting_side(s, team_of) for s in both if batting_side(s, team_of)), "?")
        print(f"innings {k + 1} ({side}): ball counts seen {len(seen)}/121, single-ball steps {steps}, "
              f"final {last.runs}-{last.wickets} after {last.legal_balls // 6}.{last.legal_balls % 6} overs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
