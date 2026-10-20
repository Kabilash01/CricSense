"""When each over was bowled in a whole match, from its score-bar scan - to choose what to cut.

Times come from the score bar: an over starts where the bar first counts its first ball, and
its clip runs from a little before the previous over's last state to a little after the next
over appears. An over the bar did not show cleanly is missing or approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

from cricsense.fuse.outcome import batting_side, score_track, trusted_states
from cricsense.identity.match_sheet import MatchSheet
from cricsense.schema import ScoreboardDoc

BEFORE_S = 20.0  # a clip starts this long before the over's first state on the bar
AFTER_S = 15.0  # ... and ends this long after the next over appears
UNSEEN_OVER_S = 300.0  # the length assumed for an over whose end the bar did not show


@dataclass(frozen=True)
class Over:
    innings: int  # 1 or 2
    over: int  # 1..20
    batting: str | None  # team code
    bowler: str | None
    start_s: float  # clip window, seconds into the video
    end_s: float
    before: tuple[int, int] | None  # (runs, wickets) at the start
    after: tuple[int, int] | None  # ... and at the end

    @property
    def runs(self) -> int | None:
        return self.after[0] - self.before[0] if self.before and self.after else None

    @property
    def wickets(self) -> int | None:
        return self.after[1] - self.before[1] if self.before and self.after else None


def overs(board: ScoreboardDoc, sheet: MatchSheet) -> list[Over]:
    """Every over the score bar shows, both innings, in order."""
    squad = set(sheet.all_names)
    team_of = {n: t.code for t in sheet.teams for n in t.squad}
    states = trusted_states(board.readings, squad, team_of)
    starts = [b.t for a, b in zip(states, states[1:]) if batting_side(a, team_of) != batting_side(b, team_of)]
    cut = starts[0] if starts else float("inf")
    track = score_track(board.readings, starts)
    out: list[Over] = []
    for k, (lo, hi) in enumerate(((0.0, cut), (cut, float("inf")))):
        inn = [s for s in track if lo <= s.t < hi]
        named = [s for s in states if lo <= s.t < hi]
        side = next((batting_side(s, team_of) for s in named if batting_side(s, team_of)), None)
        for over in range(20):
            first = next((s for s in inn if over * 6 <= s.legal_balls < over * 6 + 6), None)
            if first is None:
                continue
            end = next((s for s in inn if s.legal_balls >= (over + 1) * 6), None)
            before = max((s for s in inn if s.legal_balls <= over * 6), key=lambda s: s.t, default=None)
            bowler = next((s.bowler for s in named if over * 6 < s.legal_balls <= over * 6 + 6), None)
            out.append(Over(
                innings=k + 1, over=over + 1, batting=side, bowler=bowler,
                start_s=max(0.0, (before.t if before else first.t) - BEFORE_S),
                end_s=(end.t if end else first.t + UNSEEN_OVER_S) + AFTER_S,
                before=(before.runs, before.wickets) if before else None,
                after=(end.runs, end.wickets) if end else None,
            ))  # fmt: skip
    return out


def innings_totals(board: ScoreboardDoc, sheet: MatchSheet) -> list[tuple[str | None, int, int, int, int]]:
    """Per innings: (batting team, runs, wickets, legal balls, ball counts seen of 121)."""
    squad = set(sheet.all_names)
    team_of = {n: t.code for t in sheet.teams for n in t.squad}
    states = trusted_states(board.readings, squad, team_of)
    starts = [b.t for a, b in zip(states, states[1:]) if batting_side(a, team_of) != batting_side(b, team_of)]
    cut = starts[0] if starts else float("inf")
    track = score_track(board.readings, starts)
    totals = []
    for lo, hi in ((0.0, cut), (cut, float("inf"))):
        both = [s for s in states + track if lo <= s.t < hi]
        if not both:
            continue
        last = max(both, key=lambda s: (s.legal_balls, s.runs, s.t))
        side = next((batting_side(s, team_of) for s in states if lo <= s.t < hi and batting_side(s, team_of)), None)
        totals.append((side, last.runs, last.wickets, last.legal_balls, len({s.legal_balls for s in both})))
    return totals
