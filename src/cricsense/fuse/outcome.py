"""What each ball produced, from how the broadcast score bar changed across it.

The bar is read every half second (scoreboard.json) but some readings are taken while
another graphic sits on it ("BARBADOS 1(521)", a batter misread as another squad player).
States are therefore only trusted when complete, made of squad names, seen in at least two
consecutive readings, and never going backwards. For a delivery, the state before it is the
last trusted one before its release; the state after it is the last trusted one before the
next delivery's release. The difference gives runs, extras, wickets, wide / no-ball and who
was on strike. The score bar is evidence, not the authoritative score.
"""

from __future__ import annotations

from dataclasses import dataclass

from cricsense.schema import ScoreboardReading

MAX_LEGAL_BALLS = 120  # a T20 innings
MIN_REPEATS = 2
SECONDS_PER_BALL = 15.0  # no ball is bowled faster than this, replays and breaks included
MAX_RUNS_PER_BALL = 8  # a six off a no-ball, plus a bit


@dataclass(frozen=True)
class ScoreState:
    t: float  # first time this state was seen
    runs: int
    wickets: int
    legal_balls: int
    bowler: str
    batters: tuple[tuple[str, int, int, bool], ...]  # (name, runs, balls, on strike)
    # the bowler's figures as shown; noisier than the score, so not part of what must be stable
    bowler_runs: int | None = None
    bowler_legal_balls: int | None = None
    bowler_wickets: int | None = None

    @property
    def striker(self) -> tuple[str, int, int, bool] | None:
        on = [b for b in self.batters if b[3]]
        return on[0] if len(on) == 1 else None

    def batter(self, name: str) -> tuple[str, int, int, bool] | None:
        return next((b for b in self.batters if b[0] == name), None)


@dataclass(frozen=True)
class Outcome:
    before: ScoreState
    after: ScoreState
    striker: str | None  # None: from the score alone (the bar did not show the batters)
    bowler: str | None
    runs: int  # total runs added by the ball
    off_bat: int | None  # credited to the striker; None when only the score was read
    extras: int | None
    wicket: bool
    kind: str  # "legal" | "wide" | "no_ball" | "extra" (a wide or a no-ball: the score alone cannot tell)
    boundary: str | None  # "FOUR" | "SIX"
    # when the bar first showed the new score (later states may only change the bowler shown)
    settled_t: float | None = None


def trusted_states(
    readings: list[ScoreboardReading], squad: set[str], team_of: dict[str, str] | None = None
) -> list[ScoreState]:
    """Score-bar states that are complete, plausible and stable, in time order.

    team_of (player -> team) lets a new innings through: when the batters belong to the other
    side, the score starting again from 0 is the second innings, not a misread going backwards."""
    candidates: list[ScoreState] = []
    for r in readings:
        if r.runs is None or r.wickets is None or r.legal_balls is None or r.bowler not in squad:
            continue
        if r.legal_balls > MAX_LEGAL_BALLS or any(b.name not in squad or b.runs is None or b.balls is None for b in r.batters):
            continue
        if sum(b.on_strike for b in r.batters) != 1:
            continue
        candidates.append(
            ScoreState(
                r.t,
                r.runs,
                r.wickets,
                r.legal_balls,
                r.bowler,
                tuple((b.name, b.runs, b.balls, b.on_strike) for b in r.batters),
                r.bowler_runs,
                r.bowler_legal_balls,
                r.bowler_wickets,
            )
        )
    states: list[ScoreState] = []
    run: list[ScoreState] = []
    for s in candidates:
        if run and _key(s) == _key(run[-1]):
            run.append(s)
        else:
            _accept(run, states, team_of)
            run = [s]
    _accept(run, states, team_of)
    return states


def _key(s: ScoreState) -> tuple:
    return (s.runs, s.wickets, s.legal_balls, s.bowler, s.batters)


def batting_side(s: ScoreState, team_of: dict[str, str] | None) -> str | None:
    teams = {team_of.get(b[0]) for b in s.batters} if team_of else set()
    return teams.pop() if len(teams) == 1 else None


def _accept(run: list[ScoreState], states: list[ScoreState], team_of: dict[str, str] | None = None) -> None:
    if len(run) < MIN_REPEATS:
        return
    s = run[0]
    if states:
        last = states[-1]
        side, last_side = batting_side(s, team_of), batting_side(last, team_of)
        new_innings = side is not None and last_side is not None and side != last_side
        if not new_innings and (s.runs < last.runs or s.wickets < last.wickets or s.legal_balls < last.legal_balls):
            return  # a score never goes backwards: a misread
        if not new_innings:
            # nor faster than the game allows: "1.0" misread as "10.0" would otherwise be
            # accepted and then block every genuine state after it as "going backwards"
            balls = s.legal_balls - last.legal_balls
            if balls > 1 + (s.t - last.t) / SECONDS_PER_BALL:
                return
            if s.runs - last.runs > MAX_RUNS_PER_BALL * (max(balls, 0) + 1):
                return
        if _key(s) == _key(last):
            return
    states.append(s)


def score_track(readings: list[ScoreboardReading], innings_starts: list[float] = ()) -> list[ScoreState]:
    """Every stable reading of score, wickets and overs - including the score-only ones, when a
    compact bar or a moved one shows no batters - by the same rules: seen twice, never going
    backwards, never faster than the game. innings_starts (times) allow the reset to 0."""
    starts = sorted(innings_starts)
    track: list[ScoreState] = []
    run: list[ScoreState] = []

    def accept(run: list[ScoreState]) -> None:
        if len(run) < MIN_REPEATS:
            return
        s = run[0]
        if track:
            last = track[-1]
            new_innings = any(last.t < st <= s.t for st in starts)
            if not new_innings:
                if s.runs < last.runs or s.wickets < last.wickets or s.legal_balls < last.legal_balls:
                    return
                balls = s.legal_balls - last.legal_balls
                if balls > 1 + (s.t - last.t) / SECONDS_PER_BALL or s.runs - last.runs > MAX_RUNS_PER_BALL * (balls + 1):
                    return
                if (s.runs, s.wickets, s.legal_balls) == (last.runs, last.wickets, last.legal_balls):
                    return
        track.append(s)

    for r in readings:
        if r.runs is None or r.wickets is None or r.legal_balls is None or r.legal_balls > MAX_LEGAL_BALLS:
            continue
        s = ScoreState(r.t, r.runs, r.wickets, r.legal_balls, r.bowler or "", ())
        if run and (s.runs, s.wickets, s.legal_balls) == (run[-1].runs, run[-1].wickets, run[-1].legal_balls):
            run.append(s)
        else:
            accept(run)
            run = [s]
    accept(run)
    return track


def outcome(
    states: list[ScoreState], release_t: float, next_release_t: float | None, track: list[ScoreState] | None = None
) -> Outcome | None:
    """The ball's outcome, or None if the bar did not show a clean before and after.

    From the full bar when it brackets the ball; else, given the score track, from the score
    alone: runs, legal or extra, wicket - the striker and the runs off the bat stay unknown."""
    full = _full_outcome(states, release_t, next_release_t)
    if full is not None or not track:
        return full
    before = next((s for s in reversed(track) if s.t <= release_t), None)
    limit = next_release_t if next_release_t is not None else release_t + 45.0
    after_candidates = [s for s in track if release_t < s.t < limit]
    if before is None or not after_candidates:
        return None
    after = after_candidates[-1]
    d_runs, d_balls, d_wkts = after.runs - before.runs, after.legal_balls - before.legal_balls, after.wickets - before.wickets
    if d_balls not in (0, 1) or d_runs < 0 or d_wkts < 0:
        return None
    if d_balls == 1:
        kind = "legal"
    elif d_runs >= 1:
        kind = "extra"
    else:
        return None
    # six off a legal ball: nothing but a six adds 6 (no byes or overthrows give exactly 6 here)
    boundary = "SIX" if kind == "legal" and d_runs == 6 else None
    bowler = next((s.bowler for s in reversed(states) if s.t <= release_t), None)
    core = (after.runs, after.wickets, after.legal_balls)
    settled = min(s.t for s in after_candidates if (s.runs, s.wickets, s.legal_balls) == core)
    return Outcome(before, after, None, bowler, d_runs, None, None, d_wkts > 0, kind, boundary, settled)


def _full_outcome(states: list[ScoreState], release_t: float, next_release_t: float | None) -> Outcome | None:
    before = next((s for s in reversed(states) if s.t <= release_t), None)
    limit = next_release_t if next_release_t is not None else release_t + 45.0
    after_candidates = [s for s in states if release_t < s.t < limit]
    if before is None or not after_candidates:
        return None
    after = after_candidates[-1]
    striker = before.striker
    if striker is None:
        return None
    name = striker[0]
    now = after.batter(name)
    if now is None:  # the striker is gone from the bar: out
        off_bat, faced = 0, 1
    else:
        off_bat, faced = now[1] - striker[1], now[2] - striker[2]
    d_runs = after.runs - before.runs
    d_balls = after.legal_balls - before.legal_balls
    d_wkts = after.wickets - before.wickets
    if d_balls not in (0, 1) or d_runs < 0 or d_wkts < 0 or off_bat < 0 or off_bat > d_runs or faced not in (0, 1):
        return None  # more than one ball's change, or a misread: do not guess
    if d_balls == 1:
        kind = "legal"
    elif faced == 0 and d_runs >= 1:
        kind = "wide"
    elif faced == 1 and d_runs >= 1:
        kind = "no_ball"
    else:
        return None  # nothing changed yet
    boundary = {4: "FOUR", 6: "SIX"}.get(off_bat)
    core = (after.runs, after.wickets, after.legal_balls)
    settled = min(s.t for s in after_candidates if (s.runs, s.wickets, s.legal_balls) == core)
    return Outcome(before, after, name, before.bowler, d_runs, off_bat, d_runs - off_bat, d_wkts > 0, kind, boundary, settled)
