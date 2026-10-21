"""Cricket score engine: the authoritative match state, ball by ball.

A ball-and-over state machine: the score, wickets, legal balls, who is on strike, and each
batter's and bowler's figures, updated from each delivery's outcome by the laws of the game -
strike changes on odd runs and at the end of an over, wides don't count as balls faced, byes
and leg-byes are not the bowler's runs, a new over needs a new bowler.

The broadcast score bar validates it (it is evidence, not the authority): after every ball the
engine's state is compared with the bar's, check by check. Where the engine had no evidence
(a delivery the ball pass missed, a ball with no clean score-bar change) it cannot know what
happened, so it re-syncs from the bar and says so - it never silently copies the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cricsense.fuse.outcome import Outcome, ScoreState
from cricsense.schema import BallState

BALLS_PER_OVER = 6


@dataclass
class Figures:
    runs: int = 0
    balls: int = 0
    wickets: int = 0


@dataclass
class EngineState:
    runs: int
    wickets: int
    legal_balls: int
    striker: str | None
    non_striker: str | None
    bowler: str | None
    batters: dict[str, Figures] = field(default_factory=dict)
    bowlers: dict[str, Figures] = field(default_factory=dict)

    @property
    def over(self) -> str:
        return f"{self.legal_balls // BALLS_PER_OVER}.{self.legal_balls % BALLS_PER_OVER}"


class ScoreEngine:
    def __init__(self, start: ScoreState) -> None:
        self.s = _from_bar(start)
        self.awaiting_bowler = False

    # ---- the laws -------------------------------------------------------------------------

    def apply(self, o: Outcome) -> dict[str, bool]:
        """Update the state with one ball; return the checks of the engine's expectations."""
        s = self.s
        checks: dict[str, bool] = {}
        if self.awaiting_bowler:  # a new over: whoever bowls it now
            checks["new bowler at the over"] = o.bowler != s.bowler
            s.bowler = o.bowler
            self.awaiting_bowler = False
        else:
            checks["same bowler within the over"] = o.bowler == s.bowler
        checks["striker as predicted"] = o.striker == s.striker
        if s.striker != o.striker and o.striker in (s.non_striker,):
            s.striker, s.non_striker = s.non_striker, s.striker  # follow the evidence, flagged above

        striker = s.batters.setdefault(o.striker, Figures())
        bowler = s.bowlers.setdefault(o.bowler, Figures())
        s.runs += o.runs
        striker.runs += o.off_bat
        if o.kind == "wide":
            bowler.runs += o.runs  # all wide runs go against the bowler
            ran = o.runs - 1
        elif o.kind == "no_ball":
            striker.balls += 1
            bowler.runs += o.off_bat + 1
            ran = o.runs - 1
        else:  # legal: extras are byes / leg-byes, not the bowler's
            striker.balls += 1
            bowler.runs += o.off_bat
            ran = o.runs
            s.legal_balls += 1
            bowler.balls += 1
        if o.wicket:
            s.wickets += 1
            bowler.wickets += 1  # run-outs are not the bowler's; the bar tells (checked below)
            s.striker = None  # the incoming batter is named by the bar
        if ran % 2 == 1:
            s.striker, s.non_striker = s.non_striker, s.striker
        if o.kind == "legal" and s.legal_balls % BALLS_PER_OVER == 0:
            s.striker, s.non_striker = s.non_striker, s.striker
            self.awaiting_bowler = True
        return checks

    # ---- validation against the score bar -------------------------------------------------

    def compare(self, bar: ScoreState) -> dict[str, bool]:
        s = self.s
        checks = {
            "score": s.runs == bar.runs,
            "wickets": s.wickets == bar.wickets,
            "overs": s.legal_balls == bar.legal_balls,
        }
        for name, runs, balls, _ in bar.batters:
            mine = s.batters.get(name)
            if mine is not None:
                checks[f"{name.title()} runs/balls"] = (mine.runs, mine.balls) == (runs, balls)
        bar_striker = bar.striker[0] if bar.striker else None
        if s.striker is not None and bar_striker is not None:
            checks["on strike"] = s.striker == bar_striker
        b = s.bowlers.get(bar.bowler)
        if b is not None and bar.bowler_runs is not None and bar.bowler_legal_balls is not None:
            checks[f"{bar.bowler.title()} figures"] = (b.runs, b.balls) == (bar.bowler_runs, bar.bowler_legal_balls)
        return checks

    def resync(self, bar: ScoreState) -> None:
        """Take the bar's state where the engine had no evidence (flagged by the caller)."""
        bowlers = self.s.bowlers
        self.s = _from_bar(bar)
        for name, fig in bowlers.items():  # the bar only shows the current bowler
            self.s.bowlers.setdefault(name, fig)
        self.awaiting_bowler = False

    def fill_names(self, bar: ScoreState) -> None:
        """An incoming batter, or the strike after a wicket, comes from the bar (flagged by the caller)."""
        s = self.s
        names = [b[0] for b in bar.batters]
        for name, runs, balls, _ in bar.batters:
            s.batters.setdefault(name, Figures(runs, balls))
        if s.striker is None or s.striker not in names or s.non_striker not in names:
            on = bar.striker[0] if bar.striker else None
            s.striker = on
            s.non_striker = next((n for n in names if n != on), None)


def _from_bar(bar: ScoreState) -> EngineState:
    on = bar.striker[0] if bar.striker else None
    names = [b[0] for b in bar.batters]
    s = EngineState(
        runs=bar.runs,
        wickets=bar.wickets,
        legal_balls=bar.legal_balls,
        striker=on,
        non_striker=next((n for n in names if n != on), None),
        bowler=bar.bowler,
        batters={name: Figures(runs, balls) for name, runs, balls, _ in bar.batters},
    )
    if bar.bowler_runs is not None and bar.bowler_legal_balls is not None:
        s.bowlers[bar.bowler] = Figures(bar.bowler_runs, bar.bowler_legal_balls, bar.bowler_wickets or 0)
    return s


CORE = ("score", "wickets", "overs")


def score_innings(
    deliveries: list[tuple[int, int, float, Outcome | None]], states: list[ScoreState], fps: float
) -> list[BallState]:
    """Run the engine over a clip. deliveries: (delivery_id, release_frame, release_t, outcome)."""
    first_t = deliveries[0][2] if deliveries else 0.0
    start = next((s for s in reversed(states) if s.t <= first_t), states[0] if states else None)
    if start is None:
        return []
    engine = ScoreEngine(start)
    out: list[BallState] = []
    for did, release_frame, _, o in deliveries:
        notes: list[str] = []
        if o is None:
            out.append(_snapshot(engine, did, release_frame, fps, "no_evidence", {}, ["no clean score-bar change for this ball"]))
            continue
        if o.off_bat is None:  # the score alone: not enough for the laws (runs off the bat; maybe who faced)
            out.append(_snapshot(engine, did, release_frame, fps, "no_evidence", {}, [f"score only: {o.kind}, {o.runs} run(s); batters not shown"]))
            continue
        source = "engine"
        before = engine.compare(o.before)
        if not all(before[k] for k in CORE):
            engine.resync(o.before)
            source = "resync"
            notes.append(f"engine had no evidence for a ball before this one; took the bar's {o.before.runs}-{o.before.wickets} ({o.before.legal_balls // 6}.{o.before.legal_balls % 6})")
        elif engine.s.striker is None:
            engine.fill_names(o.before)
            notes.append("incoming batter named from the score bar")
        if o.bowler not in engine.s.bowlers and o.before.bowler == o.bowler and o.before.bowler_legal_balls is not None:
            # a bowler who bowled before the clip began: the engine can only know the earlier figures from the bar
            engine.s.bowlers[o.bowler] = Figures(o.before.bowler_runs or 0, o.before.bowler_legal_balls, o.before.bowler_wickets or 0)
            notes.append(f"{o.bowler.title()}'s figures before this clip from the score bar")
        checks = engine.apply(o)
        if o.wicket:
            engine.fill_names(o.after)
            notes.append("wicket: incoming batter and strike from the score bar")
        checks.update(engine.compare(o.after))
        frame = int(round((o.settled_t if o.settled_t is not None else o.after.t) * fps))
        out.append(_snapshot(engine, did, frame, fps, source, checks, notes))
    return out


def _snapshot(engine: ScoreEngine, did: int, frame: int, fps: float, source: str, checks: dict, notes: list) -> BallState:
    s = engine.s
    return BallState(
        delivery_id=did,
        frame_id=frame,
        t=frame / fps,
        over=s.over,
        runs=s.runs,
        wickets=s.wickets,
        legal_balls=s.legal_balls,
        striker=s.striker,
        non_striker=s.non_striker,
        bowler=s.bowler,
        batters={k: (v.runs, v.balls) for k, v in s.batters.items()},
        bowlers={k: (v.runs, v.balls, v.wickets) for k, v in s.bowlers.items()},
        source=source,
        checks=checks,
        notes=notes,
    )
