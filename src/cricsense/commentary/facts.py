"""The facts of each ball, from the pipeline outputs: the only thing commentary may use.

Templates and the LLM both speak from these fact sheets and nothing else, so every claim in
the commentary traces back to a measurement or a score-bar reading - and the fact sheet says
which (speed_source "radar" or "estimate", bounce_confirmed, shot_source "vlm").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cricsense.identity.match_sheet import MatchSheet
from cricsense.schema import BallDoc, EventsDoc, EventType, MatchDoc, PoseDoc

MILESTONES = (50, 100)


@dataclass
class BallFacts:
    delivery_id: int
    over: str | None  # "12.1"
    bowler: str | None  # display names: "Nortje"
    striker: str | None
    non_striker: str | None
    batting_team: str | None  # "India"
    release_frame: int
    contact_frame: int | None
    outcome_frame: int | None
    next_release_frame: int | None
    length_class: str | None  # yorker | full | good | short
    length_m: float | None
    line_m: float | None  # + to the right as seen on screen
    bounce_confirmed: bool
    speed_kph: float | None
    speed_source: str | None  # "radar" | "estimate"
    bat_contact: bool
    runs: int | None  # None: the outcome is not known
    off_bat: int | None
    extras: int | None
    kind: str | None  # legal | wide | no_ball
    boundary: str | None  # FOUR | SIX
    wicket: bool
    score: str | None  # "91-3" after the ball
    striker_score: str | None  # "36 (25)" after the ball
    milestone: int | None  # the striker reached 50 / 100 with this ball
    over_complete: bool
    overs_completed: int | None  # after the last ball of an over: how many overs are done (11.6 -> 12)
    over_runs: int | None  # runs in the over just completed
    bowler_figures: str | None  # "0-18 (2.0)" after the ball
    footwork: str | None = None  # measured from pose: front foot | back foot | in the crease
    swing: str | None = None  # measured from pose: played | left
    shot: str | None = None  # from the VLM, when it is confident
    direction: str | None = None
    shot_source: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name(s: str | None) -> str | None:
    return None if s is None else " ".join(w if w.lower() in ("de", "van", "du") else w.title() for w in s.split()).replace("De ", "de ", 1)


def build_facts(
    ball: BallDoc, events: EventsDoc, match: MatchDoc | None, sheet: MatchSheet | None, pose: PoseDoc | None = None
) -> list[BallFacts]:
    by: dict[int, dict[EventType, object]] = {}
    for e in events.events:
        if "delivery_id" in e.payload:
            by.setdefault(e.payload["delivery_id"], {})[e.type] = e
    states = {s.delivery_id: s for s in match.states} if match else {}
    pose_by = {b.delivery_id: b for b in pose.batters} if pose else None
    out: list[BallFacts] = []
    runs_at: dict[int, int] = {}  # legal balls -> the score when that count was first reached
    prev_state = None
    for i, d in enumerate(ball.deliveries):
        evs = by.get(d.delivery_id, {})
        rel = evs.get(EventType.RELEASE)
        p = rel.payload if rel else {}
        runs_ev = evs.get(EventType.RUNS)
        q = runs_ev.payload if runs_ev else {}
        st = states.get(d.delivery_id)
        known = st is not None and st.source != "no_evidence" and runs_ev is not None
        striker = p.get("striker")
        team = sheet.team_of(striker) if sheet and striker else None
        milestone = None
        if known and striker in st.batters and prev_state is not None and striker in prev_state.batters:
            before, after = prev_state.batters[striker][0], st.batters[striker][0]
            milestone = next((m for m in MILESTONES if before < m <= after), None)
        over_complete = bool(known and q.get("kind") == "legal" and st.legal_balls % 6 == 0)
        if known and not runs_at:  # the score before the first ball we know
            runs_at[st.legal_balls - (1 if q.get("kind") == "legal" else 0)] = st.runs - q.get("runs", 0)
        if known:
            runs_at.setdefault(st.legal_balls, st.runs)
        over_runs = st.runs - runs_at[st.legal_balls - 6] if over_complete and st.legal_balls - 6 in runs_at else None
        if p.get("radar_speed_kph"):
            speed, source = p["radar_speed_kph"], "radar"
        elif p.get("speed_estimate_kph"):
            speed, source = p["speed_estimate_kph"], "estimate"
        else:
            speed, source = None, None
        contact = evs.get(EventType.CONTACT)
        bowler_fig = None
        if known and st.bowler in st.bowlers:
            r, b, w = st.bowlers[st.bowler]
            bowler_fig = f"{w}-{r} ({b // 6}.{b % 6})"
        out.append(
            BallFacts(
                delivery_id=d.delivery_id,
                over=p.get("over"),
                bowler=_name(p.get("bowler")),
                striker=_name(striker),
                non_striker=_name(st.non_striker) if known and st.non_striker else None,
                batting_team=team.display if team else None,
                release_frame=d.release_frame,
                contact_frame=contact.frame_id if contact else None,
                outcome_frame=runs_ev.frame_id if runs_ev else None,
                next_release_frame=ball.deliveries[i + 1].release_frame if i + 1 < len(ball.deliveries) else None,
                length_class=d.length_class.value if d.length_class else None,
                length_m=None if d.length_m is None else round(d.length_m, 1),
                line_m=None if d.line_m is None else round(d.line_m, 2),
                bounce_confirmed=d.bounce_confirmed,
                speed_kph=speed,
                speed_source=source,
                bat_contact=contact is not None,
                runs=q.get("runs") if known else None,
                off_bat=q.get("off_bat") if known else None,
                extras=q.get("extras") if known else None,
                kind=q.get("kind") if known else None,
                boundary=evs[EventType.BOUNDARY].payload["kind"] if EventType.BOUNDARY in evs else None,
                wicket=EventType.WICKET in evs,
                score=f"{st.runs}-{st.wickets}" if known else None,
                striker_score=f"{st.batters[striker][0]} ({st.batters[striker][1]})" if known and striker in st.batters else None,
                milestone=milestone,
                over_complete=over_complete,
                overs_completed=st.legal_balls // 6 if over_complete else None,
                over_runs=over_runs,
                bowler_figures=bowler_fig,
                footwork=batter.footwork if (batter := (pose_by or {}).get(d.delivery_id)) else None,
                swing=batter.swing if batter else None,
            )
        )
        if known:
            prev_state = st
    return out
