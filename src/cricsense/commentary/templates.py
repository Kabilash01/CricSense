"""Template commentary: deterministic, from the fact sheet only; the fallback for the LLM.

One line as the ball is played (who bowled what, and what the batter did with it), and one
after it when there is more to say (a boundary, a wicket, a milestone, the end of an over).
The wording varies ball to ball but is fixed per delivery, so a run is reproducible.
"""

from __future__ import annotations

from cricsense.commentary.facts import BallFacts


def _pick(options: list[str], f: BallFacts, salt: int = 0) -> str:
    return options[(f.delivery_id * 7 + salt * 3) % len(options)]


LENGTH = {
    "yorker": ["right up in the blockhole", "a yorker", "full and fast at the toes"],
    "full": ["pitched up", "full", "a fuller one"],
    "good": ["on a good length", "back of a length", "on that good length"],
    "short": ["short", "banged in short", "a short one"],
}


HEDGED = {"yorker": "that looks a yorker", "full": "looks full", "good": "looks a good length", "short": "looks short"}


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def delivery_phrase(f: BallFacts) -> str:
    who = f.bowler or "The bowler"
    parts = []
    if f.length_class:
        # an unconfirmed bounce (the ball was not seen after pitching) is said with a hedge
        parts.append(_pick(LENGTH[f.length_class], f) if f.bounce_confirmed else HEDGED[f.length_class])
    if f.line_m is not None and abs(f.line_m) < 0.12:
        parts.append("on the stumps")
    text = f"{who}, " + ", ".join(parts) if parts else f"{who} in"
    if f.speed_kph and f.speed_source == "radar":
        text += f", {f.speed_kph:g} kph"
    elif f.speed_kph and f.speed_source == "estimate":
        text += f", around {int(round(f.speed_kph / 5.0) * 5)} kph"
    return text


def outcome_phrase(f: BallFacts) -> str | None:
    s = f.striker or "the batter"
    if f.runs is None:
        return None
    if f.wicket:
        return _pick([f"and that's OUT! {s} has to go", f"OUT! {s} is gone", f"and {s} is out!"], f)
    if f.kind == "wide":
        extra = f", {f.runs - 1} more run" + ("s" if f.runs - 1 != 1 else "") + " to go with it" if f.runs > 1 else ""
        return f"and that's called wide{extra}"
    if f.kind == "extra":
        return f"and that's an extra, {f.runs} added"
    if f.kind == "no_ball":
        return f"and it's a no-ball! {f.runs} added, free hit to come"
    if f.boundary == "SIX":
        return _pick([f"and {s} sends it into the stands, SIX!", f"SIX! {s} launches it", f"{s} clears the rope, that's SIX"], f)
    if f.boundary == "FOUR":
        return _pick([f"and {s} finds the rope, FOUR!", f"FOUR! {s} beats the field", f"{s} times it away, four runs"], f)
    if f.off_bat is not None and f.off_bat == 0 and f.extras:
        return f"{f.extras} extra run" + ("s" if f.extras > 1 else "") + " as they scamper through"
    if f.runs == 0:
        if f.bat_contact:
            return _pick([f"{s} defends it, no run", f"solid from {s}, dot ball", f"{s} gets behind it, no run"], f)
        return _pick(["no run", "dot ball", "nothing off it"], f)
    if f.runs == 1:
        return _pick([f"{s} works it away for a single", f"{s} pushes it into the gap for one", f"just the one for {s}"], f)
    if f.runs == 2:
        return _pick([f"{s} places it and they come back for two", f"two runs for {s}", f"good running, they get two"], f)
    return f"{f.runs} runs for {s}"


def ball_line(f: BallFacts) -> str:
    out = outcome_phrase(f)
    text = delivery_phrase(f)
    if out is None:
        return text + "."
    sep = " - " if out[0].isupper() else ", "
    line = f"{text}{sep}{out}"
    return line[0].upper() + line[1:] + ("" if line.endswith(("!", ".")) else ".")


def after_line(f: BallFacts) -> str | None:
    """Score, milestone or end of the over, when there is something worth saying."""
    team = f.batting_team or "The batting side"
    runs, _, wkts = (f.score or "").partition("-")
    score = f"{team} {runs} for {wkts}" if f.score else None
    bits = []
    if f.milestone and f.striker:
        bits.append(f"That brings up {f.milestone} for {f.striker}, {f.striker_score}.")
    if f.over_complete:
        over_no = int(f.over.split(".")[0]) + 1 if f.over else None  # balls 11.1-11.6 are the 12th over
        head = f"End of the {ordinal(over_no)} over" if over_no else "End of the over"
        if f.over_runs is not None:
            head += f", {f.over_runs} from it"
        bits.append(f"{head}. {score}." if score else head + ".")
        if f.bowler and f.bowler_figures:
            bits.append(f"{f.bowler} {f.bowler_figures.split(' ')[0]} so far.")
    elif (f.boundary or f.wicket) and score:
        bits.append(f"{score}." if not f.wicket else f"{score}, and a new batter to come.")
        if f.boundary and f.striker_score:
            bits.append(f"{f.striker} on {f.striker_score}.")
    return " ".join(bits) if bits else None
