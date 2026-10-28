"""Live mode's score worker: each ball's outcome as soon as the bar settles, and the radar."""

import time

from cricsense.config import load_config
from cricsense.live.runner import LiveState, ScoreWorker
from cricsense.schema import BatterLine, ScoreboardReading


def bar(t, runs, balls, kohli, axar, strike="KOHLI", speed=None):
    return ScoreboardReading(
        frame_id=int(t * 25), t=t,
        batters=[BatterLine(name="KOHLI", runs=kohli[0], balls=kohli[1], on_strike=strike == "KOHLI"),
                 BatterLine(name="AXAR", runs=axar[0], balls=axar[1], on_strike=strike == "AXAR")],
        runs=runs, wickets=3, legal_balls=balls, bowler="SHAMSI", bowler_runs=None, bowler_legal_balls=None, speed_kph=speed,
    )


def test_balls_and_radar_are_emitted_once_each_as_the_bar_settles():
    state = LiveState()
    w = ScoreWorker(load_config(), state, time.perf_counter(), "final_0051_6min")  # a clip of the 2024 final
    for t in (0.0, 0.5):
        w.update(bar(t, 85, 69, (40, 30), (30, 24), strike="AXAR"))
    for t in (20.0, 20.5, 21.0):  # a six by Axar, radar shown
        w.update(bar(t, 91, 70, (40, 30), (36, 25), strike="AXAR", speed=89.2))
    for t in (40.0, 40.5):  # a wide: no ball faced, one run
        w.update(bar(t, 92, 70, (40, 30), (36, 25), strike="AXAR"))
    balls = [e.payload for e in state.events if e.kind == "BALL"]
    assert [(b["over"], b["runs"], b["kind"], b["boundary"]) for b in balls] == [("11.4", 6, "legal", "SIX"), ("11.5", 1, "wide", None)]
    assert [e.payload["radar_kph"] for e in state.events if e.kind == "SPEED"] == [89.2]
    assert state.score == "92-3 (11.4)"


def test_live_plus_card_ties_each_radar_reading_and_bounce_to_its_ball():
    """The timings of Bumrah's over 1.1-1.2: 1.1's radar came 4.3 s after its result, a replay of
    1.2 gave a second, unconfirmed bounce."""
    from cricsense.live.plus import ball_card
    from cricsense.live.runner import LiveEvent, LiveState

    def ball(t, over, runs=0, boundary=None):
        return LiveEvent("BALL", 0, t, t, {"over": over, "runs": runs, "kind": "legal", "boundary": boundary, "wicket": False, "striker": "DE KOCK", "bowler": "BUMRAH"})

    def bounce(t, length, confirmed=True, est=None):
        return LiveEvent("BOUNCE", 0, t, t + 15, {"length_m": length, "line_m": -0.1, "length_class": "good", "confirmed": confirmed, "speed_estimate_kph": est})

    s = LiveState(batters=[("DE KOCK", 1, 3, True), ("HENDRICKS", 4, 4, False)], bowler="BUMRAH", next_ball="1.2")
    s.events = [bounce(12.2, 7.35, est=142), ball(15.4, "1.1"), LiveEvent("SPEED", 0, 19.7, 19.8, {"radar_kph": 139.3}),
                bounce(44.5, 4.95, est=144), LiveEvent("SPEED", 0, 48.0, 48.1, {"radar_kph": 142.4}), bounce(48.0, 0.95, confirmed=False),
                ball(58.6, "1.2", runs=4, boundary="FOUR")]
    first = ball_card(s, 21.0)  # 1.1's result is in, and its radar after it: still 1.1's
    assert first["title"] == "1.1  Bumrah → De Kock" and first["speed"] == "139.3 km/h  radar" and first["outcome"] == "DOT BALL"
    assert ball_card(s, 42.0)["title"] == "1.2  Bumrah → De Kock" and ball_card(s, 42.0)["speed"] is None  # about to be bowled
    under_way = ball_card(s, 46.0)  # its bounce is known, no radar yet: our estimate
    assert under_way["speed"] == "≈ 144 km/h  estimate" and under_way["outcome"] is None
    assert ball_card(s, 49.0)["speed"] == "142.4 km/h  radar"  # the radar wins once on the bar
    done = ball_card(s, 59.0)
    assert done["title"] == "1.2  Bumrah → De Kock" and done["outcome"] == "FOUR"
    assert done["delivery"]["length_m"] == 4.95  # the delivery, not the replay's unconfirmed bounce
