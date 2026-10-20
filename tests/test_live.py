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
