"""`cricsense run`: every pass, in order, on one video.

    calibrate (automatic when the video has no calibration) -> scoreboard -> players -> ball
    -> fuse (events + score engine) -> pose -> commentary -> voice (optional) -> render

A pass whose output already exists is skipped unless --fresh names it (or "all"), so a
failed run resumes where it stopped. Each step reports its time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cricsense.config import Config, video_id_from_path

log = logging.getLogger("cricsense.run")

STEPS = ("calibrate", "scoreboard", "players", "ball", "fuse", "pose", "commentary", "voice", "render")


@dataclass
class StepResult:
    name: str
    status: str  # "ran" | "skipped" | "failed"
    seconds: float
    note: str = ""


def _outputs(cfg: Config, video_id: str) -> dict[str, Path]:
    out = cfg.output_dir_for(video_id)
    return {
        "calibrate": out / "camera.json",
        "scoreboard": out / "scoreboard.json",
        "players": out / "players.json",
        "ball": out / "ball.json",
        "fuse": out / "events.json",
        "pose": out / "pose.json",
        "commentary": out / "commentary.json",
        "voice": out / "commentary.wav",
        "render": out / "overlay.mp4",
    }


def run_all(
    cfg: Config,
    video: Path,
    fresh: set[str] | None = None,
    voice: bool = False,
    stop_on_error: bool = True,
    only: set[str] | None = None,
) -> list[StepResult]:
    from cricsense.passes.ball import run_ball
    from cricsense.passes.calibrate import run_calibrate
    from cricsense.passes.commentary import run_commentary
    from cricsense.passes.fuse import run_fuse
    from cricsense.passes.players import run_players
    from cricsense.passes.pose import run_pose
    from cricsense.passes.render import run_render
    from cricsense.passes.scoreboard import run_scoreboard
    from cricsense.passes.voice import run_voice

    video_id = video_id_from_path(video)
    fresh = set(STEPS) if fresh and "all" in fresh else (fresh or set())
    outputs = _outputs(cfg, video_id)

    def calibrate() -> str:
        if cfg.calibration_path(video_id).is_file():  # never replace existing reference views
            run_calibrate(cfg, video, propagate_only=True)
            return "reference views from " + cfg.calibration_path(video_id).name
        run_calibrate(cfg, video, auto=True)
        return "automatic (no calibration existed)"

    steps: dict[str, Callable[[], str | None]] = {
        "calibrate": calibrate,
        "scoreboard": lambda: run_scoreboard(cfg, video) and None,
        "players": lambda: run_players(cfg, video) and None,
        "ball": lambda: run_ball(cfg, video) and None,
        "fuse": lambda: run_fuse(cfg, video_id) and None,
        "pose": lambda: run_pose(cfg, video) and None,
        "commentary": lambda: run_commentary(cfg, video_id)[0].engine,
        "voice": lambda: f"{len(run_voice(cfg, video_id)[1])} lines spoken",
        "render": lambda: run_render(cfg, video_id) and None,
    }
    results: list[StepResult] = []
    for name in STEPS:
        if only is not None and name not in only:
            continue
        if name == "voice" and not voice:
            continue
        if outputs[name].is_file() and name not in fresh:
            results.append(StepResult(name, "skipped", 0.0, f"{outputs[name].name} exists"))
            continue
        t0 = time.perf_counter()
        try:
            note = steps[name]() or ""
            results.append(StepResult(name, "ran", time.perf_counter() - t0, note))
        except Exception as e:  # report which step broke and why; later steps need its output
            results.append(StepResult(name, "failed", time.perf_counter() - t0, f"{type(e).__name__}: {e}"))
            log.exception("step %s failed", name)
            if stop_on_error:
                break
        log.info("%s: %s in %.0f s", name, results[-1].status, results[-1].seconds)
    return results
