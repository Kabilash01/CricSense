"""`cricsense scoreboard`: read the broadcast score bar through a video -> scoreboard.json."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from cricsense.config import Config, video_id_from_path
from cricsense.identity.match_sheet import MatchSheet
from cricsense.identity.scoreboard_ocr import ScoreboardLayout, ScoreboardReader
from cricsense.io import VideoReader, write_doc
from cricsense.schema import RunInfo, ScoreboardDoc

log = logging.getLogger("cricsense.scoreboard")


def run_scoreboard(
    cfg: Config,
    video: Path,
    layout_path: Path | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    out_name: str = "scoreboard.json",
) -> tuple[ScoreboardDoc, Path]:
    layout = ScoreboardLayout.from_yaml(layout_path or cfg.root / "configs" / "scoreboard.yaml")
    sheet_path = cfg.root / "configs" / "match.yaml"
    squad = MatchSheet.from_yaml(sheet_path).all_names if sheet_path.is_file() else ()
    reader = ScoreboardReader(layout, squad)
    video_id = video_id_from_path(video)
    readings = []
    t0 = time.perf_counter()
    with VideoReader(video, resize=cfg.video.resize) as probe:
        fps = probe.fps
    start = int(round((start_s or 0) * fps))
    end = None if end_s is None else int(round(end_s * fps))
    with VideoReader(video, skip=layout.sample_every, start=start, end=end, resize=cfg.video.resize) as frames:
        meta = frames.meta(video_id)
        sampled = 0
        for frame in frames:
            sampled += 1
            reading = reader.read(frame.image, frame.frame_id, frame.t)
            if reading is not None:
                readings.append(reading)
            if sampled % 2000 == 0:
                log.info("at %.0f s of the video: %d readings, %d OCR calls, %.0f s elapsed", frame.t, len(readings), reader.ocr_calls, time.perf_counter() - t0)
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=sampled, device="cpu")
    doc = ScoreboardDoc(video=meta, run=run, layout=layout.name, readings=readings)
    path = write_doc(doc, cfg.output_dir_for(video_id) / out_name)
    log.info("wrote %s (%d readings, %d OCR calls)", path, len(readings), reader.ocr_calls)
    return doc, path


def summarize(doc: ScoreboardDoc, path: Path) -> list[str]:
    run = doc.run
    speeds = sorted({r.speed_kph for r in doc.readings if r.speed_kph is not None})
    lines = [
        f"score bar visible on {len(doc.readings)}/{run.frames_processed} sampled frames "
        f"({run.wall_time_s:.1f}s)",
        f"radar speeds shown: {', '.join(f'{s:g} km/h' for s in speeds) or 'none'}",
    ]
    if doc.readings:
        last = doc.readings[-1]
        striker = next((b.name for b in last.batters if b.on_strike), None)
        lines.append(
            f"last reading: {last.runs}-{last.wickets} after {last.legal_balls} balls, "
            f"striker {striker}, bowler {last.bowler}"
        )
    lines.append(f"scoreboard: {path}")
    return lines
