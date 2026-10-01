"""Per-stage speed of the pipeline on one video, for the report.

    python scripts/benchmark.py --video data/final_0051_6min.mp4
    python scripts/benchmark.py --video data/final_0051_6min.mp4 --steps players ball pose

Runs each step fresh (cricsense.passes.run), in order, and records its wall time, the frames
it processed (from the pass's own RunInfo where it has one), frames per second, and the
real-time factor: seconds of video per second of compute (> 1 is faster than real time).
Writes outputs/<video>/benchmark.json and benchmark.md (a table for the report), with the
hardware and library versions. Run it on an otherwise idle machine: the numbers are only as
good as that.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_OF = {"calibrate": "camera.json", "scoreboard": "scoreboard.json", "players": "players.json", "ball": "ball.json",
          "fuse": "events.json", "pose": "pose.json", "commentary": "commentary.json", "voice": "voice.json", "render": None}
WHAT = {
    "calibrate": "find the pitch's views, track the camera every frame",
    "scoreboard": "read the score bar every 0.5 s",
    "players": "detect, track and name the players (calibrated frames)",
    "ball": "ball detection (2 models), delivery, bounce, speed",
    "fuse": "events, outcomes, score engine (+ bat detector near contact)",
    "pose": "striker pose around each delivery",
    "commentary": "facts, LLM lines with fact check, shot reading",
    "voice": "speech for each line (local Kokoro)",
    "render": "overlay video with audio, pitch map, wagon wheel",
}


def hardware() -> dict:
    info = {"python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
        info["gpu"] = out.strip().splitlines()[0] if out.strip() else None
    except OSError:
        info["gpu"] = None
    try:
        import torch
        import ultralytics

        info["torch"], info["ultralytics"] = torch.__version__, ultralytics.__version__
    except ImportError:
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--steps", nargs="*", default=["calibrate", "scoreboard", "players", "ball", "fuse", "pose", "commentary", "voice", "render"])
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from cricsense.config import load_config, video_id_from_path
    from cricsense.io import VideoReader
    from cricsense.passes.run import run_all

    cfg = load_config(ROOT)
    vid = video_id_from_path(args.video)
    with VideoReader(args.video) as r:
        n_frames, fps = r.n_frames, r.fps
    duration = n_frames / fps
    out_dir = cfg.output_dir_for(vid)
    rows = []
    for step in args.steps:
        t0 = time.perf_counter()
        results = run_all(cfg, args.video, fresh={step}, voice=step == "voice", only={step})
        wall = time.perf_counter() - t0
        res = next((r for r in results if r.name == step), None)
        processed = None
        doc = DOC_OF.get(step)
        if doc and (out_dir / doc).is_file():
            run = json.loads((out_dir / doc).read_text()).get("run") or {}
            processed = run.get("frames_processed")
        rows.append({
            "step": step, "status": res.status if res else "not run", "wall_s": round(wall, 1),
            "frames_processed": processed, "fps": round(processed / wall, 1) if processed else None,
            "realtime_factor": round(duration / wall, 2), "what": WHAT.get(step, ""),
        })
        print(f"{step:11s} {rows[-1]['status']:7s} {wall:7.1f} s  x{rows[-1]['realtime_factor']} real time  {processed or ''}")
    total = sum(r["wall_s"] for r in rows)
    report = {"video": vid, "duration_s": round(duration, 1), "frames": n_frames, "hardware": hardware(), "steps": rows,
              "total_s": round(total, 1), "total_realtime_factor": round(duration / total, 2)}
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=1))
    lines = [f"Benchmark: {vid} ({duration:.0f} s, {n_frames} frames); {report['hardware'].get('cpu')}, {report['hardware'].get('gpu')}", "",
             "| step | what | time (s) | frames | fps | x real time |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['step']} | {r['what']} | {r['wall_s']} | {r['frames_processed'] or ''} | {r['fps'] or ''} | {r['realtime_factor']} |")
    lines.append(f"| **all** | | **{report['total_s']}** | | | **{report['total_realtime_factor']}** |")
    (out_dir / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
