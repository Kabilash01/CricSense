"""Single entry point: `cricsense <subcommand>`."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Subcommand -> the build phase that implements it. Removed from here as each phase lands.
_PENDING_PHASE = {
    "live": 7,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cricsense",
        description="Cricket video analysis in independent passes joined by JSON files.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(
        name: str,
        help_text: str,
        handler: Callable[[argparse.Namespace], int] | None = None,
        *,
        video: bool = False,
        video_id: bool = False,
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        if video:
            p.add_argument("--video", required=True, type=Path, help="path to the source video")
        if video_id:
            p.add_argument("--video-id", required=True, help="id of an already-processed video (file stem)")
        p.set_defaults(handler=handler or _not_implemented(name))
        return p

    cal = add(
        "calibrate",
        "find the pitch (--auto, no clicks) or click 4 crease corners, then track the pitch mapping through every frame",
        _calibrate,
        video=True,
    )
    cal.add_argument("--frame", type=int, default=0, help="reference frame to start the picker on (default 0)")
    cal.add_argument("--view", default="main", help="name for this reference view (default: main)")
    cal.add_argument(
        "--points",
        help="skip the picker: 'name=x,y;name=x,y' with the 4 crease corners plus optional check points",
    )
    cal.add_argument("--force", action="store_true", help="replace an existing view with the same name")
    mode = cal.add_mutually_exclusive_group()
    mode.add_argument("--no-propagate", action="store_true", help="save the reference view only")
    mode.add_argument("--propagate-only", action="store_true", help="recompute camera.json from the saved views")
    mode.add_argument("--auto", action="store_true", help="find the reference views from the painted creases, no clicks")
    cal.add_argument("--cameras-from", type=Path, metavar="CALIBRATION_YAML",
                     help="with --auto: where this match's cameras stand, from another clip's calibration (helps short clips)")
    mode.add_argument(
        "--import-views",
        type=Path,
        metavar="CALIBRATION_YAML",
        help="reuse the reference views of another clip from the same match, then track",
    )

    add(
        "scoreboard",
        "CPU pass: read the broadcast score bar (batters, striker, score, bowler, radar speed) -> scoreboard.json",
        _scoreboard,
        video=True,
    )
    pl = add("players", "GPU pass: player tracks, stable IDs, roles, identities -> players.json", _players, video=True)
    pl.add_argument("--tracker", type=Path, help="tracker config (default: configs/players.yaml tracker.config)")
    pl.add_argument("--out-name", default="players.json", help="output file name inside outputs/<video_id>/")
    pl.add_argument("--device", help="override configs/default.yaml device, e.g. cpu or cuda:0")
    bl = add("ball", "GPU pass: deliveries - release, ball track, bounce, line and length, speeds -> ball.json", _ball, video=True)
    bl.add_argument("--weights", type=Path, help="ball model (default: configs/ball.yaml detector.weights)")
    bl.add_argument("--fresh", action="store_true", help="re-run the detector instead of using cached candidates")
    bl.add_argument("--device", help="override configs/default.yaml device, e.g. cpu or cuda:0")
    po = add("pose", "the striker's body on each delivery: footwork and swing -> pose.json", _pose, video=True)
    po.add_argument("--device", help="override configs/default.yaml device, e.g. cpu or cuda:0")
    fu = add("fuse", "event timeline: release, bounce, bat contact, runs, boundaries, wickets, wides -> events.json", _fuse, video_id=True)
    fu.add_argument("--device", help="override configs/default.yaml device for the bat detector")
    co = add("commentary", "commentary lines per ball from the facts (LLM checked against them, or templates) -> commentary.json", _commentary, video_id=True)
    co.add_argument("--engine", choices=["auto", "llm", "template"], help="default: configs/commentary.yaml engine")
    co.add_argument("--no-vlm", action="store_true", help="skip shot type / direction from the vision model")
    add("voice", "speak the commentary (local Kokoro TTS, or Gemini) -> commentary.wav + voice.json", _voice, video_id=True)
    rn = add("render", "draw overlay.mp4 and pitch_map.png from the pass outputs", _render, video_id=True)
    rn.add_argument("--start", type=float, help="first second to render (default: the start)")
    rn.add_argument("--end", type=float, help="last second to render (default: the end)")
    rn.add_argument("--out-name", default="overlay.mp4", help="output file name inside outputs/<video_id>/")
    rn.add_argument("--no-voice", action="store_true", help="leave the commentary audio and captions out")
    ru = add("run", "every pass on one video, in order: calibrate (automatic if needed) ... render", _run, video=True)
    ru.add_argument("--fresh", nargs="*", default=[], metavar="STEP", help="re-run these steps even if their output exists (or: all)")
    ru.add_argument("--voice", action="store_true", help="also speak the commentary (local Kokoro by default)")
    pu = add("publish", "a video's outputs into the database and object storage (configs/store.yaml)", _publish, video_id=True)
    sv = add("serve", "the HTTP API: upload videos, read deliveries, events, scorecard, commentary, files", _serve)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    add("live", "demo: players + ball + fuse in one loop (degraded, optional)", video=True)
    return parser


def _calibrate(args: argparse.Namespace) -> int:
    from cricsense.calibration.homography import CalibrationError
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.calibrate import parse_points, run_calibrate, summarize

    setup_logging()
    try:
        result = run_calibrate(
            load_config(),
            args.video,
            frame_id=args.frame,
            view_name=args.view,
            points=parse_points(args.points) if args.points else None,
            force=args.force,
            propagate=not args.no_propagate,
            propagate_only=args.propagate_only,
            import_views=args.import_views,
            auto=args.auto,
            cameras_from=args.cameras_from,
        )
    except CalibrationError as e:
        print(f"cricsense calibrate: {e}", file=sys.stderr)
        return 1
    print("\n".join(summarize(result)))
    return 0


def _scoreboard(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.scoreboard import run_scoreboard, summarize

    setup_logging()
    doc, path = run_scoreboard(load_config(), args.video)
    print("\n".join(summarize(doc, path)))
    return 0


def _players(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.players import run_players, summarize

    setup_logging()
    cfg = load_config()
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    print("\n".join(summarize(run_players(cfg, args.video, args.tracker, args.out_name))))
    return 0


def _ball(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.ball import run_ball, summarize

    setup_logging()
    cfg = load_config()
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    doc, path = run_ball(cfg, args.video, args.weights, args.fresh)
    print("\n".join(summarize(doc, path)))
    return 0


def _pose(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.pose import run_pose, summarize

    setup_logging()
    doc, path = run_pose(load_config(), args.video, args.device)
    print("\n".join(summarize(doc, path)))
    return 0


def _fuse(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.fuse import run_fuse, summarize

    setup_logging()
    cfg = load_config()
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    doc, path = run_fuse(cfg, args.video_id)
    print("\n".join(summarize(doc, path)))
    return 0


def _commentary(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.commentary import run_commentary, summarize

    setup_logging()
    doc, path = run_commentary(load_config(), args.video_id, args.engine, False if args.no_vlm else None)
    print("\n".join(summarize(doc, path)))
    return 0


def _voice(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.voice import run_voice

    setup_logging()
    wav, lines, problems = run_voice(load_config(), args.video_id)
    for line in lines:
        print(f"{line['start_s']:7.1f}-{line['end_s']:6.1f}s  {line['text']}")
    for p in problems:
        print("  ! " + p)
    print(f"commentary audio: {wav} ({len(lines)} lines spoken)")
    return 0 if lines else 1


def _run(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.run import run_all

    setup_logging()
    results = run_all(load_config(), args.video, set(args.fresh), args.voice)
    for r in results:
        print(f"{r.name:11s} {r.status:8s} {r.seconds:7.0f} s  {r.note}")
    return 1 if any(r.status == "failed" for r in results) else 0


def _publish(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.publish import run_publish

    setup_logging()
    print(run_publish(load_config(), args.video_id))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from cricsense.api.app import create_app
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.publish import store_config
    from cricsense.store import db

    setup_logging()
    cfg = load_config()
    uvicorn.run(create_app(cfg, db.connect(store_config(cfg)["database_url"])), host=args.host, port=args.port)
    return 0


def _render(args: argparse.Namespace) -> int:
    from cricsense.config import load_config
    from cricsense.io import setup_logging
    from cricsense.passes.render import run_render

    setup_logging()
    video, pitch_map, n = run_render(load_config(), args.video_id, args.start, args.end, args.out_name, not args.no_voice)
    print(f"overlay: {video} ({n} frames)\npitch map: {pitch_map}")
    return 0


def _not_implemented(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        print(f"cricsense {name}: not implemented yet (build phase {_PENDING_PHASE[name]})", file=sys.stderr)
        return 2

    return handler


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
