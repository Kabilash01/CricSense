"""Single entry point: `cricsense <subcommand>`."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# Subcommand -> the build phase that implements it. Removed from here as each phase lands.
_PENDING_PHASE = {
    "ball": 3,
    "fuse": 4,
    "render": 5,
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
        "click 4 crease corners on a reference frame, then track the pitch mapping through every frame",
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
    add("ball", "GPU pass: ball trajectory, bounce, line and length -> ball.json", video=True)
    add("fuse", "CPU pass: contact, boundary and wicket events -> events.json", video_id=True)
    add("render", "draw overlay.mp4 and pitch_map.png from the pass outputs", video_id=True)
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
