"""CricSense in one command: give it a cricket video, get the analysis.

    python main.py data/my_clip.mp4                              # a clip (up to 20 min): everything
    python main.py "Cricket match/match.mp4"                     # a whole match: scores, then the overs
    python main.py "Cricket match/match.mp4" --overs 1:3 2:15    # ... and analyse those overs (innings:over)
    python main.py "Cricket match/match.mp4" --overs big         # ... every over of 15+ runs or a wicket
    python main.py "Cricket match/match.mp4" --overs all         # ... all of them (hours: overnight)
    python main.py "Cricket match/match.mp4" --start 6840 --end 7160   # ... or any stretch (seconds)

A whole match needs its team sheet, configs/matches/<video name>.yaml (the squads as the score bar
prints them); without one, main.py lists the names it sees on the bar and stops. Then it reads the
score bar over the whole video once (~45 min), prints both innings' final scores (compare them
with the scorecard), and lists every over. Each chosen over is cut into data/ and run through
every step - calibration, score bar, players, ball, events and score, pose, commentary, voice,
overlay - into outputs/<clip>/. Anything already done is reused, so it can be stopped and run
again. --no-voice skips the speech; a clip of another match needs --match <match video name>.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
CLIP_MAX_S = 20 * 60  # longer than this is treated as a whole match


def clip_prefix(stem: str) -> str:
    """A short clip name prefix from the match video's name ("replay-super-8s")."""
    return stem[:16].rstrip("-_")


def duration_s(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps


def names_on_bar(video: Path, layout_file: Path, every_s: int = 30) -> tuple[collections.Counter, collections.Counter]:
    """Names the score bar prints (batters, bowlers), sampled across the video - for a team sheet."""
    from cricsense.identity.scoreboard_ocr import ScoreboardLayout, ScoreboardReader

    reader = ScoreboardReader(ScoreboardLayout.from_yaml(layout_file))
    cap = cv2.VideoCapture(str(video))
    batters, bowlers = collections.Counter(), collections.Counter()
    for t in range(0, int(duration_s(video)), every_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, image = cap.read()
        if not ok or not reader.is_visible(image):
            continue
        for field, (x0, y0, x1, y1) in reader.layout.fields.items():
            if field.endswith("_name"):
                text = re.sub(r"[^A-Z ]", "", reader._tesseract(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), field).upper()).strip()
                if len(text) >= 3:
                    (bowlers if field == "bowler_name" else batters)[text] += 1
    cap.release()
    return batters, bowlers


def run_clip(cfg, video: Path, voice: bool) -> dict:
    """Every step on one clip; a short report of what came out."""
    from cricsense.passes.run import run_all

    results = run_all(cfg, video, voice=voice)
    for r in results:
        print(f"    {r.name:11s} {r.status:8s} {r.seconds:6.0f} s  {r.note}")
    out = cfg.output_dir_for(video.stem)
    report = {"clip": video.stem, "failed": [r.name for r in results if r.status == "failed"], "out": out}
    if (out / "match.json").is_file():
        states = json.loads((out / "match.json").read_text())["states"]
        report["balls"] = sum(1 for s in states if s["checks"])
        report["checks"] = (sum(sum(s["checks"].values()) for s in states), sum(len(s["checks"]) for s in states))
    return report


def cut(video: Path, start: float, end: float, name: str) -> Path:
    clip = ROOT / "data" / f"{name}.mp4"
    if not clip.is_file():
        print(f"  cutting {name} ({start:.0f}-{end:.0f} s) ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_clip.py"), "--video", str(video),
                        "--start", f"{start:.2f}", "--end", f"{end:.2f}", "--name", name],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    return clip


def record_match(clip_id: str, match: str) -> None:
    """A clip not cut by make_clip: say which match it is from (configs/clips.yaml)."""
    path = ROOT / "configs" / "clips.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    entry = raw.setdefault("clips", {}).setdefault(clip_id, {})
    if entry.get("source") or entry.get("match") == match:
        return
    entry["match"] = match
    header = "".join(line + "\n" for line in path.read_text().splitlines() if line.startswith("#"))
    path.write_text(header + yaml.safe_dump(raw, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="See the top of main.py and docs/RUNBOOK.md.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--overs", nargs="+", metavar="I:O", help="innings:over pairs (1:3 2:15), 'big' or 'all'")
    ap.add_argument("--start", type=float, help="seconds into the video")
    ap.add_argument("--end", type=float)
    ap.add_argument("--no-voice", action="store_true", help="no spoken commentary")
    ap.add_argument("--match", help="a clip's match (its video file name without .mp4), when it was not cut by make_clip")
    args = ap.parse_args(argv)

    from cricsense.config import load_config
    from cricsense.io import read_doc, setup_logging
    from cricsense.identity.match_sheet import layout_path, match_of, sheet_path

    setup_logging()
    cfg = load_config()
    video = args.video if args.video.is_absolute() else (Path.cwd() / args.video)
    if not video.is_file():
        ap.error(f"no such video: {args.video}")
    length = duration_s(video)
    whole_match = length > CLIP_MAX_S
    voice = not args.no_voice
    if args.match:
        record_match(video.stem, args.match)
    sheet_file = sheet_path(ROOT, video.stem)
    print(f"{video.name}: {length / 60:.0f} min -> {'a whole match' if whole_match else 'a clip'}; "
          f"match {match_of(ROOT, video.stem)}, team sheet {'found' if sheet_file.is_file() else 'MISSING'}")

    if not sheet_file.is_file():
        print(f"\nNo team sheet: {sheet_file.relative_to(ROOT)}. Names, teams and the score need it.")
        if whole_match:
            print("Reading the names the score bar prints (a few minutes) ...", flush=True)
            batters, bowlers = names_on_bar(video, layout_path(ROOT, video.stem))
            print("  batters seen:", ", ".join(f"{n} ({c})" for n, c in batters.most_common(30) if c >= 2))
            print("  bowlers seen:", ", ".join(f"{n} ({c})" for n, c in bowlers.most_common(20) if c >= 2))
        example = next((ROOT / "configs" / "matches").glob("*.yaml"))
        print(f"\nCopy {example.relative_to(ROOT)} to {sheet_file.relative_to(ROOT)}, set both squads exactly as the bar "
              f"prints them, the teams and the keepers, then run this again." + ("" if whole_match else
              " (Or, if this clip is from a match that has a sheet: --match <that match's video name>.)"))
        return 2

    if not whole_match:
        report = run_clip(cfg, video, voice)
        return finish([report])

    # a whole match: the score bar once over the whole video
    from cricsense.fuse.overs import innings_totals, overs
    from cricsense.identity.match_sheet import MatchSheet
    from cricsense.passes.scoreboard import run_scoreboard
    from cricsense.schema import ScoreboardDoc

    scan = cfg.output_dir_for(video.stem) / "scoreboard.json"
    if not scan.is_file():
        print("Reading the score bar over the whole match (~45 min for a T20; progress below) ...", flush=True)
        run_scoreboard(cfg, video)
    board = read_doc(scan, ScoreboardDoc)
    sheet = MatchSheet.from_yaml(sheet_file)
    print("\nWhole match, as read from the score bar (check against the official scorecard):")
    for k, (side, runs, wkts, balls, seen) in enumerate(innings_totals(board, sheet)):
        print(f"  innings {k + 1}: {side or '?'} {runs}-{wkts} after {balls // 6}.{balls % 6} overs  ({seen}/121 ball counts read)")
    table = overs(board, sheet)

    chosen = []
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None or args.end <= args.start:
            ap.error("--start and --end together, end after start")
        chosen = [(f"{clip_prefix(video.stem)}_{args.start:.0f}_{args.end:.0f}", args.start, args.end)]
    elif args.overs:
        want = set(args.overs)
        for o in table:
            pick = "all" in want or f"{o.innings}:{o.over}" in want
            pick |= "big" in want and ((o.runs or 0) >= 15 or (o.wickets or 0) > 0)
            if pick:
                chosen.append((f"{clip_prefix(video.stem)}_i{o.innings}_o{o.over}", o.start_s, o.end_s))
        missing = [w for w in want - {"all", "big"} if not any(c[0].endswith(f"_i{w.replace(':', '_o')}") for c in chosen)]
        if missing:
            print(f"  not found on the score bar (or not in the form innings:over): {', '.join(sorted(missing))}")

    if not chosen:
        print("\nOvers (innings:over  bowler  score  runs/wkts):")
        for o in table:
            score = (f"{o.before[0]}-{o.before[1]}" if o.before else "?") + (f" > {o.after[0]}-{o.after[1]}" if o.after else "")
            print(f"  {o.innings}:{o.over:<3} {(o.bowler or '?').title():<11} {score:>13}  "
                  f"{'' if o.runs is None else o.runs}/{'' if o.wickets is None else o.wickets}")
        print(f'\nNext: python main.py "{args.video}" --overs 1:3 2:15   (or --overs big, --overs all)')
        return 0

    est = sum(e - s for _, s, e in chosen) / 60
    print(f"\nAnalysing {len(chosen)} clip(s), {est:.0f} min of video (roughly 2-5 min of work per minute) ...")
    reports = []
    for name, start, end in chosen:
        print(f"\n== {name}", flush=True)
        try:
            reports.append(run_clip(cfg, cut(video, start, end, name), voice))
        except Exception as e:  # one over failing must not stop the rest
            print(f"    FAILED: {e}")
            reports.append({"clip": name, "failed": [str(e)[:80]], "out": cfg.output_dir_for(name)})
    return finish(reports)


def finish(reports: list[dict]) -> int:
    print("\nResults:")
    for r in reports:
        out = Path(r["out"])
        rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
        checks = f"{r['checks'][0]}/{r['checks'][1]} checks agree with the score bar" if r.get("checks") else "no score checks"
        status = f"FAILED at {', '.join(r['failed'])}" if r["failed"] else "ok"
        print(f"  {r['clip']}: {status}; {r.get('balls', 0)} balls, {checks}")
        if (out / "overlay.mp4").is_file():
            print(f"      video: {rel}/overlay.mp4   check the pitch: {rel}/calibration_check.png")
    print("\nWatch each overlay.mp4; if the green lines in calibration_check.png are not on the painted creases, "
          "see docs/RUNBOOK.md 'When something goes wrong'.")
    return 1 if any(r["failed"] for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
