"""Regression check: re-run the pipeline on the hand-verified clips and compare ball by ball.

    python scripts/check.py                 # re-run scoreboard, ball and fuse, then compare
    python scripts/check.py --no-rerun      # compare the outputs already there
    python scripts/check.py --save          # record the current outputs as the expected answers

configs/golden.yaml holds, for each verified clip, every ball's over, score, wickets, striker,
bowler and radar speed, checked by a person against the broadcast. A change to the code passes
if every clip still gives exactly those balls and every score-engine check agrees with the
score bar. Only --save after looking at the new results yourself: the file is the answer key.
Exit status 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "configs" / "golden.yaml"
CLIPS = ("final_0051_6min", "st_022000", "st_bumrah_o2")  # verified by hand against the broadcast


def observed(clip: str) -> dict:
    out = ROOT / "outputs" / clip
    match = json.loads((out / "match.json").read_text())
    ball = json.loads((out / "ball.json").read_text())
    radar = {d["delivery_id"]: d["radar_speed_kph"] for d in ball["deliveries"]}
    balls, passed, total = [], 0, 0
    for s in match["states"]:
        passed += sum(s["checks"].values())
        total += len(s["checks"])
        if s["source"] == "engine" and s["checks"]:  # a ball the score bar confirmed
            balls.append({
                "over": s["over"], "score": f"{s['runs']}-{s['wickets']}", "striker": s["striker"],
                "bowler": s["bowler"], "radar_kph": radar.get(s["delivery_id"]),
            })
    return {"deliveries": len(ball["deliveries"]), "balls": balls, "checks": [passed, total]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-rerun", action="store_true", help="compare existing outputs only")
    ap.add_argument("--save", action="store_true", help="write the current outputs as the expected answers")
    ap.add_argument("--clips", nargs="+", default=list(CLIPS))
    args = ap.parse_args(argv)

    if not args.no_rerun and not args.save:
        for clip in args.clips:
            print(f"re-running scoreboard, ball, fuse on {clip} ...", flush=True)
            subprocess.run([sys.executable, "-m", "cricsense", "run", "--video", f"data/{clip}.mp4", "--fresh", "scoreboard", "ball", "fuse"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    got = {clip: observed(clip) for clip in args.clips}
    if args.save:
        GOLDEN.write_text("# Expected answers for scripts/check.py - verified by hand against the broadcast.\n"
                          + yaml.safe_dump(got, sort_keys=False, allow_unicode=True))
        print(f"saved {GOLDEN}")
        return 0

    want = yaml.safe_load(GOLDEN.read_text())
    ok = True
    for clip in args.clips:
        w, g = want[clip], got[clip]
        problems = []
        if g["deliveries"] < w["deliveries"]:
            problems.append(f"deliveries {g['deliveries']}, expected {w['deliveries']}")
        if g["checks"][0] != g["checks"][1]:
            problems.append(f"score-engine checks {g['checks'][0]}/{g['checks'][1]} - some disagree with the score bar")
        gb = {b["over"]: b for b in g["balls"]}
        for b in w["balls"]:
            h = gb.get(b["over"])
            if h is None:
                problems.append(f"ball {b['over']} missing (expected {b['score']})")
            elif h != b:
                diff = ", ".join(f"{k} {h[k]} (expected {b[k]})" for k in b if h[k] != b[k])
                problems.append(f"ball {b['over']}: {diff}")
        extra = sorted(set(gb) - {b["over"] for b in w["balls"]})
        if extra:
            problems.append(f"balls not in the answer key: {', '.join(extra)} (look, then --save if right)")
        status = "PASS" if not problems else "FAIL"
        ok &= not problems
        print(f"{status}  {clip}: {len(g['balls'])} balls, checks {g['checks'][0]}/{g['checks'][1]}, deliveries {g['deliveries']}")
        for p in problems:
            print(f"      - {p}")
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
