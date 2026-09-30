"""Evaluation set for wides: the match's own wides, found by the score bar, plus legal balls.

    python scripts/wide_eval.py --source "Cricket match/replay-final-sa-vs-ind.mp4" --legal 15

1. outputs/<source>/ball_outcomes.json (from the full-broadcast score-bar scan) lists every
   ball whose outcome the bar showed cleanly: wides and legal balls with their times;
2. a clip is cut around each wide and around a random sample of legal balls
   (scripts/make_clip.py, so configs/clips.yaml records where each came from);
3. the vision passes run on each clip - automatic calibration, score bar, players, ball, fuse,
   pose - with no clicks;
4. for the delivery in each clip: where the ball pitched and where it passed the batter's
   crease (lateral, metres from the middle-stump line), whether the bat touched it, the
   batting hand; written to outputs/wide_eval.json for the wide rule to be fitted and scored.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def cricsense(*args: str) -> None:
    subprocess.run([PY, "-m", "cricsense", *args], check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def crossing(clip: str, delivery) -> dict:
    """Lateral position where the ball passes the batter's crease, from the track after the bounce."""
    from cricsense.io import read_doc
    from cricsense.schema import BallDoc, CameraDoc

    out = ROOT / "outputs" / clip
    ball = read_doc(out / "ball.json", BallDoc)
    cam = {c.frame_id: np.array(c.image_to_pitch) for c in read_doc(out / "camera.json", CameraDoc).frames if c.image_to_pitch}
    pts = []
    for f in ball.frames:
        if f.delivery_id != delivery.delivery_id or f.frame_id not in cam:
            continue
        q = cam[f.frame_id] @ np.array([f.uv[0], f.uv[1], 1.0])
        x, y = q[:2] / q[2]
        # after the bounce, until the batter: the ground point of an airborne ball runs a little
        # past the ball along the sight line, but its sideways position barely changes (~5%)
        if delivery.bounce_t is None or f.t >= delivery.bounce_t:
            pts.append((x, y))
    near = [y for x, y in pts if -0.5 <= x <= 2.5]
    return {"y_at_crease": float(np.median(near)) if near else None, "n_near": len(near)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--legal", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from cricsense.io import read_doc
    from cricsense.schema import BallDoc, EventsDoc, EventType, PoseDoc

    outcomes = json.loads((ROOT / "outputs" / args.source.stem / "ball_outcomes.json").read_text())
    wides = [o for o in outcomes if o["kind"] == "wide"]
    legal = [o for o in outcomes if o["kind"] == "legal" and o["batting"] == wides[0]["batting"]]
    random.Random(args.seed).shuffle(legal)
    chosen = wides + legal[: args.legal]
    results = []
    for k, o in enumerate(chosen):
        clip = f"we_{o['kind']}_{o['over'].replace('.', '_')}"
        start = max(0.0, o["before_t"] - 3.0)
        dur = o["after_t"] - start + 1.0
        if not (ROOT / "data" / f"{clip}.mp4").is_file():
            subprocess.run([PY, "scripts/make_clip.py", "--video", str(args.source), "--start", f"{start:.2f}",
                            "--duration", f"{dur:.2f}", "--name", clip], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        video = f"data/{clip}.mp4"
        row = {"clip": clip, **o}
        try:
            out = ROOT / "outputs" / clip
            if not (out / "camera.json").is_file():
                cricsense("calibrate", "--video", video, "--auto")
            for step, target in (("scoreboard", "scoreboard.json"), ("players", "players.json"), ("ball", "ball.json")):
                if not (out / target).is_file():
                    cricsense(step, "--video", video)
            if not (out / "events.json").is_file():
                cricsense("fuse", "--video-id", clip)
            if not (out / "pose.json").is_file():
                cricsense("pose", "--video", video)
            ball = read_doc(out / "ball.json", BallDoc)
            events = read_doc(out / "events.json", EventsDoc)
            pose = read_doc(out / "pose.json", PoseDoc)
            if not ball.deliveries:
                row["status"] = "no delivery found"
            else:
                d = ball.deliveries[-1]  # the ball just before the bar changed
                contact = any(e.type == EventType.CONTACT and e.payload.get("delivery_id") == d.delivery_id and "bat_detector" in e.source for e in events.events)
                hand = next((b.batting_hand for b in pose.batters if b.delivery_id == d.delivery_id), None)
                row.update({"status": "ok", "line_at_bounce": d.line_m, "length_m": d.length_m, "bat_contact_seen": contact,
                            "batting_hand": hand, "swing": next((b.swing for b in pose.batters if b.delivery_id == d.delivery_id), None),
                            **crossing(clip, d)})
        except subprocess.CalledProcessError as e:
            row["status"] = f"pipeline failed at {' '.join(e.cmd[3:5])}"
        results.append(row)
        print(f"[{k + 1}/{len(chosen)}] {clip}: {row.get('status')}  "
              + (f"y at crease {row.get('y_at_crease')}, line at bounce {row.get('line_at_bounce')}, hand {row.get('batting_hand')}, swing {row.get('swing')}, bat {row.get('bat_contact_seen')}" if row.get("status") == "ok" else ""))
    (ROOT / "outputs" / "wide_eval.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
