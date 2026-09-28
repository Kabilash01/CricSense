"""Per-shot report of a players.json: IDs, dominant roles, names, and role-consistency checks.

    python scripts/players_report.py outputs/final_0051_6min/players.json

A shot is a run of processed frames between camera cuts. In a shot from the main camera
the expected cast is one striker, one non-striker, one bowler, at most one keeper and one
umpire; anything else is a fielder. The summary counts shots that break that.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

from cricsense.io import read_doc
from cricsense.schema import PlayersDoc

UNIQUE = ("striker", "non_striker", "bowler", "keeper")


def shots(doc: PlayersDoc, gap_frames: int = 5) -> list[tuple[int, int, list[int]]]:
    by_sid: dict[int, list] = defaultdict(list)
    for f in doc.frames:
        by_sid[f.sid].append(f.frame_id)
    spans: list[list] = []
    for sid, frames in sorted(by_sid.items(), key=lambda kv: kv[1][0]):
        a, b = frames[0], frames[-1]
        if spans and a <= spans[-1][1] + gap_frames:
            spans[-1][1] = max(spans[-1][1], b)
            spans[-1][2].append(sid)
        else:
            spans.append([a, b, [sid]])
    return [tuple(s) for s in spans]


def main(path: str) -> int:
    doc = read_doc(Path(path), PlayersDoc)
    fps = doc.video.fps
    frames_of: dict[int, list] = defaultdict(list)
    for f in doc.frames:
        frames_of[f.sid].append(f)
    problems = Counter()
    all_shots = shots(doc)
    for a, b, sids in all_shots:
        parts, dominant = [], Counter()
        for sid in sids:
            fs = frames_of[sid]
            role, n = Counter(f.role.value for f in fs).most_common(1)[0]
            dominant[role] += 1
            xs = [f.pitch_xy[0] for f in fs if f.pitch_xy]
            where = f"x{sum(xs) / len(xs):.0f}" if xs else "x?"
            name = f"/{fs[0].identity}" if fs[0].identity else ""
            parts.append(f"{role}{name}@{where}({100 * n / len(fs):.0f}%)")
        issues = [f"2x {r}" for r in UNIQUE if dominant[r] > 1]
        issues += [f"no {r}" for r in ("striker", "non_striker", "bowler") if dominant[r] == 0]
        if dominant["umpire"] > 2:
            issues.append(f"{dominant['umpire']}x umpire")
        for i in issues:
            problems[i.split(" ", 1)[1]] += 1
        flag = f"   <-- {', '.join(issues)}" if issues else ""
        print(f"{a / fps:6.1f}-{b / fps:6.1f}s  {len(sids)} IDs: {', '.join(parts)}{flag}")
    named = sum(1 for _, _, sids in all_shots if any(frames_of[s][0].identity for s in sids))
    n_ids = len({f.sid for f in doc.frames})
    print(f"\n{len(all_shots)} shots, {n_ids} IDs, {named} shots with names; role problems: {dict(problems) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
