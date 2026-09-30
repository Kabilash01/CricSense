"""`cricsense fuse`: one event timeline from the players, ball and score-bar passes.

Per delivery (ball.json):
  RELEASE   when the ball left the hand; actors: bowler, striker, non-striker, keeper
  BOUNCE    where it pitched: length (m + class) and line
  CONTACT   bat on ball: a bat box touching the ball near the striker after the bounce
            (bat detector), or runs credited to the striker (score bar)
  RUNS      the ball's outcome from the score bar: runs off the bat, extras, legal/wide/no-ball
  BOUNDARY  FOUR / SIX;  WICKET;  WIDE;  NO_BALL
Every event carries its source and a confidence, so the overlay can tell a measurement from
an inference. Runs only the bat detector on the GPU, on a few frames per delivery.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

from cricsense.ball.candidates import detect_candidates
from cricsense.config import Config
from cricsense.fuse.contact import find_contact
from cricsense.fuse.outcome import outcome, trusted_states
from cricsense.fuse.score_engine import score_innings
from cricsense.identity.match_sheet import MatchSheet
from cricsense.io import read_doc, write_doc
from cricsense.schema import (
    BallDoc,
    Event,
    MatchDoc,
    EventsDoc,
    EventType,
    PlayersDoc,
    Role,
    RunInfo,
    ScoreboardDoc,
)

log = logging.getLogger("cricsense.fuse")


def run_fuse(cfg: Config, video_id: str, bat_weights: Path | None = None) -> tuple[EventsDoc, Path]:
    out_dir = cfg.output_dir_for(video_id)
    ball = read_doc(out_dir / "ball.json", BallDoc)
    players = read_doc(out_dir / "players.json", PlayersDoc) if (out_dir / "players.json").is_file() else None
    board = read_doc(out_dir / "scoreboard.json", ScoreboardDoc) if (out_dir / "scoreboard.json").is_file() else None
    sheet_path = cfg.root / "configs" / "match.yaml"
    sheet = MatchSheet.from_yaml(sheet_path) if sheet_path.is_file() else None
    squad = set(sheet.all_names) if sheet else set()
    team_of = {n: t.code for t in sheet.teams for n in t.squad} if sheet else None
    fps, n_frames = ball.video.fps, ball.video.n_frames
    size = (ball.video.width, ball.video.height)
    t0 = time.perf_counter()

    states = trusted_states(board.readings, squad, team_of) if board else []
    by_frame_role: dict[int, dict[Role, tuple[int, tuple]]] = defaultdict(dict)
    if players:
        for f in players.frames:
            by_frame_role[f.frame_id][f.role] = (f.sid, f.box)

    def actors_at(frame: int) -> dict[Role, int]:
        for df in (0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8):
            roles = by_frame_role.get(frame + df)
            if roles:
                return {r: sid for r, (sid, _) in roles.items() if r in (Role.BOWLER, Role.STRIKER, Role.NON_STRIKER, Role.KEEPER)}
        return {}

    def striker_box(frame: int):
        for df in range(0, 12):
            for f in (frame + df, frame - df):
                hit = by_frame_role.get(f, {}).get(Role.STRIKER)
                if hit:
                    return hit[1]
        return None

    # bat detections only where contact can happen
    windows: set[int] = set()
    for d in ball.deliveries:
        centre = int(round(d.bounce_t * fps)) if d.bounce_t is not None else d.release_frame + int(0.7 * fps)
        windows |= {f for f in range(centre - 3, centre + 15) if 0 <= f < n_frames}
    bat_weights = bat_weights or cfg.root / "models" / "bat_best.pt"
    video = next(cfg.paths.video_dir.glob(f"{video_id}.*"))
    bats = detect_candidates(video, windows, bat_weights, out_dir, 1920, 0.3, cfg.device, tag="bat_candidates") if windows else {}
    balls = detect_candidates(video, windows, cfg.root / cfg.ball["detector"]["weights"], out_dir, 1920, 0.1, cfg.device) if windows else {}

    events: list[Event] = []

    def add(frame: int, kind: EventType, conf: float, source: str, payload: dict, actors: dict | None = None) -> None:
        frame = max(0, min(frame, n_frames - 1))
        events.append(
            Event(frame_id=frame, t=frame / fps, type=kind, actors=actors or {}, payload=payload, confidence=conf, source=source)
        )

    releases = [d.release_t for d in ball.deliveries]
    per_ball = []
    for i, d in enumerate(ball.deliveries):
        nxt = releases[i + 1] if i + 1 < len(releases) else None
        res = outcome(states, d.release_t, nxt) if states else None
        per_ball.append((d.delivery_id, d.release_frame, d.release_t, res))
        actors = actors_at(d.release_frame)
        names = {}
        if res:
            names = {"bowler": res.bowler, "striker": res.striker}
            names["over"] = f"{res.before.legal_balls // 6}.{res.before.legal_balls % 6 + 1}"
        speed = {"radar_speed_kph": d.radar_speed_kph} if d.radar_speed_kph else {}
        if d.speed_estimate_kph and d.bounce_confirmed:  # shown as "≈", always labelled an estimate
            speed["speed_estimate_kph"] = round(d.speed_estimate_kph)
        add(d.release_frame, EventType.RELEASE, 0.8, "ball.delivery", {"delivery_id": d.delivery_id, **names, **speed}, actors)
        bounce_frame = None
        if d.bounce_t is not None:
            bounce_frame = int(round(d.bounce_t * fps))
            add(
                bounce_frame,
                EventType.BOUNCE,
                0.9 if d.bounce_confirmed else 0.6,
                "ball.bounce",
                {
                    "delivery_id": d.delivery_id,
                    "length_m": round(d.length_m, 2),
                    "length_class": d.length_class.value if d.length_class else None,
                    "line_m": round(d.line_m, 2),
                    "confirmed": d.bounce_confirmed,
                },
                actors,
            )
        centre = bounce_frame if bounce_frame is not None else d.release_frame + int(0.7 * fps)
        contact = find_contact(bats, balls, centre, striker_box(centre), size, cfg.camera.overlay_mask)
        credited = res is not None and res.off_bat > 0
        if contact is not None:
            add(
                contact.frame_id,
                EventType.CONTACT,
                0.95 if credited else 0.75,
                "bat_detector" + ("+scoreboard" if credited else ""),
                {"delivery_id": d.delivery_id, "bat_conf": contact.bat_conf, "ball_conf": contact.ball_conf},
                actors,
            )
        elif credited:
            add(centre + 5, EventType.CONTACT, 0.8, "scoreboard", {"delivery_id": d.delivery_id, "evidence": "runs off the bat"}, actors)
        if res is None:
            continue
        at = int(round((res.settled_t if res.settled_t is not None else res.after.t) * fps))
        score = f"{res.after.runs}-{res.after.wickets}"
        add(
            at,
            EventType.RUNS,
            0.9,
            "scoreboard",
            {
                "delivery_id": d.delivery_id,
                **names,
                "runs": res.runs,
                "off_bat": res.off_bat,
                "extras": res.extras,
                "kind": res.kind,
                "score": score,
            },
            actors,
        )
        if res.boundary:
            add(at, EventType.BOUNDARY, 0.95 if res.boundary == "SIX" else 0.85, "scoreboard", {"delivery_id": d.delivery_id, "kind": res.boundary, **names}, actors)
        if res.wicket:
            add(at, EventType.WICKET, 0.9, "scoreboard", {"delivery_id": d.delivery_id, **names, "score": score}, actors)
        if res.kind == "wide":
            add(at, EventType.WIDE, 0.85, "scoreboard", {"delivery_id": d.delivery_id, "runs": res.runs}, actors)
        elif res.kind == "no_ball":
            add(at, EventType.NO_BALL, 0.85, "scoreboard", {"delivery_id": d.delivery_id, "runs": res.runs}, actors)

    events.sort(key=lambda e: (e.frame_id, e.type.value))
    run = RunInfo(wall_time_s=time.perf_counter() - t0, frames_processed=len(windows), device=cfg.device)
    doc = EventsDoc(video=ball.video, run=run, events=events)
    path = write_doc(doc, out_dir / "events.json")
    log.info("wrote %s: %d events", path, len(events))
    match = MatchDoc(video=ball.video, run=run, states=score_innings(per_ball, states, fps))
    write_doc(match, out_dir / "match.json")
    return doc, path


def summarize(doc: EventsDoc, path: Path) -> list[str]:
    lines = []
    by_delivery: dict[int, list[Event]] = defaultdict(list)
    for e in doc.events:
        by_delivery[e.payload.get("delivery_id", -1)].append(e)
    for did, evs in sorted(by_delivery.items()):
        rel = next((e for e in evs if e.type == EventType.RELEASE), None)
        if rel and "bowler" in rel.payload:
            head = f"{rel.payload['over']} {rel.payload['bowler']} to {rel.payload['striker']}"
        else:
            head = f"delivery {did} at {evs[0].t:.1f} s"
        parts = []
        for e in evs:
            p = e.payload
            if e.type == EventType.BOUNCE:
                parts.append(f"pitched {p['length_m']} m ({p['length_class']}){'' if p['confirmed'] else '?'}")
            elif e.type == EventType.CONTACT:
                parts.append(f"bat contact [{e.source}]")
            elif e.type == EventType.RUNS:
                parts.append(f"{p['runs']} run{'s' if p['runs'] != 1 else ''}" + (f" ({p['extras']} extras)" if p["extras"] else "") + f" -> {p['score']}")
            elif e.type in (EventType.BOUNDARY,):
                parts.append(p["kind"])
            elif e.type in (EventType.WICKET, EventType.WIDE, EventType.NO_BALL):
                parts.append(e.type.value)
        if rel and rel.payload.get("radar_speed_kph"):
            parts.append(f"{rel.payload['radar_speed_kph']:g} km/h (radar)")
        if rel and rel.payload.get("speed_estimate_kph"):
            parts.append(f"≈{rel.payload['speed_estimate_kph']} km/h (estimate)")
        if not any(e.type == EventType.RUNS for e in evs):
            parts.append("no clean score-bar update before the next ball / end of clip")
        lines.append(f"{head}: " + ", ".join(parts))
    lines.append(f"events: {path}")
    match_path = path.parent / "match.json"
    if match_path.is_file():
        lines += summarize_match(read_doc(match_path, MatchDoc), match_path)
    return lines


def _figures(f: tuple[int, int, int] | None) -> str:
    if f is None:
        return ""
    runs, balls, wkts = f
    return f"{wkts}-{runs} ({balls // 6}.{balls % 6})"


def summarize_match(doc: MatchDoc, path: Path) -> list[str]:
    lines = ["", "score engine (engine state after each ball; checks against the score bar):"]
    passed = total = 0
    for st in doc.states:
        failed = [k for k, ok in st.checks.items() if not ok]
        passed += sum(st.checks.values())
        total += len(st.checks)
        tag = {"engine": "", "resync": "  [RESYNC]", "no_evidence": "  [no evidence]"}[st.source]
        on = f"{st.striker.title()}*" if st.striker else "?"
        lines.append(
            f"  {st.over:>5}  {st.runs}-{st.wickets}  {on} {st.non_striker.title() if st.non_striker else '?'}  "
            f"{(st.bowler or '?').title()} {_figures(st.bowlers.get(st.bowler))}"
            f"  checks {sum(st.checks.values())}/{len(st.checks)}" + (f"  FAILED: {', '.join(failed)}" if failed else "") + tag
        )
        for n in st.notes:
            lines.append(f"         note: {n}")
    lines.append(f"  all checks: {passed}/{total} agree with the score bar   match state: {path}")
    return lines
