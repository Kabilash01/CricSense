"""`cricsense publish`: a video's outputs into the database and object storage.

One row per ball in `deliveries` joins what the passes found about it - the ball pass (length,
line, speeds), the events (outcome, from the score bar), the commentary facts (the shot, read
by the VLM) and pose (footwork, swing, batting hand) - beside the events, the score engine's
states and the commentary lines. The rendered files go to object storage with checksums.
Publishing again replaces the video's rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from cricsense.config import Config
from cricsense.store import db
from cricsense.store.objects import open_store

FILES = {  # artifact kind -> file in outputs/<video_id>/
    "overlay": "overlay.mp4",
    "pitch_map": "pitch_map.png",
    "wagon_wheel": "wagon_wheel.png",
    "commentary_audio": "commentary.wav",
    "events_json": "events.json",
    "ball_json": "ball.json",
    "match_json": "match.json",
    "commentary_json": "commentary.json",
}
DEFAULT_DB = "postgresql+psycopg://cricsense@localhost:54329/cricsense"


def store_config(cfg: Config) -> dict:
    path = cfg.root / "configs" / "store.yaml"
    raw = yaml.safe_load(path.read_text()) if path.is_file() else {}
    raw.setdefault("database_url", os.environ.get("CRICSENSE_DB_URL", DEFAULT_DB))
    if os.environ.get("CRICSENSE_DB_URL"):
        raw["database_url"] = os.environ["CRICSENSE_DB_URL"]
    raw.setdefault("storage", {"backend": "local", "root": "outputs/store"})
    return raw


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def delivery_rows(out_dir: Path) -> list[dict]:
    ball = _load(out_dir / "ball.json") or {"deliveries": []}
    events = (_load(out_dir / "events.json") or {"events": []})["events"]
    facts = {f["delivery_id"]: f for f in (_load(out_dir / "commentary.json") or {"facts": []})["facts"]}
    pose = {b["delivery_id"]: b for b in (_load(out_dir / "pose.json") or {"batters": []})["batters"]}
    by: dict[int, dict[str, dict]] = {}
    for e in events:
        did = e["payload"].get("delivery_id")
        if did is not None:
            by.setdefault(did, {})[e["type"]] = e["payload"]
    rows = []
    for d in ball["deliveries"]:
        did = d["delivery_id"]
        ev = by.get(did, {})
        rel, runs = ev.get("RELEASE", {}), ev.get("RUNS", {})
        f, p = facts.get(did, {}), pose.get(did, {})
        rows.append({
            "delivery_id": did,
            "over": rel.get("over"),
            "bowler": rel.get("bowler"),
            "striker": rel.get("striker"),
            "release_t": d["release_t"],
            "bounce_t": d.get("bounce_t"),
            "length_m": d.get("length_m"),
            "line_m": d.get("line_m"),
            "length_class": d.get("length_class"),
            "bounce_confirmed": bool(d.get("bounce_confirmed")),
            "radar_kph": d.get("radar_speed_kph"),
            "estimate_kph": d.get("speed_estimate_kph"),
            "kind": runs.get("kind"),
            "runs": runs.get("runs"),
            "off_bat": runs.get("off_bat"),
            "extras": runs.get("extras"),
            "wicket": "WICKET" in ev if runs else None,
            "boundary": ev.get("BOUNDARY", {}).get("kind"),
            "score_after": runs.get("score"),
            "shot": f.get("shot"),
            "footwork": p.get("footwork"),
            "swing": p.get("swing"),
            "batting_hand": p.get("batting_hand"),
        })
    return rows


def run_publish(cfg: Config, video_id: str, engine=None) -> dict:
    out_dir = cfg.output_dir_for(video_id)
    ball = _load(out_dir / "ball.json")
    if ball is None:
        raise FileNotFoundError(f"{out_dir / 'ball.json'} not found: run the pipeline first")
    scfg = store_config(cfg)
    engine = engine or db.connect(scfg["database_url"])
    store = open_store(scfg["storage"], cfg.root)
    clips = yaml.safe_load((cfg.root / "configs" / "clips.yaml").read_text()) if (cfg.root / "configs" / "clips.yaml").is_file() else {}
    src = ((clips or {}).get("clips") or {}).get(video_id)
    sheet = yaml.safe_load((cfg.root / "configs" / "match.yaml").read_text()) if (cfg.root / "configs" / "match.yaml").is_file() else {}
    counts = {}
    with engine.begin() as conn:
        match_id = None
        if sheet and sheet.get("teams"):
            name = sheet.get("name") or " v ".join(sheet["teams"])
            match_id = db.upsert_match(conn, name, sheet["teams"])
        v = ball["video"]
        vid = db.replace_video(
            conn, video_id, match_id=match_id, fps=v["fps"], n_frames=v["n_frames"], width=v["width"], height=v["height"],
            source=f"{src['source']} @ {src['start_s']} s" if src else None,
        )
        rows = delivery_rows(out_dir)
        if rows:
            conn.execute(db.insert(db.deliveries), [{"video_id": vid, **r} for r in rows])
        counts["deliveries"] = len(rows)
        events = (_load(out_dir / "events.json") or {"events": []})["events"]
        if events:
            conn.execute(db.insert(db.events), [
                {"video_id": vid, "frame_id": e["frame_id"], "t": e["t"], "type": e["type"], "payload": e["payload"],
                 "confidence": e["confidence"], "source": e["source"]} for e in events
            ])
        counts["events"] = len(events)
        states = (_load(out_dir / "match.json") or {"states": []})["states"]
        if states:
            conn.execute(db.insert(db.match_states), [
                {"video_id": vid, "delivery_id": s["delivery_id"], "t": s["t"], "over": s["over"], "runs": s["runs"],
                 "wickets": s["wickets"], "striker": s["striker"], "non_striker": s["non_striker"], "bowler": s["bowler"],
                 "batters": s["batters"], "bowlers": s["bowlers"], "source": s["source"], "checks": s["checks"]} for s in states
            ])
        counts["match_states"] = len(states)
        lines = (_load(out_dir / "commentary.json") or {"lines": []})["lines"]
        if lines:
            conn.execute(db.insert(db.commentary), [
                {"video_id": vid, "delivery_id": c["delivery_id"], "t": c["t"], "kind": c["kind"], "text": c["text"], "source": c["source"]}
                for c in lines
            ])
        counts["commentary_lines"] = len(lines)
        stored = []
        for kind, name in FILES.items():
            path = out_dir / name
            if path.is_file():
                s = store.put(path, f"{video_id}/{name}")
                stored.append({"video_id": vid, "kind": kind, "uri": s.uri, "sha256": s.sha256, "bytes": s.bytes})
        if stored:
            conn.execute(db.insert(db.artifacts), stored)
        counts["artifacts"] = len(stored)
    return {"video_id": video_id, "db_id": vid, **counts}
