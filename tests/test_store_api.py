"""Publishing into the database and object storage, and the HTTP API over it (SQLite, no models)."""

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from cricsense.api.app import create_app  # noqa: E402
from cricsense.config import load_config  # noqa: E402
from cricsense.passes.publish import delivery_rows, run_publish  # noqa: E402
from cricsense.store import db  # noqa: E402

VIDEO = {"video_id": "clip", "fps": 25.0, "n_frames": 1000, "width": 1920, "height": 1080}


def outputs(root: Path) -> Path:
    out = root / "outputs" / "clip"
    out.mkdir(parents=True)
    (out / "ball.json").write_text(json.dumps({"video": VIDEO, "frames": [], "deliveries": [
        {"delivery_id": 0, "release_frame": 100, "release_t": 4.0, "bounce_t": 4.4, "length_m": 3.9, "line_m": 0.12,
         "length_class": "full", "bounce_confirmed": True, "radar_speed_kph": 89.2, "speed_estimate_kph": 95.0},
    ]}))
    (out / "events.json").write_text(json.dumps({"video": VIDEO, "events": [
        {"frame_id": 100, "t": 4.0, "type": "RELEASE", "payload": {"delivery_id": 0, "over": "11.5", "bowler": "SHAMSI", "striker": "AXAR"}, "confidence": 0.8, "source": "t"},
        {"frame_id": 300, "t": 12.0, "type": "RUNS", "payload": {"delivery_id": 0, "runs": 6, "off_bat": 6, "extras": 0, "kind": "legal", "score": "91-3"}, "confidence": 0.9, "source": "t"},
        {"frame_id": 300, "t": 12.0, "type": "BOUNDARY", "payload": {"delivery_id": 0, "kind": "SIX"}, "confidence": 0.95, "source": "t"},
    ]}))
    (out / "match.json").write_text(json.dumps({"video": VIDEO, "states": [
        {"delivery_id": 0, "frame_id": 300, "t": 12.0, "over": "11.5", "runs": 91, "wickets": 3, "legal_balls": 71,
         "striker": "AXAR", "non_striker": "KOHLI", "bowler": "SHAMSI", "batters": {"AXAR": [36, 25]}, "bowlers": {"SHAMSI": [16, 11, 0]},
         "source": "engine", "checks": {"score": True}, "notes": []},
    ]}))
    (out / "commentary.json").write_text(json.dumps({"video": VIDEO, "engine": "template", "facts": [{"delivery_id": 0, "shot": "slog sweep"}],
        "lines": [{"delivery_id": 0, "frame_id": 110, "t": 4.4, "kind": "ball", "text": "SIX! Axar launches it.", "source": "template", "max_s": 5.0}]}))
    (out / "pose.json").write_text(json.dumps({"video": VIDEO, "frames": [], "batters": [
        {"delivery_id": 0, "footwork": "front foot", "swing": "played", "batting_hand": "left"}]}))
    (out / "pitch_map.png").write_bytes(b"\x89PNG fake")
    return out


def test_one_row_per_ball_joins_every_pass(tmp_path):
    rows = delivery_rows(outputs(tmp_path))
    r = rows[0]
    assert (r["over"], r["bowler"], r["striker"], r["runs"], r["boundary"], r["shot"], r["footwork"], r["batting_hand"]) == (
        "11.5", "SHAMSI", "AXAR", 6, "SIX", "slog sweep", "front foot", "left")


def test_publish_is_idempotent_and_stores_files(tmp_path, monkeypatch):
    outputs(tmp_path)
    cfg = load_config()
    monkeypatch.setattr(type(cfg), "output_dir_for", lambda self, vid: tmp_path / "outputs" / vid)
    monkeypatch.setenv("CRICSENSE_DB_URL", f"sqlite:///{tmp_path / 'c.db'}")
    import cricsense.passes.publish as pub
    from cricsense.store.objects import LocalStore

    monkeypatch.setattr(pub, "open_store", lambda c, root: LocalStore(tmp_path / "store"))
    engine = db.connect(f"sqlite:///{tmp_path / 'c.db'}")
    first = run_publish(cfg, "clip", engine)
    again = run_publish(cfg, "clip", engine)
    assert first["deliveries"] == again["deliveries"] == 1 and again["events"] == 3 and again["artifacts"] == 5  # pitch map + 4 JSON outputs
    with engine.connect() as conn:
        assert len(conn.execute(db.select(db.deliveries)).all()) == 1  # replaced, not duplicated
    assert (tmp_path / "store" / "clip" / "pitch_map.png").is_file()


def test_api_serves_what_was_published(tmp_path, monkeypatch):
    outputs(tmp_path)
    cfg = load_config()
    monkeypatch.setattr(type(cfg), "output_dir_for", lambda self, vid: tmp_path / "outputs" / vid)
    import cricsense.passes.publish as pub
    from cricsense.store.objects import LocalStore

    monkeypatch.setattr(pub, "open_store", lambda c, root: LocalStore(tmp_path / "store"))
    engine = db.connect(f"sqlite:///{tmp_path / 'c.db'}")
    run_publish(cfg, "clip", engine)
    client = TestClient(create_app(cfg, engine, run_pipeline=lambda p: [], publish=lambda v: {}))
    assert client.get("/health").json() == {"ok": True}
    assert [v["video_id"] for v in client.get("/videos").json()] == ["clip"]
    d = client.get("/videos/clip/deliveries").json()[0]
    assert (d["boundary"], d["radar_kph"], d["shot"]) == ("SIX", 89.2, "slog sweep")
    assert [e["type"] for e in client.get("/videos/clip/events", params={"type": "boundary"}).json()] == ["BOUNDARY"]
    assert client.get("/videos/clip/scorecard").json()["runs"] == 91
    assert client.get("/videos/clip/commentary").json()[0]["text"].startswith("SIX")
    assert client.get("/videos/clip/artifacts/pitch_map").content == b"\x89PNG fake"
    assert client.get("/videos/nope").status_code == 404


def test_upload_runs_a_job(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(type(cfg.paths), "video_dir", property(lambda self: tmp_path / "data"), raising=False)
    engine = db.connect(f"sqlite:///{tmp_path / 'c.db'}")
    ran = []
    client = TestClient(create_app(cfg, engine, run_pipeline=lambda p: ran.append(p.name) or [("render", "ran", "")],
                                   publish=lambda v: {"video_id": v}))
    job = client.post("/videos", files={"file": ("My Clip.mp4", b"not really a video", "video/mp4")}).json()
    for _ in range(50):
        state = client.get(f"/jobs/{job['job_id']}").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["state"] == "done" and ran == ["My_Clip.mp4"] and state["published"] == {"video_id": "My_Clip"}
    assert client.post("/videos", files={"file": ("notes.txt", b"x", "text/plain")}).status_code == 415
