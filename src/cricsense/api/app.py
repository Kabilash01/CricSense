"""The HTTP API: upload a video, follow its analysis, read the results.

    cricsense serve --port 8000        # then http://localhost:8000/docs

POST /videos                      upload an MP4; the whole pipeline runs, then publishes (a job)
GET  /jobs/{job_id}               that job's state, step by step
GET  /videos                      published videos
GET  /videos/{video_id}           one video, with counts of what was found
GET  /videos/{video_id}/deliveries        ball by ball: length, line, speeds, outcome, shot, footwork
GET  /videos/{video_id}/events?type=      the event timeline
GET  /videos/{video_id}/scorecard         the score engine's state after the last ball
GET  /videos/{video_id}/commentary        the commentary lines
GET  /videos/{video_id}/artifacts/{kind}  overlay | pitch_map | wagon_wheel | commentary_audio | *_json
Jobs run one at a time in a background thread (the GPU is the bottleneck); their state is kept
in memory, the results in the database.
"""

from __future__ import annotations

import re
import shutil
import threading
import uuid
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select

from cricsense.config import Config
from cricsense.store import db

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def create_app(cfg: Config, engine, run_pipeline=None, publish=None) -> FastAPI:
    """run_pipeline(video_path) -> list of (step, status, note); publish(video_id) -> dict.
    Injectable so tests can run the API without the models."""
    if run_pipeline is None:
        from cricsense.passes.run import run_all

        def run_pipeline(path: Path):
            return [(r.name, r.status, r.note) for r in run_all(cfg, path)]

    if publish is None:
        from cricsense.passes.publish import run_publish

        def publish(video_id: str):
            return run_publish(cfg, video_id, engine)

    app = FastAPI(title="CricSense", description="Cricket broadcast analysis: deliveries, events, score, commentary.")
    jobs: dict[str, dict] = {}
    queue: Queue = Queue()

    def worker() -> None:
        while True:
            job_id, path = queue.get()
            job = jobs[job_id]
            job["state"] = "running"
            try:
                steps = run_pipeline(path)
                job["steps"] = [{"step": s, "status": st, "note": n} for s, st, n in steps]
                if any(st == "failed" for _, st, _ in steps):
                    job["state"] = "failed"
                else:
                    job["published"] = publish(path.stem)
                    job["state"] = "done"
            except Exception as e:  # report, keep serving
                job["state"], job["error"] = "failed", f"{type(e).__name__}: {e}"
            finally:
                queue.task_done()

    threading.Thread(target=worker, daemon=True).start()

    def video_row(conn, video_id: str):
        row = conn.execute(select(db.videos).where(db.videos.c.video_id == video_id)).mappings().first()
        if row is None:
            raise HTTPException(404, f"no published video {video_id!r}")
        return row

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/videos", status_code=202)
    async def upload(file: UploadFile = File(...)) -> dict:
        name = SAFE_NAME.sub("_", Path(file.filename or "upload.mp4").stem).strip("_") or "upload"
        if not (file.filename or "").lower().endswith((".mp4", ".mov", ".mkv")):
            raise HTTPException(415, "upload a video file (.mp4, .mov, .mkv)")
        dest = cfg.paths.video_dir / f"{name}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"job_id": job_id, "video_id": name, "state": "queued"}
        queue.put((job_id, dest))
        return jobs[job_id]

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

    @app.get("/videos")
    def list_videos() -> list[dict]:
        with engine.connect() as conn:
            return [dict(r) for r in conn.execute(select(db.videos).order_by(db.videos.c.id)).mappings()]

    @app.get("/videos/{video_id}")
    def get_video(video_id: str) -> dict:
        with engine.connect() as conn:
            v = dict(video_row(conn, video_id))
            for name, t in (("deliveries", db.deliveries), ("events", db.events), ("commentary_lines", db.commentary)):
                v[name] = len(conn.execute(select(t.c.id).where(t.c.video_id == v["id"])).all())
            v["artifacts"] = [r.kind for r in conn.execute(select(db.artifacts.c.kind).where(db.artifacts.c.video_id == v["id"]))]
            return v

    @app.get("/videos/{video_id}/deliveries")
    def get_deliveries(video_id: str) -> list[dict]:
        with engine.connect() as conn:
            v = video_row(conn, video_id)
            q = select(db.deliveries).where(db.deliveries.c.video_id == v["id"]).order_by(db.deliveries.c.release_t)
            return [dict(r) for r in conn.execute(q).mappings()]

    @app.get("/videos/{video_id}/events")
    def get_events(video_id: str, type: str | None = Query(None, description="e.g. BOUNDARY, WICKET, BOUNCE")) -> list[dict]:
        with engine.connect() as conn:
            v = video_row(conn, video_id)
            q = select(db.events).where(db.events.c.video_id == v["id"])
            if type:
                q = q.where(db.events.c.type == type.upper())
            return [dict(r) for r in conn.execute(q.order_by(db.events.c.frame_id)).mappings()]

    @app.get("/videos/{video_id}/scorecard")
    def scorecard(video_id: str) -> dict:
        with engine.connect() as conn:
            v = video_row(conn, video_id)
            q = select(db.match_states).where(db.match_states.c.video_id == v["id"]).order_by(db.match_states.c.t.desc())
            last = conn.execute(q).mappings().first()
            if last is None:
                raise HTTPException(404, "no score state for this video")
            return dict(last)

    @app.get("/videos/{video_id}/commentary")
    def get_commentary(video_id: str) -> list[dict]:
        with engine.connect() as conn:
            v = video_row(conn, video_id)
            q = select(db.commentary).where(db.commentary.c.video_id == v["id"]).order_by(db.commentary.c.t)
            return [dict(r) for r in conn.execute(q).mappings()]

    @app.get("/videos/{video_id}/artifacts/{kind}")
    def artifact(video_id: str, kind: str):
        with engine.connect() as conn:
            v = video_row(conn, video_id)
            a = conn.execute(select(db.artifacts).where(db.artifacts.c.video_id == v["id"], db.artifacts.c.kind == kind)).mappings().first()
        if a is None:
            raise HTTPException(404, f"no {kind!r} for {video_id!r}")
        if a["uri"].startswith("file://"):
            return FileResponse(a["uri"].removeprefix("file://"))
        return RedirectResponse(a["uri"])

    app.state.jobs = jobs
    app.state.queue = queue
    return app
