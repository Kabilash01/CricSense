"""The database: matches, videos, ball-by-ball deliveries, events, score states, commentary,
and the files rendered for each video.

SQLAlchemy Core, so the same tables run on PostgreSQL (the deployment) and SQLite (tests).
Loading is idempotent: publishing a video again replaces its rows, never duplicates them.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

meta = MetaData()

matches = Table(
    "matches", meta,
    Column("id", Integer, primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("teams", JSON, nullable=False),  # {"IND": {"name": "India", "squad": [...]}, ...}
)

videos = Table(
    "videos", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", String(200), nullable=False, unique=True),
    Column("match_id", ForeignKey("matches.id", ondelete="SET NULL")),
    Column("fps", Float, nullable=False),
    Column("n_frames", Integer, nullable=False),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column("source", Text),  # the broadcast it was cut from, and where (configs/clips.yaml)
    Column("published_at", DateTime(timezone=True), server_default=func.now()),
)

deliveries = Table(
    "deliveries", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("delivery_id", Integer, nullable=False),
    Column("over", String(8)),
    Column("bowler", String(80)),
    Column("striker", String(80)),
    Column("release_t", Float, nullable=False),
    Column("bounce_t", Float),
    Column("length_m", Float),
    Column("line_m", Float),
    Column("length_class", String(10)),
    Column("bounce_confirmed", Boolean, nullable=False, default=False),
    Column("radar_kph", Float),
    Column("estimate_kph", Float),
    Column("kind", String(10)),  # legal | wide | no_ball | extra
    Column("runs", Integer),
    Column("off_bat", Integer),
    Column("extras", Integer),
    Column("wicket", Boolean),
    Column("boundary", String(4)),
    Column("score_after", String(10)),
    Column("shot", String(40)),  # read by the VLM (posture), named by rule
    Column("footwork", String(20)),  # measured by pose
    Column("swing", String(10)),
    Column("batting_hand", String(5)),
    UniqueConstraint("video_id", "delivery_id"),
)

events = Table(
    "events", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("frame_id", Integer, nullable=False),
    Column("t", Float, nullable=False),
    Column("type", String(20), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("source", String(60), nullable=False),
)

match_states = Table(
    "match_states", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("delivery_id", Integer, nullable=False),
    Column("t", Float, nullable=False),
    Column("over", String(8), nullable=False),
    Column("runs", Integer, nullable=False),
    Column("wickets", Integer, nullable=False),
    Column("striker", String(80)),
    Column("non_striker", String(80)),
    Column("bowler", String(80)),
    Column("batters", JSON, nullable=False),
    Column("bowlers", JSON, nullable=False),
    Column("source", String(12), nullable=False),  # engine | resync | no_evidence
    Column("checks", JSON, nullable=False),
)

commentary = Table(
    "commentary_lines", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("delivery_id", Integer, nullable=False),
    Column("t", Float, nullable=False),
    Column("kind", String(6), nullable=False),
    Column("text", Text, nullable=False),
    Column("source", String(10), nullable=False),  # llm | template
)

artifacts = Table(
    "artifacts", meta,
    Column("id", Integer, primary_key=True),
    Column("video_id", ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(30), nullable=False),  # overlay | pitch_map | wagon_wheel | commentary_audio | <name>_json
    Column("uri", Text, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("bytes", Integer, nullable=False),
    UniqueConstraint("video_id", "kind"),
)

CHILD_TABLES = (deliveries, events, match_states, commentary, artifacts)


def connect(url: str) -> Engine:
    engine = create_engine(url, future=True)
    meta.create_all(engine)
    return engine


def upsert_match(conn, name: str, teams: dict) -> int:
    row = conn.execute(select(matches.c.id).where(matches.c.name == name)).first()
    if row:
        return row.id
    return conn.execute(insert(matches).values(name=name, teams=teams)).inserted_primary_key[0]


def replace_video(conn, video_id: str, **cols) -> int:
    """The video's row, its children cleared: a re-publish replaces, never duplicates."""
    row = conn.execute(select(videos.c.id).where(videos.c.video_id == video_id)).first()
    if row:
        for t in CHILD_TABLES:
            conn.execute(delete(t).where(t.c.video_id == row.id))
        conn.execute(videos.update().where(videos.c.id == row.id).values(**cols, published_at=func.now()))
        return row.id
    return conn.execute(insert(videos).values(video_id=video_id, **cols)).inserted_primary_key[0]
