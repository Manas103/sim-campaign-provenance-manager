"""SQLite index over captured segments: query by scenario tag and time range.

One row per segment (not per frame): `frame_count` records how many frames that segment holds, so
the index describes a multi-million-frame corpus without a multi-million-row table. The query this
module exists for is "which segments, if any, cover this scenario tag and overlap this time
range", which is exactly the shape of "find me the segments to replay for scenario X between
t0 and t1".
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from capture.segment_writer import SegmentMeta

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY,
    scenario_tag TEXT NOT NULL,
    start_tick INTEGER NOT NULL,
    end_tick INTEGER NOT NULL,
    start_ts_ns INTEGER NOT NULL,
    end_ts_ns INTEGER NOT NULL,
    path TEXT NOT NULL,
    frame_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_tag_range
    ON segments (scenario_tag, start_ts_ns, end_ts_ns);

CREATE TABLE IF NOT EXISTS topic_drops (
    segment_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    dropped_count INTEGER NOT NULL,
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
);
CREATE INDEX IF NOT EXISTS idx_topic_drops_segment ON topic_drops (segment_id);
"""


def open_index(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_segment(conn: sqlite3.Connection, meta: SegmentMeta) -> None:
    conn.execute(
        "INSERT INTO segments (segment_id, scenario_tag, start_tick, end_tick, start_ts_ns, end_ts_ns, path, frame_count)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meta.segment_id,
            meta.scenario_tag,
            meta.start_tick,
            meta.end_tick,
            meta.start_ts_ns,
            meta.end_ts_ns,
            meta.path,
            meta.frame_count,
        ),
    )
    conn.executemany(
        "INSERT INTO topic_drops (segment_id, topic, dropped_count) VALUES (?, ?, ?)",
        [(meta.segment_id, topic, dropped) for topic, dropped in meta.dropped_by_topic.items()],
    )


def total_frame_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(SUM(frame_count), 0) FROM segments").fetchone()
    return int(row[0])


def total_dropped(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT topic, SUM(dropped_count) FROM topic_drops GROUP BY topic"
    ).fetchall()
    return {topic: int(total) for topic, total in rows}


@dataclass
class SegmentMatch:
    segment_id: int
    scenario_tag: str
    path: str
    frame_count: int


def query_segments(
    conn: sqlite3.Connection, scenario_tag: str, ts_start_ns: int, ts_end_ns: int
) -> list[SegmentMatch]:
    rows = conn.execute(
        "SELECT segment_id, scenario_tag, path, frame_count FROM segments"
        " WHERE scenario_tag = ? AND start_ts_ns <= ? AND end_ts_ns >= ?",
        (scenario_tag, ts_end_ns, ts_start_ns),
    ).fetchall()
    return [SegmentMatch(*row) for row in rows]


def timed_query_segments(
    conn: sqlite3.Connection, scenario_tag: str, ts_start_ns: int, ts_end_ns: int
) -> tuple[list[SegmentMatch], float]:
    t0 = time.perf_counter()
    matches = query_segments(conn, scenario_tag, ts_start_ns, ts_end_ns)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return matches, elapsed_ms
