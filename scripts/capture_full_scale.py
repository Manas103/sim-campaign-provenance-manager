"""Builds the full-scale capture corpus: 12 topics, a 100 Hz tick, 2,000,000+ frames across
timestamped Parquet segments, indexed by scenario tag and time range in SQLite.

Usage: python3 scripts/capture_full_scale.py --out-dir data/capture --seed 11
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.segment_index import insert_segment, open_index, total_dropped, total_frame_count
from capture.segment_writer import DEFAULT_SCENARIO_TAGS, segment_buffer_capacity, write_segment
from capture.stream_simulator import simulate_ticks, total_frame_count as simulated_frame_count
from capture.topics import TOPICS

TOTAL_TICKS = 610_000
TICKS_PER_SEGMENT = 10_000
BUFFER_HEADROOM = 0.995


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/capture")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    segments_dir = out_dir / "segments"
    db_path = out_dir / "segment_index.sqlite"
    if db_path.exists():
        db_path.unlink()

    t0 = time.perf_counter()
    frames = simulate_ticks(TOTAL_TICKS, seed=args.seed)
    simulate_elapsed_s = time.perf_counter() - t0

    conn = open_index(db_path)
    num_segments = (TOTAL_TICKS + TICKS_PER_SEGMENT - 1) // TICKS_PER_SEGMENT

    buffer_capacity = {
        t.name: segment_buffer_capacity(t.rate_hz, TICKS_PER_SEGMENT, BUFFER_HEADROOM) for t in TOPICS
    }

    t0 = time.perf_counter()
    for seg_id in range(num_segments):
        start_tick = seg_id * TICKS_PER_SEGMENT
        end_tick = min(start_tick + TICKS_PER_SEGMENT, TOTAL_TICKS)
        scenario_tag = DEFAULT_SCENARIO_TAGS[seg_id % len(DEFAULT_SCENARIO_TAGS)]
        meta = write_segment(
            segments_dir, seg_id, start_tick, end_tick, frames, buffer_capacity, scenario_tag
        )
        insert_segment(conn, meta)
    conn.commit()
    write_elapsed_s = time.perf_counter() - t0

    summary = {
        "total_ticks": TOTAL_TICKS,
        "num_segments": num_segments,
        "ticks_per_segment": TICKS_PER_SEGMENT,
        "simulated_frame_count": simulated_frame_count(frames),
        "indexed_frame_count": total_frame_count(conn),
        "dropped_by_topic": total_dropped(conn),
        "simulate_elapsed_s": round(simulate_elapsed_s, 3),
        "write_and_index_elapsed_s": round(write_elapsed_s, 3),
        "db_path": str(db_path),
        "segments_dir": str(segments_dir),
    }
    print(json.dumps(summary, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
