"""Writes a simulated tick range to a timestamped Parquet segment, with per-topic drop accounting.

Drops are not scripted to a target count: each topic has a fixed per-segment buffer capacity
(roughly, but not exactly, its expected frame count for that segment), and whatever natural
Bernoulli variance in stream_simulator.py pushes a topic over capacity in a given segment is
dropped and counted. Some runs drop zero frames for a given topic; that is reported as measured,
not padded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from capture.stream_simulator import TICK_PERIOD_NS, TopicFrames

DEFAULT_SCENARIO_TAGS = (
    "intersection_left_turn",
    "highway_merge",
    "roundabout",
    "pedestrian_crossing",
    "construction_zone",
    "unprotected_left",
    "parking_lot_low_speed",
    "rain_highway",
    "night_residential",
    "school_zone",
)


@dataclass
class SegmentMeta:
    segment_id: int
    start_tick: int
    end_tick: int
    start_ts_ns: int
    end_ts_ns: int
    scenario_tag: str
    path: str
    frame_count: int
    dropped_by_topic: dict[str, int] = field(default_factory=dict)


def segment_buffer_capacity(topic_rate_hz: float, ticks_per_segment: int, headroom: float) -> int:
    """A per-topic buffer sized just under its expected frame count, so natural variance

    occasionally overflows it. `headroom` < 1.0 means the buffer is tighter than expected load.
    """
    expected = topic_rate_hz / 100.0 * ticks_per_segment
    return max(1, int(expected * headroom))


def write_segment(
    out_dir: Path,
    segment_id: int,
    start_tick: int,
    end_tick: int,
    frames_by_topic: dict[str, TopicFrames],
    buffer_capacity: dict[str, int],
    scenario_tag: str,
) -> SegmentMeta:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_topic: list[str] = []
    rows_ts: list[int] = []
    rows_seq: list[int] = []
    rows_hash: list[str] = []
    dropped_by_topic: dict[str, int] = {}

    for topic_name, tf in frames_by_topic.items():
        mask = (tf.tick_indices >= start_tick) & (tf.tick_indices < end_tick)
        idx = mask.nonzero()[0]
        capacity = buffer_capacity.get(topic_name, len(idx))
        kept_idx = idx[:capacity]
        dropped = len(idx) - len(kept_idx)
        dropped_by_topic[topic_name] = int(dropped)
        for i in kept_idx:
            rows_topic.append(topic_name)
            rows_ts.append(int(tf.timestamps_ns[i]))
            rows_seq.append(int(tf.sequence[i]))
            rows_hash.append(str(tf.payload_hash[i]))

    table = pa.table(
        {
            "topic": pa.array(rows_topic, type=pa.string()),
            "timestamp_ns": pa.array(rows_ts, type=pa.int64()),
            "sequence": pa.array(rows_seq, type=pa.int64()),
            "payload_hash": pa.array(rows_hash, type=pa.string()),
        }
    )
    path = out_dir / f"segment_{segment_id:05d}_{scenario_tag}.parquet"
    pq.write_table(table, path)

    start_ts_ns = start_tick * TICK_PERIOD_NS
    end_ts_ns = end_tick * TICK_PERIOD_NS
    return SegmentMeta(
        segment_id=segment_id,
        start_tick=start_tick,
        end_tick=end_tick,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
        scenario_tag=scenario_tag,
        path=str(path),
        frame_count=len(rows_topic),
        dropped_by_topic=dropped_by_topic,
    )
