import pyarrow.parquet as pq

from capture.segment_writer import segment_buffer_capacity, write_segment
from capture.stream_simulator import simulate_ticks


def test_write_segment_creates_parquet_with_expected_columns(tmp_path):
    frames = simulate_ticks(500, seed=10)
    capacity = {name: tf.tick_indices.size for name, tf in frames.items()}
    meta = write_segment(tmp_path, 0, 0, 500, frames, capacity, "test_scenario")
    table = pq.read_table(meta.path)
    assert set(table.column_names) == {"topic", "timestamp_ns", "sequence", "payload_hash"}
    assert table.num_rows == meta.frame_count


def test_no_drops_when_capacity_is_generous(tmp_path):
    frames = simulate_ticks(500, seed=11)
    capacity = {name: tf.tick_indices.size for name, tf in frames.items()}
    meta = write_segment(tmp_path, 0, 0, 500, frames, capacity, "test_scenario")
    assert all(d == 0 for d in meta.dropped_by_topic.values())


def test_drops_when_capacity_is_tight(tmp_path):
    frames = simulate_ticks(2000, seed=12)
    tight_capacity = {name: max(1, tf.tick_indices.size // 2) for name, tf in frames.items()}
    meta = write_segment(tmp_path, 0, 0, 2000, frames, tight_capacity, "test_scenario")
    assert sum(meta.dropped_by_topic.values()) > 0
    assert meta.frame_count == sum(tight_capacity.values())


def test_segment_buffer_capacity_is_below_expected_with_headroom_under_one():
    cap_full = segment_buffer_capacity(100.0, 10_000, 1.0)
    cap_tight = segment_buffer_capacity(100.0, 10_000, 0.9)
    assert cap_tight < cap_full


def test_segment_only_includes_ticks_in_its_range(tmp_path):
    frames = simulate_ticks(1000, seed=13)
    capacity = {name: tf.tick_indices.size for name, tf in frames.items()}
    meta = write_segment(tmp_path, 0, 200, 400, frames, capacity, "test_scenario")
    table = pq.read_table(meta.path)
    ts = table.column("timestamp_ns").to_pylist()
    assert all(meta.start_ts_ns <= t < meta.end_ts_ns for t in ts)
