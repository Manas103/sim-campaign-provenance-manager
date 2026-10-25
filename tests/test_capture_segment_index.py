from capture.segment_index import insert_segment, open_index, query_segments, total_dropped, total_frame_count
from capture.segment_writer import write_segment
from capture.stream_simulator import simulate_ticks


def _build_index(tmp_path, num_segments=3, ticks_per_segment=200, seed=20):
    db_path = tmp_path / "index.sqlite"
    conn = open_index(db_path)
    frames = simulate_ticks(num_segments * ticks_per_segment, seed=seed)
    capacity = {name: tf.tick_indices.size for name, tf in frames.items()}
    tags = ["alpha", "beta", "alpha"]
    for seg_id in range(num_segments):
        start = seg_id * ticks_per_segment
        end = start + ticks_per_segment
        meta = write_segment(tmp_path / "segs", seg_id, start, end, frames, capacity, tags[seg_id % len(tags)])
        insert_segment(conn, meta)
    conn.commit()
    return conn


def test_query_by_scenario_tag_returns_only_matching_segments(tmp_path):
    conn = _build_index(tmp_path)
    matches = query_segments(conn, "alpha", 0, 10**18)
    assert {m.scenario_tag for m in matches} == {"alpha"}
    assert len(matches) == 2


def test_query_by_time_range_excludes_out_of_range_segments(tmp_path):
    conn = _build_index(tmp_path)
    matches = query_segments(conn, "alpha", 0, 1)
    assert len(matches) == 1


def test_unknown_scenario_tag_returns_nothing(tmp_path):
    conn = _build_index(tmp_path)
    assert query_segments(conn, "nonexistent", 0, 10**18) == []


def test_total_frame_count_sums_all_segments(tmp_path):
    conn = _build_index(tmp_path)
    assert total_frame_count(conn) > 0


def test_total_dropped_has_an_entry_per_topic(tmp_path):
    conn = _build_index(tmp_path)
    dropped = total_dropped(conn)
    assert len(dropped) == 12
