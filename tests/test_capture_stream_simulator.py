from capture.stream_simulator import TICK_PERIOD_NS, simulate_ticks, total_frame_count
from capture.topics import TOPICS


def test_twelve_topics_present():
    frames = simulate_ticks(1000, seed=1)
    assert set(frames.keys()) == {t.name for t in TOPICS}


def test_deterministic_for_same_seed():
    a = simulate_ticks(2000, seed=42)
    b = simulate_ticks(2000, seed=42)
    for name in a:
        assert list(a[name].tick_indices) == list(b[name].tick_indices)
        assert list(a[name].payload_hash) == list(b[name].payload_hash)


def test_different_seeds_differ():
    a = simulate_ticks(2000, seed=1)
    b = simulate_ticks(2000, seed=2)
    assert any(list(a[name].tick_indices) != list(b[name].tick_indices) for name in a)


def test_higher_rate_topic_emits_more_on_average():
    frames = simulate_ticks(50_000, seed=3)
    assert frames["imu"].tick_indices.size > frames["battery_status"].tick_indices.size


def test_timestamps_consistent_with_tick_period():
    frames = simulate_ticks(500, seed=4)
    for tf in frames.values():
        expected = tf.tick_indices * TICK_PERIOD_NS
        assert list(tf.timestamps_ns) == list(expected)


def test_total_frame_count_matches_sum_of_topics():
    frames = simulate_ticks(1000, seed=5)
    assert total_frame_count(frames) == sum(tf.tick_indices.size for tf in frames.values())
