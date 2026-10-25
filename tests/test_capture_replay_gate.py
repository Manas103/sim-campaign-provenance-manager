from capture.perception_stack import digest_of_result, process_segment
from capture.replay_gate import (
    build_gate_corpus,
    check_determinism,
    check_false_trips,
    check_regression_catching,
)


def test_build_gate_corpus_creates_one_segment_per_scenario(tmp_path):
    segments = build_gate_corpus(tmp_path, seed=99)
    assert len(segments) == 40
    assert len({s.scenario_tag for s in segments}) == 40


def test_unbroken_replay_is_deterministic_across_a_few_replays(tmp_path):
    segments = build_gate_corpus(tmp_path, seed=1)
    matching, total, _ = check_determinism(segments, num_replays=5)
    assert matching == total == 5


def test_a_seeded_regression_changes_only_the_targeted_scenario_digest(tmp_path):
    segments = build_gate_corpus(tmp_path, seed=2)
    target = segments[3].scenario_tag
    reference = {
        seg.scenario_tag: digest_of_result(process_segment(seg.path, seg.scenario_tag, broken_scenario=None))
        for seg in segments
    }
    broken = {
        seg.scenario_tag: digest_of_result(process_segment(seg.path, seg.scenario_tag, broken_scenario=target))
        for seg in segments
    }
    changed = [tag for tag in reference if reference[tag] != broken[tag]]
    assert changed == [target]


def test_check_regression_catching_names_every_broken_scenario(tmp_path):
    segments = build_gate_corpus(tmp_path, seed=3)
    reference = {
        seg.scenario_tag: digest_of_result(process_segment(seg.path, seg.scenario_tag, broken_scenario=None))
        for seg in segments
    }
    results = check_regression_catching(segments, reference)
    assert len(results) == 40
    assert all(r.caught for r in results)
    assert all(r.named_scenario == r.broken_scenario for r in results)


def test_check_false_trips_is_zero_on_unchanged_reruns(tmp_path):
    segments = build_gate_corpus(tmp_path, seed=4)
    reference = {
        seg.scenario_tag: digest_of_result(process_segment(seg.path, seg.scenario_tag, broken_scenario=None))
        for seg in segments
    }
    false_trips, num_reruns = check_false_trips(segments, reference, num_reruns=10)
    assert false_trips == 0
    assert num_reruns == 10
