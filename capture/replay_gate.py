"""The replay regression gate: build 40 gate segments, then replay them through the perception
stack to prove three things, each measured separately.

1. Determinism: replaying the unchanged gate corpus many times produces a bit-identical SHA-256
   digest every time.
2. Regression detection: a seeded, one-scenario model regression is caught, and the gate names
   exactly the scenario it broke.
3. No false trips: replaying the unchanged gate corpus does not spuriously flag a regression.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from capture.perception_stack import GATE_SCENARIO_TAGS, digest_of_result, process_segment
from capture.segment_writer import write_segment
from capture.stream_simulator import simulate_ticks

GATE_TICKS_PER_SCENARIO = 2_000


@dataclass
class GateSegment:
    scenario_tag: str
    path: str


def build_gate_corpus(out_dir: Path, seed: int = 7) -> list[GateSegment]:
    """Builds one small segment per gate scenario tag, each from its own independently seeded

    simulated stream so the 40 scenarios are not just slices of one continuous stream.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    segments: list[GateSegment] = []
    for i, scenario_tag in enumerate(GATE_SCENARIO_TAGS):
        frames = simulate_ticks(GATE_TICKS_PER_SCENARIO, seed=seed * 1000 + i)
        capacity = {name: tf.tick_indices.size for name, tf in frames.items()}
        meta = write_segment(
            out_dir, segment_id=i, start_tick=0, end_tick=GATE_TICKS_PER_SCENARIO,
            frames_by_topic=frames, buffer_capacity=capacity, scenario_tag=scenario_tag,
        )
        segments.append(GateSegment(scenario_tag=scenario_tag, path=meta.path))
    return segments


def _per_scenario_digests(segments: list[GateSegment], broken_scenario: str | None) -> dict[str, str]:
    digests: dict[str, str] = {}
    for seg in segments:
        result = process_segment(seg.path, seg.scenario_tag, broken_scenario=broken_scenario)
        digests[seg.scenario_tag] = digest_of_result(result)
    return digests


def combined_digest(per_scenario_digests: dict[str, str]) -> str:
    canonical = json.dumps(per_scenario_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def check_determinism(segments: list[GateSegment], num_replays: int) -> tuple[int, int, str]:
    """Returns (num_matching, num_replays, reference_combined_digest)."""
    reference = combined_digest(_per_scenario_digests(segments, broken_scenario=None))
    matching = 0
    for _ in range(num_replays):
        d = combined_digest(_per_scenario_digests(segments, broken_scenario=None))
        if d == reference:
            matching += 1
    return matching, num_replays, reference


@dataclass
class RegressionCheckResult:
    broken_scenario: str
    caught: bool
    named_scenario: str | None


def check_regression_catching(
    segments: list[GateSegment], reference_digests: dict[str, str]
) -> list[RegressionCheckResult]:
    results: list[RegressionCheckResult] = []
    for scenario_tag in GATE_SCENARIO_TAGS:
        candidate_digests = _per_scenario_digests(segments, broken_scenario=scenario_tag)
        mismatched = [tag for tag in reference_digests if candidate_digests[tag] != reference_digests[tag]]
        caught = mismatched == [scenario_tag]
        results.append(
            RegressionCheckResult(
                broken_scenario=scenario_tag,
                caught=caught,
                named_scenario=mismatched[0] if len(mismatched) == 1 else None,
            )
        )
    return results


def check_false_trips(
    segments: list[GateSegment], reference_digests: dict[str, str], num_reruns: int
) -> tuple[int, int]:
    """Returns (num_false_tripped, num_reruns)."""
    false_trips = 0
    for _ in range(num_reruns):
        candidate_digests = _per_scenario_digests(segments, broken_scenario=None)
        mismatched = [tag for tag in reference_digests if candidate_digests[tag] != reference_digests[tag]]
        if mismatched:
            false_trips += 1
    return false_trips, num_reruns
