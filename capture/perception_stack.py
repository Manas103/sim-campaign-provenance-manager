"""A deterministic stand-in for 'the perception stack', not a real perception model.

What is under test by the replay regression gate is the capture/replay/regression-detection
machinery, not model accuracy: this module reads a captured segment and computes a deterministic,
purely-data-derived checksum per topic, with no randomness and no ML inference. A "seeded model
regression" is simulated as a deterministic, fixed corruption of one named scenario's checksum, as
if a code change had broken exactly that scenario; the gate's job is to notice the checksum no
longer matches the reference, for that scenario only.
"""
from __future__ import annotations

import hashlib
import json

import pyarrow.parquet as pq

GATE_SCENARIO_TAGS: tuple[str, ...] = tuple(f"gate_scenario_{i:03d}" for i in range(40))


def process_segment(path: str, scenario_tag: str, broken_scenario: str | None = None) -> dict:
    table = pq.read_table(path)
    topics = table.column("topic").to_pylist()
    hashes = table.column("payload_hash").to_pylist()

    per_topic_checksum: dict[str, int] = {}
    for topic, h in zip(topics, hashes):
        value = int(h[:8], 16)
        per_topic_checksum[topic] = (per_topic_checksum.get(topic, 0) + value) % (2**32)

    if broken_scenario is not None and scenario_tag == broken_scenario:
        first_topic = sorted(per_topic_checksum.keys())[0]
        per_topic_checksum[first_topic] = (per_topic_checksum[first_topic] ^ 0xDEADBEEF) % (2**32)

    return {
        "scenario_tag": scenario_tag,
        "frame_count": len(topics),
        "per_topic_checksum": dict(sorted(per_topic_checksum.items())),
    }


def digest_of_result(result: dict) -> str:
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
