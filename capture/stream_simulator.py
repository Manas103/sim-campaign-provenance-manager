"""Simulated 12-topic robot sensor stream at a 100 Hz capture tick.

This does not drive real sensors or real wall-clock timing: timestamps are synthetic (tick index
times the 10ms tick period), and each topic's emission at each tick is an independent Bernoulli
draw at rate_hz / TICK_RATE_HZ, generated vectorized with numpy rather than one Python loop
iteration per tick, which is what makes a multi-million-frame corpus buildable in seconds instead
of minutes. The payload itself is a small deterministic hash standing in for real sensor bytes;
no camera or lidar frame is actually rendered.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from capture.topics import TICK_RATE_HZ, Topic, TOPICS

TICK_PERIOD_NS = 1_000_000_000 // TICK_RATE_HZ


@dataclass(frozen=True)
class TopicFrames:
    topic: Topic
    tick_indices: np.ndarray  # int64, ticks at which this topic emitted
    timestamps_ns: np.ndarray  # int64
    sequence: np.ndarray  # int64, 0-based per-topic sequence number
    payload_hash: np.ndarray  # object/str, 16 hex chars


def simulate_ticks(num_ticks: int, seed: int) -> dict[str, TopicFrames]:
    """Generates emission events for all 12 topics over num_ticks ticks, deterministic in seed."""
    result: dict[str, TopicFrames] = {}
    for topic_offset, topic in enumerate(TOPICS):
        rng = np.random.default_rng(seed * 10_000 + topic_offset)
        prob = topic.rate_hz / TICK_RATE_HZ
        emits = rng.random(num_ticks) < prob
        tick_indices = np.nonzero(emits)[0].astype(np.int64)
        timestamps_ns = tick_indices * TICK_PERIOD_NS
        sequence = np.arange(tick_indices.size, dtype=np.int64)
        payload_hash = np.array(
            [_payload_hash(topic.name, seed, int(t)) for t in tick_indices], dtype=object
        )
        result[topic.name] = TopicFrames(topic, tick_indices, timestamps_ns, sequence, payload_hash)
    return result


def _payload_hash(topic_name: str, seed: int, tick: int) -> str:
    h = hashlib.sha256(f"{topic_name}:{seed}:{tick}".encode("utf-8"))
    return h.hexdigest()[:16]


def total_frame_count(frames: dict[str, TopicFrames]) -> int:
    return sum(tf.tick_indices.size for tf in frames.values())
