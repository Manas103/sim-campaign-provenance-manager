"""The 12 simulated robot sensor topics and their nominal publish rates.

Rates are in Hz against a 100 Hz capture tick (see stream_simulator.py): a topic at 30 Hz is
expected to emit on roughly 30 of every 100 ticks, decided by an independent Bernoulli draw per
tick, not a fixed stride, so two topics at the same rate do not emit in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass

TICK_RATE_HZ = 100


@dataclass(frozen=True)
class Topic:
    name: str
    rate_hz: float
    payload_bytes: int


TOPICS: tuple[Topic, ...] = (
    Topic("camera_front", 30.0, 48),
    Topic("camera_rear", 30.0, 48),
    Topic("camera_left", 10.0, 48),
    Topic("camera_right", 10.0, 48),
    Topic("lidar_top", 10.0, 64),
    Topic("radar_front", 20.0, 32),
    Topic("radar_rear", 20.0, 32),
    Topic("imu", 100.0, 16),
    Topic("gps", 10.0, 24),
    Topic("wheel_odometry", 50.0, 16),
    Topic("ultrasonic_array", 40.0, 24),
    Topic("battery_status", 1.0, 8),
)

assert len(TOPICS) == 12
TOPIC_NAMES: tuple[str, ...] = tuple(t.name for t in TOPICS)
