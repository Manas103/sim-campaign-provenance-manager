"""Runs the three replay-regression-gate measurements against the 40-scenario gate corpus:

determinism across many replays, regression catching across 40 seeded one-scenario regressions,
and false-trip rate across many unchanged reruns.

Usage: python3 scripts/replay_regression_gate_run.py --out-dir data/capture/gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.replay_gate import (
    build_gate_corpus,
    check_determinism,
    check_false_trips,
    check_regression_catching,
)

DETERMINISM_REPLAYS = 500
FALSE_TRIP_RERUNS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/capture/gate")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    segments = build_gate_corpus(out_dir, seed=args.seed)

    matching, num_replays, reference_combined_digest = check_determinism(segments, DETERMINISM_REPLAYS)

    from capture.perception_stack import process_segment, digest_of_result

    reference_digests = {
        seg.scenario_tag: digest_of_result(process_segment(seg.path, seg.scenario_tag, broken_scenario=None))
        for seg in segments
    }

    regression_results = check_regression_catching(segments, reference_digests)
    num_caught = sum(1 for r in regression_results if r.caught)
    uncaught = [r.broken_scenario for r in regression_results if not r.caught]

    false_trips, num_reruns = check_false_trips(segments, reference_digests, FALSE_TRIP_RERUNS)

    summary = {
        "num_gate_scenarios": len(segments),
        "determinism": {
            "matching_replays": matching,
            "total_replays": num_replays,
            "reference_combined_digest": reference_combined_digest,
        },
        "regression_catching": {
            "num_seeded_regressions": len(regression_results),
            "num_caught": num_caught,
            "uncaught_scenarios": uncaught,
            "named_scenarios": [r.named_scenario for r in regression_results if r.caught],
        },
        "false_trips": {
            "num_false_tripped": false_trips,
            "num_reruns": num_reruns,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
