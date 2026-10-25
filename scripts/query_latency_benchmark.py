"""Measures SQL query latency against the full-scale segment index: for each of the 10 scenario

tags, query by (scenario_tag, time range covering the whole corpus) and time it. Reports mean,
p50, p99 and max across all queries.

Usage: python3 scripts/query_latency_benchmark.py --out-dir data/capture --repeats 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.segment_index import open_index, timed_query_segments
from capture.segment_writer import DEFAULT_SCENARIO_TAGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/capture")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    db_path = Path(args.out_dir) / "segment_index.sqlite"
    conn = open_index(db_path)

    latencies_ms: list[float] = []
    total_matches = 0
    for _ in range(args.repeats):
        for scenario_tag in DEFAULT_SCENARIO_TAGS:
            matches, elapsed_ms = timed_query_segments(conn, scenario_tag, 0, 10**18)
            latencies_ms.append(elapsed_ms)
            total_matches += len(matches)

    arr = np.array(latencies_ms)
    summary = {
        "num_queries": len(latencies_ms),
        "total_matches_returned": total_matches,
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "max_ms": round(float(arr.max()), 4),
    }
    print(json.dumps(summary, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
