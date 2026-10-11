"""The 2,000,000-frame mining run this README's Measured Results table quotes.

Run with (after `mining/build.sh`):
    python3 scripts/mining_full_scale_run.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign_manager import storage
from mining import lineage, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
MINING_DIR = REPO_ROOT / "data" / "mining"
MINER_BINARY = REPO_ROOT / "mining" / "mine_frames"
DEDUP_EPSILON = 1e-2


def main() -> None:
    if not MINER_BINARY.exists():
        raise SystemExit("mining/mine_frames not built; run mining/build.sh first")

    ground_truth = json.loads((MINING_DIR / "ground_truth.json").read_text())
    scratch = MINING_DIR / "scratch"
    scratch.mkdir(exist_ok=True)

    db_path = MINING_DIR / "mining_run.sqlite"
    db_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    with storage.connect(db_path) as conn:
        decision, selected, miner_stderr = pipeline.run_mining_case(
            conn,
            MINING_DIR / "runs",
            MINER_BINARY,
            MINING_DIR / "corpus.bin",
            MINING_DIR / "queries.bin",
            DEDUP_EPSILON,
            scratch,
            run_id="full_scale_run",
        )
    elapsed = time.perf_counter() - t0

    kept_frame_ids = {int(row[0]) for row in selected}
    kept_pairs = {(int(r[0]), int(r[1])) for r in selected}

    duplicate_frame_ids = {dup for _orig, dup in ground_truth["duplicate_pairs"]}
    suppressed = duplicate_frame_ids - kept_frame_ids
    control_frame_ids = set(ground_truth["control_frame_ids"])
    falsely_suppressed = control_frame_ids - kept_frame_ids

    index = lineage.SegmentIndex(MINING_DIR / "segments.json")
    resolved = index.resolve_all(kept_pairs)
    base_corpus_size = ground_truth["num_segments"] * ground_truth["frames_per_segment"]
    traceable = sum(
        1
        for entry in resolved
        if entry.offset_in_segment is not None or entry.frame_id >= base_corpus_size
    )

    report = {
        "corpus_frames": ground_truth["num_frames"],
        "candidates_recalled": None,  # parsed from miner stderr below
        "kept_after_dedup": len(selected),
        "seeded_duplicates_total": len(duplicate_frame_ids),
        "seeded_duplicates_suppressed": len(suppressed),
        "seeded_duplicates_missed": len(duplicate_frame_ids) - len(suppressed),
        "control_frames_total": len(control_frame_ids),
        "control_frames_falsely_suppressed": len(falsely_suppressed),
        "frames_with_resolvable_lineage": traceable,
        "frames_total_checked": len(resolved),
        "gate_decision": decision.reason,
        "gate_accepted": decision.accepted,
        "wall_seconds": round(elapsed, 3),
    }
    for line in miner_stderr.splitlines():
        if line.startswith("corpus_frames="):
            for token in line.split():
                k, v = token.split("=")
                if k == "candidates":
                    report["candidates_recalled"] = int(v)

    print(json.dumps(report, indent=2))

    with open(MINING_DIR / "cpp_out_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "segment_id"])
        w.writerows(sorted(kept_pairs))


if __name__ == "__main__":
    main()
