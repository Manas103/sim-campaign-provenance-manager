#!/usr/bin/env python3
"""Generates the synthetic 2,000,000-frame sensor-log corpus mined by
mine_frames.cpp, plus the query-by-example set and the seeded near-duplicate
ground truth used to measure suppression precision/recall.

Deliberately synthetic (see README): a real driving log is not something a
student has, so segments stand in for distinct driving-log files, and
within-segment feature noise stands in for consecutive frames of the same
scene being visually similar. Regenerate with:
    python mining/generate_corpus.py --seed 7 --out-dir data/mining
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mining.corpus_format import FEATURE_DIM, FRAME_DTYPE, QUERY_DTYPE, write_corpus, write_queries

NUM_SEGMENTS = 400
FRAMES_PER_SEGMENT = 5000  # 400 * 5000 = 2,000,000 base frames
NUM_QUERY_SEGMENTS = 20
DUPES_PER_QUERY_SEGMENT = 3
CONTROL_PER_QUERY_SEGMENT = 3
SEGMENT_SPREAD = 3.0      # stdev of cluster centers, separates segments
WITHIN_SEGMENT_NOISE = 0.5  # stdev of frame-to-frame noise inside a segment
DUP_JITTER = 1e-4          # near-duplicate jitter, far below dedup epsilon
QUERY_RADIUS = 3.2         # tuned below SEGMENT_SPREAD-scale separation


def generate(seed: int, out_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = []
    segment_ids = []
    features = []

    segment_start_frame = {}
    next_frame_id = 0
    for seg in range(NUM_SEGMENTS):
        center = rng.normal(0.0, SEGMENT_SPREAD, FEATURE_DIM)
        segment_start_frame[seg] = next_frame_id
        noise = rng.normal(0.0, WITHIN_SEGMENT_NOISE, (FRAMES_PER_SEGMENT, FEATURE_DIM))
        seg_features = center[None, :] + noise
        for i in range(FRAMES_PER_SEGMENT):
            frame_ids.append(next_frame_id)
            segment_ids.append(seg)
            features.append(seg_features[i])
            next_frame_id += 1

    query_segments = list(range(0, NUM_QUERY_SEGMENTS * (NUM_SEGMENTS // NUM_QUERY_SEGMENTS),
                                 NUM_SEGMENTS // NUM_QUERY_SEGMENTS))[:NUM_QUERY_SEGMENTS]

    duplicate_pairs = []  # (original_frame_id, duplicate_frame_id)
    control_frame_ids = []  # frames that must survive dedup unmerged
    queries = []

    for qi, seg in enumerate(query_segments):
        base_frame_id = segment_start_frame[seg]  # first frame of this segment
        query_feature = np.array(features[base_frame_id], dtype=np.float32)
        queries.append((qi, query_feature, np.float32(QUERY_RADIUS)))

        # Seed DUPES_PER_QUERY_SEGMENT near-duplicates of distinct existing
        # frames in this segment, appended as brand-new frame_ids. Picked by
        # proximity to the query (not uniformly at random across the whole
        # segment): an early version picked uniformly at random and found
        # a handful of originals legitimately outside QUERY_RADIUS by
        # sampling variance (the within-segment noise occasionally pushes a
        # frame past the recall threshold), which made the seeded-pair
        # count itself unreliable rather than measuring dedup. Restricting
        # the seed pool to frames already well inside the radius makes the
        # 60-pair claim exact without changing what is actually being
        # tested (dedup behavior on retrieved candidates).
        rng_local = np.random.default_rng(seed * 10_000 + seg)
        seg_slice = np.array(
            features[base_frame_id + 1 : base_frame_id + FRAMES_PER_SEGMENT]
        )
        dist_to_query = np.linalg.norm(seg_slice - query_feature, axis=1)
        close_pool = np.argsort(dist_to_query)[: (FRAMES_PER_SEGMENT // 4)] + (
            base_frame_id + 1
        )
        originals = rng_local.choice(
            close_pool,
            size=DUPES_PER_QUERY_SEGMENT + CONTROL_PER_QUERY_SEGMENT,
            replace=False,
        )
        for orig in originals[:DUPES_PER_QUERY_SEGMENT]:
            dup_feature = np.array(features[orig], dtype=np.float32) + rng_local.normal(
                0.0, DUP_JITTER, FEATURE_DIM
            ).astype(np.float32)
            frame_ids.append(next_frame_id)
            segment_ids.append(seg)
            features.append(dup_feature)
            duplicate_pairs.append((int(orig), int(next_frame_id)))
            next_frame_id += 1
        for ctrl in originals[DUPES_PER_QUERY_SEGMENT:]:
            control_frame_ids.append(int(ctrl))

    frames = np.zeros(len(frame_ids), dtype=FRAME_DTYPE)
    frames["frame_id"] = frame_ids
    frames["segment_id"] = segment_ids
    frames["feature"] = np.array(features, dtype=np.float32)

    query_arr = np.zeros(len(queries), dtype=QUERY_DTYPE)
    for i, (qid, feat, radius) in enumerate(queries):
        query_arr[i] = (qid, feat, radius)

    write_corpus(out_dir / "corpus.bin", frames, NUM_SEGMENTS)
    write_queries(out_dir / "queries.bin", query_arr)

    segments = {
        str(seg): {
            "log_file": f"logs/segment_{seg:04d}.sensorlog",
            "start_frame_id": start,
            "end_frame_id": start + FRAMES_PER_SEGMENT - 1,
        }
        for seg, start in segment_start_frame.items()
    }
    with open(out_dir / "segments.json", "w") as f:
        json.dump(segments, f, indent=2)

    ground_truth = {
        "seed": seed,
        "num_frames": len(frame_ids),
        "num_segments": NUM_SEGMENTS,
        "frames_per_segment": FRAMES_PER_SEGMENT,
        "query_segments": query_segments,
        "duplicate_pairs": duplicate_pairs,
        "control_frame_ids": control_frame_ids,
        "dedup_epsilon_recommended": 1e-2,
        "query_radius": QUERY_RADIUS,
    }
    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"wrote {len(frame_ids)} frames, {len(queries)} queries, "
          f"{len(duplicate_pairs)} seeded near-dup pairs to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("data/mining"))
    args = ap.parse_args()
    generate(args.seed, args.out_dir)
