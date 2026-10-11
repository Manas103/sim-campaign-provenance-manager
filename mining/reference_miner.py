"""Independent, pure-numpy reference oracle for mine_frames.cpp.

Shares no code with the C++ miner beyond the binary file format (and even
that is read by hand here, via corpus_format.py, not by calling into the
C++ binary). Implements the exact same three-step algorithm described in
mine_frames.cpp's header comment: union-of-queries candidate selection,
then per-segment greedy near-duplicate suppression in frame_id order.
Deliberately not vectorized for the dedup pass (a plain nested loop per
segment group) so this stays obviously correct rather than fast; the C++
side is what is expected to be fast (see docs/mining_benchmark_output.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mining.corpus_format import read_corpus, read_queries


def mine(corpus_path, queries_path, dedup_epsilon: float):
    frames = read_corpus(corpus_path)
    queries = read_queries(queries_path)

    is_candidate = np.zeros(len(frames), dtype=bool)
    all_features = frames["feature"].astype(np.float64)
    for q in queries:
        diff = all_features - q["feature"].astype(np.float64)
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        is_candidate |= dist_sq <= (float(q["radius"]) ** 2)

    candidate_idx = np.nonzero(is_candidate)[0]
    candidate_idx = candidate_idx[np.argsort(frames["frame_id"][candidate_idx])]

    eps_sq = dedup_epsilon * dedup_epsilon
    kept = []
    dedup_pairs = []  # (kept_frame_id, suppressed_frame_id)

    by_segment: dict[int, list[int]] = {}
    for idx in candidate_idx:
        by_segment.setdefault(int(frames["segment_id"][idx]), []).append(int(idx))

    for seg, idxs in by_segment.items():
        group_kept: list[int] = []
        for idx in idxs:  # already frame_id-ascending
            feat = all_features[idx]
            is_dup = False
            for kept_idx in group_kept:
                d = feat - all_features[kept_idx]
                if float(np.dot(d, d)) <= eps_sq:
                    dedup_pairs.append(
                        (int(frames["frame_id"][kept_idx]), int(frames["frame_id"][idx]))
                    )
                    is_dup = True
                    break
            if not is_dup:
                group_kept.append(idx)
        kept.extend(group_kept)

    kept_frame_ids = sorted(int(frames["frame_id"][i]) for i in kept)
    kept_segment_of = {int(frames["frame_id"][i]): int(frames["segment_id"][i]) for i in kept}
    return kept_frame_ids, kept_segment_of, dedup_pairs, len(candidate_idx)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--dedup-epsilon", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kept_ids, seg_of, pairs, n_candidates = mine(args.corpus, args.queries, args.dedup_epsilon)
    with open(args.out, "w") as f:
        f.write("frame_id,segment_id\n")
        for fid in kept_ids:
            f.write(f"{fid},{seg_of[fid]}\n")
    print(f"candidates={n_candidates} kept={len(kept_ids)} suppressed={len(pairs)}")
