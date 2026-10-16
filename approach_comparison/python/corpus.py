"""Generates the manifest-pinned 20,000-sample corpus the three mining
approaches are compared on.

Each sample is a tabular feature row standing in for one agent-pair
timestep window (distance_m, heading_diff_deg, relative_speed_mps,
decel_mps2, time_gap_s), the same five-number vocabulary
radar-plot-to-track-processor's autolabel/ extension and
ground-truth-labeling-autolabeler-eval's generator compute from full
trajectories; this corpus starts from the feature vectors directly rather
than simulating trajectories, which is what makes 20,000 samples (versus
those two projects' hundreds of segments) a Python-generation-time
non-issue. True labels are drawn from 9 named event classes plus a
dominant "none" background class, each a Gaussian (or, for "none", a
broad near-uniform) region in feature space with deliberately overlapping
boundaries between kinematically similar classes (lane_change/merge,
crossing_paths/unprotected_left, close_proximity/cyclist_vehicle_close_approach),
so no approach can be perfect by construction.

Usage: python3 corpus.py <out.tsv> <out_manifest.json> [n=20000] [seed=1]
"""
from __future__ import annotations

import hashlib
import json
import random
import sys

EVENT_CLASSES = [
    "lane_change", "crossing_paths", "unprotected_left", "cut_in", "hard_brake",
    "close_proximity", "merge", "high_acceleration", "cyclist_vehicle_close_approach",
]

# (distance_m, heading_diff_deg, relative_speed_mps, decel_mps2, time_gap_s)
CENTROIDS = {
    "lane_change": (8.0, 5.0, 0.5, 0.0, 5.0),
    "crossing_paths": (15.0, 90.0, 2.0, 0.0, 1.0),
    "unprotected_left": (12.0, 80.0, 1.0, 0.0, 1.2),
    "cut_in": (6.0, 10.0, -1.5, 0.0, 6.0),
    "hard_brake": (20.0, 0.0, 0.0, -4.0, 8.0),
    "close_proximity": (4.0, 0.0, 0.0, 0.0, 8.0),
    "merge": (7.0, 8.0, 0.3, 0.0, 5.0),
    "high_acceleration": (20.0, 0.0, 0.0, 3.0, 8.0),
    "cyclist_vehicle_close_approach": (3.0, 2.0, 0.0, 0.0, 8.0),
}
STDS = {
    "lane_change": (3, 5, 1, 1, 3),
    "crossing_paths": (5, 20, 1, 1, 0.5),
    "unprotected_left": (5, 20, 1, 1, 0.5),
    "cut_in": (2, 5, 1, 1, 3),
    "hard_brake": (8, 5, 1, 1, 4),
    "close_proximity": (1.5, 5, 1, 1, 4),
    "merge": (3, 5, 1, 1, 3),
    "high_acceleration": (8, 5, 1, 1, 4),
    "cyclist_vehicle_close_approach": (1.0, 3, 1, 1, 4),
}
NONE_WEIGHT = 0.7  # background ("nothing happened") is the majority class


def sample_row(rng: random.Random, label: str):
    # distance_m and time_gap_s are magnitudes (clamped at 0); heading_diff_deg
    # is a magnitude too (abs). relative_speed_mps and decel_mps2 are signed
    # (decel's sign is what distinguishes hard_brake, negative, from
    # high_acceleration, positive: clamping either to non-negative would
    # silently destroy hard_brake's entire signal).
    if label == "none":
        distance, heading_diff, rel_speed, decel, time_gap = (
            rng.uniform(0.0, 40.0), rng.uniform(0.0, 180.0), rng.gauss(0.0, 2.0),
            rng.gauss(0.0, 1.0), rng.uniform(0.0, 10.0),
        )
    else:
        c_dist, c_heading, c_relspeed, c_decel, c_gap = CENTROIDS[label]
        s_dist, s_heading, s_relspeed, s_decel, s_gap = STDS[label]
        distance = rng.gauss(c_dist, s_dist)
        heading_diff = abs(rng.gauss(c_heading, s_heading))
        rel_speed = rng.gauss(c_relspeed, s_relspeed)
        decel = rng.gauss(c_decel, s_decel)
        time_gap = rng.gauss(c_gap, s_gap)
    return max(0.0, distance), heading_diff, rel_speed, decel, max(0.0, time_gap)


def generate(n: int, seed: int):
    rng = random.Random(seed)
    per_class_weight = (1.0 - NONE_WEIGHT) / len(EVENT_CLASSES)
    labels = ["none"] * int(n * NONE_WEIGHT)
    for cls in EVENT_CLASSES:
        labels += [cls] * int(n * per_class_weight)
    while len(labels) < n:
        labels.append("none")
    rng.shuffle(labels)

    rows = []
    for i, label in enumerate(labels):
        distance, heading_diff, rel_speed, decel, time_gap = sample_row(rng, label)
        rows.append((i + 1, label, round(distance, 4), round(heading_diff, 4), round(rel_speed, 4),
                     round(decel, 4), round(time_gap, 4)))
    return rows


def write_corpus(rows, out_path: str):
    with open(out_path, "w", newline="\n") as f:
        f.write("sample_id\ttrue_label\tdistance_m\theading_diff_deg\trelative_speed_mps\tdecel_mps2\ttime_gap_s\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def write_manifest(out_tsv_path: str, manifest_path: str, n: int, seed: int):
    with open(out_tsv_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()
    manifest = {"n": n, "seed": seed, "generator": "approach_comparison/python/corpus.py",
                 "content_sha256": content_hash}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/approach_comparison_corpus.tsv"
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "data/approach_comparison_manifest.json"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    rows = generate(n, seed)
    write_corpus(rows, out_path)
    manifest = write_manifest(out_path, manifest_path, n, seed)

    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print(f"wrote {len(rows)} samples to {out_path}")
    print(f"manifest content_sha256={manifest['content_sha256']}")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
