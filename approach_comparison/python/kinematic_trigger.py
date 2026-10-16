"""The kinematic-trigger baseline: one simple global rule per sample,
deliberately not composable or class-specific the way the hand-written
predicates are. It decides "something happened" from a coarse kinematic
threshold, then guesses which of the 9 classes from a crude, un-tuned
heuristic rather than a per-class signature. This is what "kinematic
trigger" means in the comparison: a single trigger condition standing in
for the whole output, not nine of them.

Usage: python3 kinematic_trigger.py <corpus.tsv> <out_predictions.tsv>
"""
import sys
import time


def classify_row(distance, heading, relspeed, decel, time_gap):
    del relspeed, time_gap  # the trigger does not look at these at all
    if abs(decel) > 2.0:
        return "hard_brake" if decel < 0 else "high_acceleration"
    if heading > 60.0:
        return "crossing_paths"  # never distinguishes unprotected_left
    if distance < 8.0:
        return "close_proximity"  # lumps lane_change/merge/cut_in/cyclist/close_proximity together
    return "none"


def main():
    corpus_path, out_path = sys.argv[1], sys.argv[2]
    t0 = time.perf_counter()
    with open(corpus_path) as f, open(out_path, "w", newline="\n") as out:
        next(f)
        out.write("sample_id\tpredicted_label\n")
        n = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sample_id = int(parts[0])
            distance, heading, relspeed, decel, time_gap = (float(x) for x in parts[2:7])
            label = classify_row(distance, heading, relspeed, decel, time_gap)
            out.write(f"{sample_id}\t{label}\n")
            n += 1
    elapsed = time.perf_counter() - t0
    print(f"classified {n} rows -> {out_path} in {elapsed:.4f}s")


if __name__ == "__main__":
    main()
