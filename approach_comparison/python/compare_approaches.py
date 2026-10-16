"""Orchestrates the three-approach comparison on one manifest-pinned
20,000-sample corpus: hand-written predicates (C++, via predicate_cli),
the kinematic-trigger baseline (Python), and a gradient-boosted classifier
(scikit-learn). All three are scored on the same held-out test split (30%
of the corpus, GBM's own held-out set) so the ML approach is not compared
against its own training rows while the rule-based approaches are
compared against all rows: that would not be apples to apples.

Usage: python3 compare_approaches.py <corpus.tsv> <predicate_cli_path> <out_report.json>
"""
import json
import os
import subprocess
import sys
import time

from kinematic_trigger import classify_row as kinematic_classify_row
from score_predictions import score


def load_rows(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((int(parts[0]), parts[1], *(float(x) for x in parts[2:7])))
    return header, rows


def write_rows(path, header, rows):
    with open(path, "w", newline="\n") as f:
        f.write(header if header.endswith("\n") else header + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def split_train_test(rows, test_fraction=0.3, seed=1):
    import random
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    return shuffled[n_test:], shuffled[:n_test]


def run_gbm(train_rows, test_rows):
    from sklearn.ensemble import GradientBoostingClassifier

    x_train = [r[2:7] for r in train_rows]
    y_train = [r[1] for r in train_rows]
    x_test = [r[2:7] for r in test_rows]

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=1)
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    train_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(x_test)
    predict_elapsed = time.perf_counter() - t0

    predictions = {row[0]: label for row, label in zip(test_rows, y_pred)}
    return predictions, train_elapsed, predict_elapsed


def run_kinematic_trigger(test_rows):
    t0 = time.perf_counter()
    predictions = {row[0]: kinematic_classify_row(*row[2:7]) for row in test_rows}
    elapsed = time.perf_counter() - t0
    return predictions, elapsed


def run_predicates(predicate_cli_path, test_tsv_path, out_predictions_path):
    t0 = time.perf_counter()
    subprocess.run([predicate_cli_path, "run", test_tsv_path, out_predictions_path], check=True,
                    capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    predictions = {}
    with open(out_predictions_path) as f:
        next(f)
        for line in f:
            sample_id, label = line.rstrip("\n").split("\t")
            predictions[int(sample_id)] = label
    return predictions, elapsed


def predictions_to_true_label_dict(rows):
    return {r[0]: r[1] for r in rows}


def main():
    corpus_path, predicate_cli_path, report_path = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
    header, rows = load_rows(corpus_path)
    train_rows, test_rows = split_train_test(rows)

    test_tsv_path = "data/approach_comparison_test_split.tsv"
    write_rows(test_tsv_path, header, test_rows)
    true_labels = predictions_to_true_label_dict(test_rows)

    predicate_preds, predicate_time = run_predicates(predicate_cli_path, test_tsv_path,
                                                        "data/predictions_predicates_test.tsv")
    trigger_preds, trigger_time = run_kinematic_trigger(test_rows)
    gbm_preds, gbm_train_time, gbm_predict_time = run_gbm(train_rows, test_rows)

    results = {}
    for name, preds, compute_time in (
        ("hand_written_predicates", predicate_preds, predicate_time),
        ("kinematic_trigger", trigger_preds, trigger_time),
        ("gradient_boosted_classifier", gbm_preds, gbm_train_time + gbm_predict_time),
    ):
        per_class, overall = score(true_labels, preds)
        results[name] = dict(overall=overall, per_class=per_class, compute_time_s=compute_time)

    predicate_time_s = results["hand_written_predicates"]["compute_time_s"]
    gbm_time_s = results["gradient_boosted_classifier"]["compute_time_s"]
    compute_ratio = gbm_time_s / predicate_time_s if predicate_time_s > 0 else float("inf")

    print(f"test_split_size={len(test_rows)} train_split_size={len(train_rows)}")
    for name, r in results.items():
        o = r["overall"]
        print(f"{name}: precision={o['precision']:.4f} recall={o['recall']:.4f} compute_time_s={r['compute_time_s']:.4f}")
    print(f"GBM_TRAIN_TIME_S={gbm_train_time:.4f} GBM_PREDICT_TIME_S={gbm_predict_time:.4f}")
    print(f"COMPUTE_RATIO_GBM_OVER_PREDICATES={compute_ratio:.2f}x")

    with open(report_path, "w") as f:
        json.dump({"test_split_size": len(test_rows), "train_split_size": len(train_rows),
                    "results": results, "compute_ratio_gbm_over_predicates": compute_ratio,
                    "gbm_train_time_s": gbm_train_time, "gbm_predict_time_s": gbm_predict_time}, f, indent=2)


if __name__ == "__main__":
    main()
