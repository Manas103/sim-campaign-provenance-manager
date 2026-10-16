"""Per-class precision/recall for one approach's predictions against the
corpus's true labels. Shared by all three approaches (hand-written
predicates, kinematic-trigger baseline, gradient-boosted classifier) so
the comparison in compare_approaches.py is apples to apples.

Usage: python3 score_predictions.py <corpus.tsv> <predictions.tsv>
"""
import sys
from collections import defaultdict

ALL_CLASSES = [
    "lane_change", "crossing_paths", "unprotected_left", "cut_in", "hard_brake",
    "close_proximity", "merge", "high_acceleration", "cyclist_vehicle_close_approach",
]


def load_labels(corpus_path):
    labels = {}
    with open(corpus_path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            labels[int(parts[0])] = parts[1]
    return labels


def load_predictions(predictions_path):
    preds = {}
    with open(predictions_path) as f:
        next(f)
        for line in f:
            sample_id, label = line.rstrip("\n").split("\t")
            preds[int(sample_id)] = label
    return preds


def score(true_labels: dict, predictions: dict):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for sample_id, true_label in true_labels.items():
        pred = predictions.get(sample_id, "none")
        if true_label == "none":
            if pred != "none":
                fp[pred] += 1
            continue
        if pred == true_label:
            tp[true_label] += 1
        else:
            fn[true_label] += 1
            if pred != "none":
                fp[pred] += 1

    per_class = {}
    for cls in ALL_CLASSES:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        per_class[cls] = dict(precision=p, recall=r, tp=tp[cls], fp=fp[cls], fn=fn[cls])

    total_tp = sum(tp[c] for c in ALL_CLASSES)
    total_fp = sum(fp[c] for c in ALL_CLASSES)
    total_fn = sum(fn[c] for c in ALL_CLASSES)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    return per_class, dict(precision=overall_p, recall=overall_r, tp=total_tp, fp=total_fp, fn=total_fn)


def main():
    corpus_path, predictions_path = sys.argv[1], sys.argv[2]
    true_labels = load_labels(corpus_path)
    predictions = load_predictions(predictions_path)
    per_class, overall = score(true_labels, predictions)

    for cls in ALL_CLASSES:
        s = per_class[cls]
        print(f"{cls}: precision={s['precision']:.4f} recall={s['recall']:.4f} tp={s['tp']} fp={s['fp']} fn={s['fn']}")
    print(f"OVERALL: precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
          f"tp={overall['tp']} fp={overall['fp']} fn={overall['fn']}")


if __name__ == "__main__":
    main()
