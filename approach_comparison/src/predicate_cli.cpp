// predicate_cli: the hand-written-predicate mining approach, plus the
// regression gate's comparison primitive.
//
// Subcommands:
//   run <corpus.tsv> <out_predictions.tsv> [mutation_id=-1]
//     Classifies every row with default_thresholds() (or, if
//     mutation_id >= 0, a single perturbed threshold field): the seeded-
//     regression harness's one building block.
//   diff <corpus.tsv> <baseline_predictions.tsv> <current_predictions.tsv>
//     Compares two prediction runs over the same corpus; prints the
//     changed-sample count and, if any, the most-affected true class by
//     name. Exit 0 if identical, 1 if a regression is detected.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "approach_comparison/predicates.hpp"
#include "approach_comparison/regression_gate.hpp"
#include "approach_comparison/types.hpp"

namespace {

int cmd_run(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: predicate_cli run <corpus.tsv> <out_predictions.tsv> [mutation_id=-1]\n";
        return 2;
    }
    auto rows = approach_comparison::read_corpus_tsv_file(argv[2]);
    int mutation_id = argc > 4 ? std::stoi(argv[4]) : -1;

    approach_comparison::Thresholds th = approach_comparison::default_thresholds();
    if (mutation_id >= 0) th = approach_comparison::apply_mutation(th, mutation_id);

    std::vector<approach_comparison::Prediction> predictions;
    predictions.reserve(rows.size());
    for (const auto& row : rows) {
        predictions.push_back({row.sample_id, approach_comparison::classify(row, th)});
    }

    std::ofstream out(argv[3], std::ios::trunc);
    approach_comparison::write_predictions_tsv(out, predictions);
    std::cout << "classified " << predictions.size() << " rows -> " << argv[3]
               << (mutation_id >= 0 ? (" (mutation_id=" + std::to_string(mutation_id) + ")") : "") << "\n";
    return 0;
}

int cmd_diff(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: predicate_cli diff <corpus.tsv> <baseline_predictions.tsv> <current_predictions.tsv>\n";
        return 2;
    }
    auto rows = approach_comparison::read_corpus_tsv_file(argv[2]);
    auto baseline = approach_comparison::read_predictions_tsv_file(argv[3]);
    auto current = approach_comparison::read_predictions_tsv_file(argv[4]);

    approach_comparison::RegressionReport report = approach_comparison::compare_predictions(rows, baseline, current);
    std::cout << "changed_samples=" << report.num_changed << "\n";
    for (const auto& [cls, count] : report.changed_counts_by_true_label) {
        std::cout << "  " << cls << ": " << count << " changed\n";
    }
    if (report.regression_detected()) {
        std::cout << "REGRESSION_DETECTED=true FAILING_EVENT_CLASS=" << report.worst_affected_class() << "\n";
    } else {
        std::cout << "REGRESSION_DETECTED=false\n";
    }
    return report.regression_detected() ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: predicate_cli <run|diff> ...\n";
        return 2;
    }
    std::string cmd = argv[1];
    if (cmd == "run") return cmd_run(argc, argv);
    if (cmd == "diff") return cmd_diff(argc, argv);
    std::cerr << "unknown subcommand: " << cmd << "\n";
    return 2;
}
