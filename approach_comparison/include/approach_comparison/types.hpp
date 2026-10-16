#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace approach_comparison {

// One tabular feature row: the agent-pair-timestep-window vocabulary
// shared with radar-plot-to-track-processor's autolabel/ extension and
// ground-truth-labeling-autolabeler-eval's generator, minus the
// trajectory simulation (see python/corpus.py's docstring for why this
// corpus starts from feature vectors directly).
struct FeatureRow {
    int sample_id = 0;
    std::string true_label; // ground truth; the predicates below never read this field
    double distance_m = 0.0;
    double heading_diff_deg = 0.0;
    double relative_speed_mps = 0.0;
    double decel_mps2 = 0.0;
    double time_gap_s = 0.0;
};

// Reads the header line "sample_id\ttrue_label\tdistance_m\t..." then one
// row per line.
std::vector<FeatureRow> read_corpus_tsv(std::istream& is);
std::vector<FeatureRow> read_corpus_tsv_file(const std::string& path);

struct Prediction {
    int sample_id = 0;
    std::string predicted_label;
};

void write_predictions_tsv(std::ostream& os, const std::vector<Prediction>& predictions);
std::vector<Prediction> read_predictions_tsv_file(const std::string& path);

} // namespace approach_comparison
