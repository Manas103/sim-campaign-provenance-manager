#include "approach_comparison/regression_gate.hpp"

#include <unordered_map>

namespace approach_comparison {

std::string RegressionReport::worst_affected_class() const {
    std::string worst;
    int worst_count = 0;
    for (const auto& [cls, count] : changed_counts_by_true_label) {
        if (count > worst_count) {
            worst = cls;
            worst_count = count;
        }
    }
    return worst;
}

RegressionReport compare_predictions(const std::vector<FeatureRow>& rows, const std::vector<Prediction>& baseline,
                                      const std::vector<Prediction>& current) {
    std::unordered_map<int, std::string> baseline_by_id, current_by_id;
    for (const auto& p : baseline) baseline_by_id[p.sample_id] = p.predicted_label;
    for (const auto& p : current) current_by_id[p.sample_id] = p.predicted_label;

    RegressionReport report;
    for (const auto& row : rows) {
        auto bit = baseline_by_id.find(row.sample_id);
        auto cit = current_by_id.find(row.sample_id);
        const std::string& b = bit != baseline_by_id.end() ? bit->second : std::string("<missing>");
        const std::string& c = cit != current_by_id.end() ? cit->second : std::string("<missing>");
        if (b != c) {
            report.num_changed++;
            report.changed_counts_by_true_label[row.true_label]++;
        }
    }
    return report;
}

} // namespace approach_comparison
