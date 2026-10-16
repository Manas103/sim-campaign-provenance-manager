#pragma once

#include <map>
#include <string>
#include <vector>

#include "approach_comparison/types.hpp"

namespace approach_comparison {

struct RegressionReport {
    int num_changed = 0;
    std::map<std::string, int> changed_counts_by_true_label;

    bool regression_detected() const { return num_changed > 0; }
    // The true-label class with the most changed predictions, or "" if none
    // changed. This is "the failing event class named" in the claim.
    std::string worst_affected_class() const;
};

// Compares two prediction runs over the same corpus (matched by
// sample_id) and reports, per true label, how many predictions differ
// between them. rows supplies the true_label lookup; neither prediction
// list needs to be sorted or complete relative to the other.
RegressionReport compare_predictions(const std::vector<FeatureRow>& rows, const std::vector<Prediction>& baseline,
                                      const std::vector<Prediction>& current);

} // namespace approach_comparison
