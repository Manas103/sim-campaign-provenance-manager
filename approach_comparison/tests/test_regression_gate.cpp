#include <gtest/gtest.h>

#include "approach_comparison/regression_gate.hpp"

using namespace approach_comparison;

namespace {
FeatureRow make_row(int id, std::string label) {
    FeatureRow r;
    r.sample_id = id;
    r.true_label = std::move(label);
    return r;
}
} // namespace

TEST(RegressionGate, IdenticalPredictionsAreNotARegression) {
    std::vector<FeatureRow> rows = {make_row(1, "lane_change"), make_row(2, "none")};
    std::vector<Prediction> baseline = {{1, "lane_change"}, {2, "none"}};
    std::vector<Prediction> current = baseline;
    RegressionReport report = compare_predictions(rows, baseline, current);
    EXPECT_FALSE(report.regression_detected());
    EXPECT_EQ(report.num_changed, 0);
}

TEST(RegressionGate, OneChangedPredictionIsDetectedAndNamed) {
    std::vector<FeatureRow> rows = {make_row(1, "hard_brake"), make_row(2, "none")};
    std::vector<Prediction> baseline = {{1, "hard_brake"}, {2, "none"}};
    std::vector<Prediction> current = {{1, "none"}, {2, "none"}};
    RegressionReport report = compare_predictions(rows, baseline, current);
    EXPECT_TRUE(report.regression_detected());
    EXPECT_EQ(report.num_changed, 1);
    EXPECT_EQ(report.worst_affected_class(), "hard_brake");
}

TEST(RegressionGate, NamesTheMostAffectedClassWhenSeveralChange) {
    std::vector<FeatureRow> rows = {make_row(1, "merge"), make_row(2, "merge"), make_row(3, "cut_in")};
    std::vector<Prediction> baseline = {{1, "merge"}, {2, "merge"}, {3, "cut_in"}};
    std::vector<Prediction> current = {{1, "lane_change"}, {2, "lane_change"}, {3, "none"}};
    RegressionReport report = compare_predictions(rows, baseline, current);
    EXPECT_EQ(report.num_changed, 3);
    EXPECT_EQ(report.worst_affected_class(), "merge");
    EXPECT_EQ(report.changed_counts_by_true_label.at("merge"), 2);
    EXPECT_EQ(report.changed_counts_by_true_label.at("cut_in"), 1);
}

TEST(RegressionGate, EmptyCorpusIsNotARegression) {
    RegressionReport report = compare_predictions({}, {}, {});
    EXPECT_FALSE(report.regression_detected());
    EXPECT_EQ(report.worst_affected_class(), "");
}
