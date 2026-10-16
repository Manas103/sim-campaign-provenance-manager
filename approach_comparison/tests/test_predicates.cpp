#include <gtest/gtest.h>

#include "approach_comparison/predicates.hpp"

using namespace approach_comparison;

namespace {
FeatureRow row(double distance, double heading, double relspeed, double decel, double gap) {
    FeatureRow r;
    r.distance_m = distance;
    r.heading_diff_deg = heading;
    r.relative_speed_mps = relspeed;
    r.decel_mps2 = decel;
    r.time_gap_s = gap;
    return r;
}
} // namespace

TEST(Classify, HardBrakeFiresOnSustainedDecel) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(20, 0, 0, -4.0, 8), th), "hard_brake");
}

TEST(Classify, HighAccelerationFiresOnSustainedAccel) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(20, 0, 0, 3.0, 8), th), "high_acceleration");
}

TEST(Classify, CyclistVehicleCloseApproachNeedsTightDistanceAndHeading) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(3.0, 2.0, 0.0, 0.0, 8.0), th), "cyclist_vehicle_close_approach");
}

TEST(Classify, CloseProximityCatchesWhatTheTighterCyclistGateMisses) {
    Thresholds th = default_thresholds();
    // Within cyclist's tight distance/heading gate: cyclist wins (checked first).
    EXPECT_EQ(classify(row(4.0, 0.0, 0.0, 0.0, 8.0), th), "cyclist_vehicle_close_approach");
    // Too far for the cyclist gate (distance > kCyclistDistanceMax) and
    // too wide-angle for lane_change/cut_in/merge's heading gates, but
    // still within the looser close_proximity gate: falls through.
    EXPECT_EQ(classify(row(6.0, 15.0, 0.0, 0.0, 1.0), th), "close_proximity");
}

TEST(Classify, CrossingPathsVsUnprotectedLeftSplitByRelativeSpeed) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(15.0, 90.0, 2.0, 0.0, 1.0), th), "crossing_paths");
    EXPECT_EQ(classify(row(12.0, 80.0, 1.0, 0.0, 1.2), th), "unprotected_left");
}

TEST(Classify, CutInNeedsClosingRelativeSpeed) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(6.0, 10.0, -1.5, 0.0, 6.0), th), "cut_in");
}

TEST(Classify, LaneChangeFiresOnModestLateralSignature) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(8.0, 5.0, 0.5, 0.0, 5.0), th), "lane_change");
}

TEST(Classify, FarAwaySlowRowIsNone) {
    Thresholds th = default_thresholds();
    EXPECT_EQ(classify(row(35.0, 170.0, 0.0, 0.0, 9.0), th), "none");
}

TEST(Mutation, PerturbsOnlyTheTargetedField) {
    Thresholds base = default_thresholds();
    Thresholds mutated = apply_mutation(base, kHardBrakeDecelMax);
    for (int i = 0; i < kNumThresholdFields; ++i) {
        if (i == kHardBrakeDecelMax) {
            EXPECT_NE(mutated[i], base[i]);
        } else {
            EXPECT_DOUBLE_EQ(mutated[i], base[i]);
        }
    }
}

TEST(Mutation, EveryFieldHasANonZeroDelta) {
    Thresholds base = default_thresholds();
    for (int i = 0; i < kNumMutations; ++i) {
        Thresholds mutated = apply_mutation(base, i);
        EXPECT_NE(mutated[i], base[i]) << "mutation " << i << " was a no-op";
    }
}
