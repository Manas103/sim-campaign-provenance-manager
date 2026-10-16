#include "approach_comparison/predicates.hpp"

#include <cmath>

namespace approach_comparison {

Thresholds default_thresholds() {
    Thresholds t{};
    t[kHardBrakeDecelMax] = -2.5;
    t[kHardBrakeDistanceMin] = 10.0;
    t[kHighAccelMin] = 1.8;
    t[kHighAccelDistanceMin] = 10.0;
    t[kCyclistDistanceMax] = 4.5;
    t[kCyclistHeadingMax] = 10.0;
    t[kCyclistRelspeedAbsMax] = 1.0;
    t[kCyclistTimeGapMin] = 3.0;
    t[kCloseProximityDistanceMax] = 6.5;
    t[kCloseProximityHeadingMax] = 20.0;
    t[kCrossingHeadingMin] = 50.0;
    t[kCrossingTimeGapMax] = 2.0;
    t[kCrossingRelspeedSplit] = 1.5;
    t[kCutInDistanceMax] = 9.0;
    t[kCutInHeadingMax] = 13.0;
    t[kCutInRelspeedMax] = -0.3;
    t[kCutInTimeGapMin] = 3.0;
    t[kLaneChangeDistanceMax] = 11.0;
    t[kLaneChangeHeadingMax] = 8.0;
    t[kLaneChangeRelspeedMax] = 1.0;
    t[kMergeDistanceMax] = 9.0;
    t[kMergeHeadingMax] = 9.5;
    t[kMergeRelspeedMin] = -0.3;
    t[kMergeRelspeedMax] = 0.8;
    return t;
}

namespace {
// One signed delta per field, scaled to roughly a quarter of that
// feature's typical spread: large enough that a mutation has a real
// chance of changing at least one of 20,000 samples' classification,
// small enough that it reads as a plausible accidental edit, not a
// wholesale rewrite.
constexpr std::array<double, kNumThresholdFields> kMutationDeltas = {
    -0.8, 2.0, 0.6, 2.0, 1.2, 3.0, 0.4, 1.0, 1.5, 4.0, 10.0, 0.5, 0.4, 1.5, 3.0, 0.3, 1.0, 1.5, 2.0, 0.4, 1.5, 2.0, 0.3, 0.3,
};
} // namespace

Thresholds apply_mutation(Thresholds base, int mutation_id) {
    base[mutation_id] += kMutationDeltas[mutation_id];
    return base;
}

std::string classify(const FeatureRow& row, const Thresholds& th) {
    const double d = row.distance_m;
    const double h = row.heading_diff_deg;
    const double v = row.relative_speed_mps;
    const double a = row.decel_mps2;
    const double g = row.time_gap_s;

    // Checked narrowest-signature-first; close_proximity is deliberately
    // last among the "something is nearby" rules, a catch-all for rows
    // that are close but match no other class's more specific shape
    // (otherwise it would shadow cut_in and cyclist rows that are close
    // for a reason close_proximity alone does not capture).
    if (a <= th[kHardBrakeDecelMax] && d >= th[kHardBrakeDistanceMin]) return "hard_brake";
    if (a >= th[kHighAccelMin] && d >= th[kHighAccelDistanceMin]) return "high_acceleration";
    if (d <= th[kCyclistDistanceMax] && h <= th[kCyclistHeadingMax] && std::abs(v) <= th[kCyclistRelspeedAbsMax] &&
        g >= th[kCyclistTimeGapMin])
        return "cyclist_vehicle_close_approach";
    if (h >= th[kCrossingHeadingMin] && g <= th[kCrossingTimeGapMax]) {
        return v >= th[kCrossingRelspeedSplit] ? "crossing_paths" : "unprotected_left";
    }
    if (d <= th[kCutInDistanceMax] && h <= th[kCutInHeadingMax] && v <= th[kCutInRelspeedMax] &&
        g >= th[kCutInTimeGapMin])
        return "cut_in";
    if (d <= th[kLaneChangeDistanceMax] && h <= th[kLaneChangeHeadingMax] && v <= th[kLaneChangeRelspeedMax])
        return "lane_change";
    if (d <= th[kMergeDistanceMax] && h <= th[kMergeHeadingMax] && v >= th[kMergeRelspeedMin] &&
        v <= th[kMergeRelspeedMax])
        return "merge";
    if (d <= th[kCloseProximityDistanceMax] && h <= th[kCloseProximityHeadingMax]) return "close_proximity";
    return "none";
}

} // namespace approach_comparison
