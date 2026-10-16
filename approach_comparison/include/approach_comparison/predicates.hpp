#pragma once

#include <array>
#include <string>

#include "approach_comparison/types.hpp"

namespace approach_comparison {

// 24 named numeric knobs, flat in one array so a "seeded regression" can
// be expressed as "perturb index i by its delta" without a bespoke mutator
// per field. This is the whole surface the regression gate's 24 scripted
// mutations exercise.
enum ThresholdField : int {
    kHardBrakeDecelMax = 0,
    kHardBrakeDistanceMin,
    kHighAccelMin,
    kHighAccelDistanceMin,
    kCyclistDistanceMax,
    kCyclistHeadingMax,
    kCyclistRelspeedAbsMax,
    kCyclistTimeGapMin,
    kCloseProximityDistanceMax,
    kCloseProximityHeadingMax,
    kCrossingHeadingMin,
    kCrossingTimeGapMax,
    kCrossingRelspeedSplit,
    kCutInDistanceMax,
    kCutInHeadingMax,
    kCutInRelspeedMax,
    kCutInTimeGapMin,
    kLaneChangeDistanceMax,
    kLaneChangeHeadingMax,
    kLaneChangeRelspeedMax,
    kMergeDistanceMax,
    kMergeHeadingMax,
    kMergeRelspeedMin,
    kMergeRelspeedMax,
    kNumThresholdFields,
};

using Thresholds = std::array<double, kNumThresholdFields>;

Thresholds default_thresholds();

// Perturbs field mutation_id (0..23) by a fixed, field-appropriate delta.
// Used only by the regression-seeding harness; production code always
// calls classify() with default_thresholds().
Thresholds apply_mutation(Thresholds base, int mutation_id);
constexpr int kNumMutations = kNumThresholdFields;

// Hand-written, priority-ordered axis-aligned rules over the five
// features; checked narrowest/most-distinctive class first so an
// ambiguous row resolves to whichever class's signature it matches best,
// not whichever happens to be checked last. Returns "none" if nothing
// matches. Never reads row.true_label.
std::string classify(const FeatureRow& row, const Thresholds& th);

} // namespace approach_comparison
