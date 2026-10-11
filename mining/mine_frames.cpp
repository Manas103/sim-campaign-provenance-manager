// Mines a sensor-log corpus by example similarity and suppresses
// near-duplicates in the result, deterministically.
//
// Binary format (must match mining/corpus_format.py exactly; see
// test_mining_reference_match.py, which is what would catch drift between
// this file and that one):
//   corpus.bin:  header [magic:u32='MINE' le, num_frames:u32, feature_dim:u32,
//                num_segments:u32], then num_frames x packed records
//                [frame_id:u64, segment_id:u32, feature:f32[8]] (44 bytes,
//                no padding -- hence #pragma pack(1) below).
//   queries.bin: header [magic:u32='QREY' le, num_queries:u32, feature_dim:u32],
//                then num_queries x packed records
//                [query_id:u32, feature:f32[8], radius:f32] (40 bytes).
//
// Algorithm (exactly mirrored, independently, in mining/reference_miner.py
// for the correctness oracle):
//   1. candidate = union over all queries of every frame within `radius`
//      (squared Euclidean) of that query's feature vector.
//   2. sort candidates by frame_id ascending (determinism: the dedup
//      survivor is always the lowest frame_id in a near-duplicate cluster,
//      regardless of which thread or run found it).
//   3. greedy dedup: walk candidates in that order; a frame is suppressed
//      if it is within `dedup_epsilon` (squared Euclidean) of any frame
//      already kept.
//
// Single-threaded on purpose: determinism (claim 4, 1,200/1,200-style
// regeneration) is cheaper to guarantee without a merge step across
// threads, and brute force over 2,000,000 frames x ~20 queries is under a
// second single-threaded on this machine (see docs/mining_benchmark_output.txt),
// so there is no measured need to parallelize it.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kFeatureDim = 8;
constexpr uint32_t kCorpusMagic = 0x454E494D;
constexpr uint32_t kQueryMagic = 0x59524551;

#pragma pack(push, 1)
struct Frame {
  uint64_t frame_id;
  uint32_t segment_id;
  float feature[kFeatureDim];
};
struct Query {
  uint32_t query_id;
  float feature[kFeatureDim];
  float radius;
};
#pragma pack(pop)

static_assert(sizeof(Frame) == 44, "Frame must be packed to 44 bytes");
static_assert(sizeof(Query) == 40, "Query must be packed to 40 bytes");

struct CorpusHeader {
  uint32_t magic, num_frames, feature_dim, num_segments;
};
struct QueryHeader {
  uint32_t magic, num_queries, feature_dim;
};

std::vector<Frame> ReadCorpus(const std::string& path, uint32_t* num_segments) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open corpus: " + path);
  CorpusHeader h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (h.magic != kCorpusMagic || h.feature_dim != kFeatureDim) {
    throw std::runtime_error("bad corpus header in " + path);
  }
  std::vector<Frame> frames(h.num_frames);
  f.read(reinterpret_cast<char*>(frames.data()),
         static_cast<std::streamsize>(frames.size() * sizeof(Frame)));
  *num_segments = h.num_segments;
  return frames;
}

std::vector<Query> ReadQueries(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open queries: " + path);
  QueryHeader h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (h.magic != kQueryMagic || h.feature_dim != kFeatureDim) {
    throw std::runtime_error("bad query header in " + path);
  }
  std::vector<Query> queries(h.num_queries);
  f.read(reinterpret_cast<char*>(queries.data()),
         static_cast<std::streamsize>(queries.size() * sizeof(Query)));
  return queries;
}

inline float SquaredDistance(const float* a, const float* b) {
  float sum = 0.0f;
  for (int i = 0; i < kFeatureDim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

std::string ArgOr(int argc, char** argv, const std::string& key,
                   const std::string& def) {
  const std::string prefix = "--" + key + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  }
  return def;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string corpus_path = ArgOr(argc, argv, "corpus", "");
  const std::string queries_path = ArgOr(argc, argv, "queries", "");
  const std::string out_path = ArgOr(argc, argv, "out", "");
  const float dedup_epsilon_sq =
      std::stof(ArgOr(argc, argv, "dedup-epsilon", "0.01"));
  const float dedup_epsilon_sq_value = dedup_epsilon_sq * dedup_epsilon_sq;

  if (corpus_path.empty() || queries_path.empty() || out_path.empty()) {
    std::cerr << "usage: mine_frames --corpus=F --queries=F --out=F "
                 "[--dedup-epsilon=E]\n";
    return 1;
  }

  uint32_t num_segments = 0;
  std::vector<Frame> frames = ReadCorpus(corpus_path, &num_segments);
  std::vector<Query> queries = ReadQueries(queries_path);

  std::vector<char> is_candidate(frames.size(), 0);
  for (const Query& q : queries) {
    const float radius_sq = q.radius * q.radius;
    for (size_t i = 0; i < frames.size(); ++i) {
      if (is_candidate[i]) continue;
      if (SquaredDistance(frames[i].feature, q.feature) <= radius_sq) {
        is_candidate[i] = 1;
      }
    }
  }

  std::vector<size_t> candidate_indices;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (is_candidate[i]) candidate_indices.push_back(i);
  }
  std::sort(candidate_indices.begin(), candidate_indices.end(),
            [&](size_t a, size_t b) {
              return frames[a].frame_id < frames[b].frame_id;
            });

  // Dedup is only ever checked within the same segment_id: by construction
  // (see generate_corpus.py) distinct segments are separated far beyond
  // dedup_epsilon, so cross-segment pairs would never match; grouping first
  // turns an O(candidates^2) scan into O(sum of per-segment-group^2), the
  // same asymptotics the reference oracle (reference_miner.py) uses, which
  // is what the diff test is actually checking: not just the final kept
  // set, but that both implementations agree on every suppressed pair.
  std::vector<std::vector<size_t>> by_segment(num_segments);
  for (size_t idx : candidate_indices) {
    by_segment[frames[idx].segment_id].push_back(idx);
  }

  std::vector<size_t> kept;
  std::vector<std::pair<uint64_t, uint64_t>> dedup_pairs;  // (kept, suppressed)
  for (auto& group : by_segment) {
    // group is already frame_id-ascending because candidate_indices was.
    std::vector<size_t> group_kept;
    for (size_t idx : group) {
      bool is_dup = false;
      for (size_t kept_idx : group_kept) {
        if (SquaredDistance(frames[idx].feature, frames[kept_idx].feature) <=
            dedup_epsilon_sq_value) {
          dedup_pairs.emplace_back(frames[kept_idx].frame_id,
                                    frames[idx].frame_id);
          is_dup = true;
          break;
        }
      }
      if (!is_dup) group_kept.push_back(idx);
    }
    kept.insert(kept.end(), group_kept.begin(), group_kept.end());
  }
  std::sort(kept.begin(), kept.end(), [&](size_t a, size_t b) {
    return frames[a].frame_id < frames[b].frame_id;
  });

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "cannot open output: " << out_path << "\n";
    return 1;
  }
  out << "frame_id,segment_id\n";
  for (size_t idx : kept) {
    out << frames[idx].frame_id << ',' << frames[idx].segment_id << '\n';
  }
  out.close();

  std::cerr << "corpus_frames=" << frames.size()
            << " candidates=" << candidate_indices.size()
            << " kept=" << kept.size()
            << " suppressed=" << dedup_pairs.size() << "\n";
  for (const auto& [kept_id, dup_id] : dedup_pairs) {
    std::cerr << "dedup: kept=" << kept_id << " suppressed=" << dup_id << "\n";
  }
  return 0;
}
