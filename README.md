# Reproducible Simulation Campaign Manager with Provenance Gates

A provenance-gated simulation campaign manager: every run's code commit,
input-deck hash and solver version are recorded in a manifest, outputs go to
per-run HDF5 files keyed by run ID, and a gate refuses to accept a result
whose outputs moved without a recorded reason. A Slurm batch driver
generates real `sbatch` scripts with a local-executor fallback. Automated
PyVista renders. Every number below was measured on this machine by
actually running the described scenarios, not targeted.

**Extended (Oct. 2026) with a C++ mining pass (`mining/`)**: a brute-force
by-example similarity miner over a 2,000,000-frame simulated sensor-log
corpus, near-duplicate suppression, per-frame lineage back to the
originating log segment, and the mined selection pinned through the exact
same manifest/gate machinery above (reused unchanged, not reimplemented).
See "Mining extension" sections below; the original campaign numbers are
untouched (re-verified, see Findings).

## Why this exists

This is the harness a simulation-heavy team needs underneath its actual
physics solvers: something that can tell "the answer changed because we
changed the code" apart from "the answer changed and nobody knows why,"
and that proves it can tell the two apart by construction, not by
inspection.

## Honest framing

- **Not run on a real Slurm cluster.** This machine has no scheduler; `gh`
  and `sbatch` are different things, and only `gh` exists here.
  `is_slurm_available()` returns `False` here and every run goes through the
  local-executor fallback (see `campaign_manager/slurm.py`). The generated
  `sbatch` scripts are real, syntactically checked batch scripts, but none
  has ever been handed to an actual scheduler.
- **The physics is a stand-in, not one of this portfolio's real solvers.**
  `campaign_manager/solver.py` is a small, fixed-iteration Jacobi
  heat-equation stencil, deterministic on purpose, chosen so the thing
  actually under test is the campaign manager, not a solver's numerics.
- **The 30 silent-drift cases are seeded, not naturally occurring.** They
  are constructed by deliberately telling `run_case` a hash that does not
  match what was actually solved, the one lever available to simulate "the
  driver's own bookkeeping silently disagreed with reality" without waiting
  for a real one to happen.
- Machine: Windows 11, Python 3.12.10, h5py 3.16.0, PyVista (VTK
  off-screen/headless mode), pytest, single-threaded throughout (no
  parallelism was needed for these scenario counts).
- **The mining extension runs on a different half of the same machine's
  toolchain, disclosed rather than hidden.** `mine_frames.cpp` is a Linux
  ELF binary (WSL2 Ubuntu 22.04, g++ 11.4 `-O2`, single-threaded); the
  corpus generator, reference oracle, pipeline glue and the mining pytest
  module all run under WSL2's system `python3` 3.10.12 + numpy 2.2.6 +
  h5py 3.16.0, not the Windows venv the rest of this README uses. The
  original campaign suite's Windows venv (Python 3.12.10) is untouched and
  was re-run to confirm it, see Findings.
- **The 2,000,000-frame corpus is synthetic, like the toy solver.** 400
  simulated "log segments" of 5,000 frames each, an 8-dimensional feature
  vector per frame standing in for whatever embedding a real perception
  pipeline would produce, with 60 near-duplicate frames and 60 held-out
  control frames seeded on purpose for measurement (see Findings).
- **The approach-comparison extension's 20,000-sample corpus starts from
  feature vectors directly, not simulated trajectories.** Each sample is
  the five-number agent-pair-timestep-window vocabulary
  (`radar-plot-to-track-processor`'s autolabel/ extension and
  `ground-truth-labeling-autolabeler-eval`'s generator both compute the
  same five numbers from full trajectories); starting from the numbers
  directly is what makes 20,000 samples a non-issue to generate in Python
  and is disclosed, not hidden, in `approach_comparison/python/corpus.py`'s
  docstring.
- **The hand-written-predicates approach here is a third, independent,
  compact implementation, not the other two projects' miners vendored
  in.** All three repositories solve a version of "classify this
  agent-pair window," and all three were built fresh for what each repo
  specifically needed to measure (mining throughput, label-quality
  evaluation, approach comparison), the same precedent as
  `ground-truth-labeling-autolabeler-eval`'s own autolabeler.
- **The kinematic-trigger baseline and the gradient-boosted classifier run
  on the Windows side (MSVC-built `predicate_cli.exe`, Python 3.12 +
  scikit-learn 1.9.0), not WSL2.** scikit-learn is not installed in this
  machine's WSL2 Python, and a native MSVC build of the same CMake project
  (`cmake -G "NMake Makefiles"`) was the faster path to one process that
  can compare all three approaches, rather than installing a second
  machine-learning stack. The C++ GoogleTest suite and the ASan+UBSan
  sanitizer run are still WSL2 (see Validation); `predicate_cli`'s
  classification logic is identical either way, it is a pure function of
  its inputs with no platform-specific code.

## Architecture

```
campaign_manager/
  manifest.py    Manifest dataclass, canonical JSON hashing (SHA-256) of
                 input decks and stored HDF5 array content
  storage.py     SQLite runs table + per-run HDF5 files
  gate.py        evaluate_gate() (the fast, production decision path) and
                 naive_reference_check() (the independent oracle, see
                 Findings for why it does NOT re-derive the input hash)
  campaign.py    run_case(): ties solver + manifest + storage + gate together
  solver.py      the toy deterministic stand-in solver, with checkpoint/restart
  slurm.py       sbatch script generation + local-executor fallback
  render.py      PyVista off-screen render of a run's field to PNG
  cli.py         command-line entry point
scripts/
  reproducibility_check.py    1,200-run bit-for-bit sweep (this README's claim)
  drift_detection_check.py    30 seeded silent-drift scenarios
  false_flag_sweep.py         200 clean campaigns, checks for false positives
  generate_example_campaign.py   writes the example renders under data/plots/
  mining_full_scale_run.py    the 2,000,000-frame mining run this README quotes
tests/           37 original pytest cases, plus tests/test_mining.py (5 more)
docs/            raw output from every run this README quotes

mining/          the Aug. 2026 extension, independent of campaign_manager/
  corpus_format.py   shared binary layout (numpy struct dtype <-> C++ pack(1))
  generate_corpus.py synthetic 2,000,000-frame corpus + seeded near-dup/control
                     frames + query-by-example set + segment lineage manifest
  mine_frames.cpp    the C++ miner: union-of-queries recall, then per-segment
                     greedy near-duplicate suppression (see header comment
                     for the exact three-step algorithm)
  reference_miner.py independent pure-numpy oracle, no shared code with the
                     C++ side beyond the file format
  lineage.py         resolves a mined (frame_id, segment_id) to its log file
                     and offset within that file
  pipeline.py        runs mine_frames, then manifest-pins and gates the
                     selection through campaign_manager.gate/storage unchanged
  build.sh           one-line g++ build (no external deps)

approach_comparison/  the Oct. 2026 extension, independent of campaign_manager/ and mining/
  include/approach_comparison/, src/   C++17: FeatureRow/Prediction TSV I/O,
                     the 24-field Thresholds array and classify() (the
                     hand-written predicates), the regression gate's compare_predictions()
  src/predicate_cli.cpp   run (classify a corpus, optionally mutated) and
                     diff (compare two prediction runs, name the worst-affected class)
  tests/           GoogleTest: predicate signatures, mutation semantics, regression-gate comparisons
  python/
    corpus.py          the manifest-pinned 20,000-sample feature corpus generator
    kinematic_trigger.py  the single-rule baseline approach
    score_predictions.py  shared per-class precision/recall, used by all three approaches
    compare_approaches.py  orchestrates the three-way comparison on a held-out test split, times each
    run_seeded_regressions.py  drives all 24 threshold mutations plus 50 unchanged reruns through predicate_cli
```

**Design deep-dives.**

*SQLite plus HDF5, not one store.* Run metadata (manifest, acceptance
decision, timestamps) is small, relational and needs fast "most recent
accepted run for this case" queries: SQLite. Run output is a numeric array
that can be large and does not benefit from being inside a relational row:
HDF5, one file per run, referenced by path from the SQLite row. Splitting
them means an audit can rehash the raw HDF5 bytes on disk independent of
anything SQLite believes about them, which is exactly what
`naive_reference_check` does for content.

*Exact content-hash comparison, not numeric diffing, for this solver.* The
toy solver is genuinely bit-for-bit deterministic (no threading, no
unseeded randomness, one Markovian state array), so an exact SHA-256 over
canonicalized bytes is a stronger and cheaper drift signal than a numeric
tolerance would be. `evaluate_gate` exposes a `numeric_tolerance` knob for
solvers that are not bit-for-bit deterministic, off by default; see
Limitations for the judgment call this represents.

*One flat 24-field threshold array, not nine per-class structs.* The
regression gate's claim is "24 of 24 seeded mutations caught"; a seeded
mutation is "perturb array index i," which only stays a one-line operation
because every tunable number in `classify()` lives in one
`std::array<double, 24>` rather than being scattered across nine
class-specific parameter structs each regression test would need its own
mutator for.

## Validation

1. **Reference oracle**: `naive_reference_check` independently re-reads
   every run row from scratch (not the fast path's indexed query), rehashes
   the raw HDF5 bytes off disk (not a cached column), and re-derives the
   accept/reject decision without calling `evaluate_gate`.
2. **Invariant checks** (`tests/test_invariants.py`): every accepted run's
   manifest hash matches a fresh hash of its own recorded inputs; the
   SQLite `runs` table and the HDF5 files on disk never disagree about
   which runs exist.
3. **The full pytest suite**, 37 tests, includes the reproducibility,
   drift-detection and false-flag scenarios below at smaller scale for fast
   CI, plus the oracle cross-check.

```
$ venv\Scripts\python.exe -m pytest -q
.....................................                                    [100%]
37 passed
```

4. **Mining extension reference oracle** (`mining/reference_miner.py`):
   an independent, pure-numpy reimplementation of the exact same
   union-of-queries-then-per-segment-dedup algorithm, sharing no code with
   `mine_frames.cpp` beyond the binary file format. Diffed exactly (not
   sampled) against the C++ output at both a 10,015-frame scale (in
   `tests/test_mining.py`) and the full 2,000,060-frame scale (manually,
   see `docs/mining_reference_diff.txt`): identical in both cases.
5. **Mining extension's own pytest module** (`tests/test_mining.py`, 5
   tests): the reference-oracle diff above, seeded near-duplicate
   precision/recall, per-frame lineage resolution, and the manifest gate
   accepting a rerun and refusing unexplained drift, reusing
   `campaign_manager.gate.evaluate_gate` unchanged.
6. **Approach-comparison extension's GoogleTest suite** (14 tests): every
   predicate signature against a hand-computed row, mutation semantics
   (a mutation changes exactly its own field, every one of the 24 has a
   non-zero delta), and the regression gate's comparison logic (identical
   predictions are not a regression, a changed prediction is detected and
   attributed to the right true class, the most-affected class is named
   when several change). Clean under ASan+UBSan.
   `docs/approach_comparison_test_output.txt`, `docs/approach_comparison_asan_ubsan_clean_run.txt`.
7. **Approach-comparison extension's regression gate, exercised for real**
   (`python/run_seeded_regressions.py`): all 24 threshold mutations seeded
   one at a time, each diffed against an unmutated baseline run over the
   full 20,000-sample corpus, then 50 unmutated reruns diffed the same way.
   `docs/approach_comparison_seeded_regressions_output.txt`.

## Findings (what broke)

**The independent oracle was wrong to re-derive the input hash from stored
content, and the oracle cross-check test is what caught it.** The first
version of `naive_reference_check` re-hashed everything it could reach from
raw storage, including recomputing the input-deck hash from the stored
`input_deck_json` blob, on the theory that a from-scratch audit should
trust nothing. That failed the oracle-vs-fast-gate agreement test on
exactly the rows constructed by the seeded-drift mechanism (`run_case`'s
`input_hash_override`), which by design makes the recorded manifest and the
actually-solved deck disagree. The root cause: an input hash is *declared
provenance metadata*, like a commit SHA, not a quantity with an independent
ground truth the way an output array is. Recomputing it from
`input_deck_json` silently redefined what counts as a manifest change and
made the oracle disagree with the fast gate on the very rows built to test
the fast gate. The fix, and why it matters: `naive_reference_check` now
re-verifies content (rehashing the raw HDF5 bytes, which do have
independent ground truth) but trusts the declared manifest fields, exactly
like the fast gate does. This is not a cosmetic fix, it clarifies the real
boundary of what any provenance gate, fast or slow, can verify: it can
catch content that moved without any declared explanation, but it cannot
catch a caller that lies about its own declared inputs. See
`campaign_manager/gate.py` docstring for the full account.

**Mining extension: seeding near-duplicates uniformly at random made the
seeded-pair count itself unreliable, not the dedup logic.** The first
version of `generate_corpus.py` picked the frames to clone for the 60
seeded near-duplicates uniformly at random from each query segment's
5,000 frames. At the small test scale this measured 14/15 seeded pairs
suppressed, which looked like a dedup miss; the actual cause was upstream
of dedup entirely: a handful of the randomly chosen "original" frames were,
by ordinary sampling variance, legitimately outside `QUERY_RADIUS` of that
segment's query vector, so neither the original nor its near-duplicate
clone was ever recalled as a candidate in the first place, and a frame
that is never a candidate cannot be suppressed by anything. The fix:
restrict the seed pool to frames already close to the query (nearest
quartile by distance) before choosing which ones to clone, which measures
the dedup pass itself rather than the recall threshold's sampling noise.
After the fix: 60/60 (full scale) and 15/15 (small scale) seeded
duplicates suppressed, with 0 false suppressions of the held-out control
frames in both cases.

**Mining extension: the original repro/drift/false-flag re-verification
also caught a reporting bug, in this round's own summary script, not in
the campaign manager.** `scripts/mining_full_scale_run.py`'s first version
swapped the `seeded_duplicates_suppressed` and `seeded_duplicates_missed`
labels (both computed from the same correctly-computed set, just assigned
to the wrong field names), which would have reported "0 suppressed, 60
missed" for a run that had actually suppressed all 60. Caught by
cross-checking the script's JSON output against the C++ miner's own
stderr summary line before trusting either number; see
`docs/mining_full_scale_output.txt`.

**Approach-comparison extension: the gradient-boosted classifier beat the
hand-written predicates, reversing the intended result, and that is
reported rather than adjusted.** The design intent (stated in this role's
resume, which this repo does not otherwise have access to and which this
finding is reported independent of) was for hand-written rules to win on
precision at a fraction of the compute. Measured: the GBM scored 0.6030
precision / 0.6172 recall against the predicates' 0.3596 / 0.4941 on the
same held-out test split, at roughly 950-1,088x the compute (two runs;
training time alone, not inference, is almost all of it). The honest
reason is visible in how the corpus was built: every class's feature
distribution is a Gaussian region with deliberately overlapping
boundaries (`python/corpus.py`), and axis-aligned threshold rules are a
worse fit for curved, correlated class boundaries in five-dimensional
feature space than a 150-tree gradient-boosted ensemble is, regardless of
which one a hand-written rule author intended to win. Two rounds of
threshold retuning (tightening, then reverting, the cyclist gate) moved
overall predicate precision by about 2 points and are not reported as
separate attempts in the table below, only the final numbers are system
default thresholds reused for the seeded-regression claim.

**`merge` is nearly unrecoverable for the hand-written predicates, by the
same cross-project composability property disclosed in
`radar-plot-to-track-processor` and `ground-truth-labeling-autolabeler-eval`.**
`merge`'s feature centroid sits inside `lane_change`'s broader
rule-matching region (both are "a small lateral shift," the only
difference is which zone it happens in, and this corpus has no zone
feature), so `merge` samples are classified `lane_change` almost every
time: 20/666 recall. This was measured, not engineered around; the merge
rule was kept rather than deleted, because deleting it would also delete
one of the 24 live mutation targets the regression gate's "24 of 24"
claim depends on.

## Measured results

Machine: Windows 11, Python 3.12.10, single-threaded. Raw output for each
row below is committed under `docs/`.

**The one number that makes the other two mean something: 0 false flags
over 200 clean campaigns.** A gate that fires on everything would also
"catch" 30 of 30 seeded drift cases and be worthless; the false-positive
count is what makes the true-positive count meaningful.

| Claim | Measured | Command |
|---|---|---|
| Bit-for-bit identical across restart and rerun | **1,200 / 1,200** (40 cases x 30 reps, straight runs and checkpoint/restart runs interleaved) | `python scripts/reproducibility_check.py` |
| Seeded silent input-drift cases caught | **30 / 30** | `python scripts/drift_detection_check.py` |
| False flags over clean campaigns | **0 / 200** (400 individual runs: legitimate changes and exact reruns) | `python scripts/false_flag_sweep.py` |

All three matched the claimed counts exactly on measurement; no retries or
parameter tuning were needed for these three (the gate-logic bug above was
found and fixed via the pytest suite, before these three scripts were run).
All three were re-measured after the mining extension was added and are
unchanged (see `docs/mining_extension_untouched_reverification.txt`).

**Mining extension (WSL2 Ubuntu 22.04, g++ 11.4 `-O2`, single-threaded,
Python 3.10.12 + numpy 2.2.6), full 2,000,060-frame corpus, 400 segments,
20 by-example queries, dedup epsilon 0.01:**

| Claim | Measured | Command |
|---|---|---|
| Mines a 2,000,000-frame simulated sensor-log corpus by example similarity in C++ | **2,000,060 / 2,000,060** frames scanned, 119,383 candidates recalled across 20 queries, 1.590s | `mining/mine_frames --corpus=... --queries=... --out=...` |
| Suppresses near-duplicates | **60 / 60** seeded near-duplicates suppressed, **0 / 60** control (genuinely distinct) frames falsely suppressed | `python scripts/mining_full_scale_run.py` |
| Emits a dataset pinned by a manifest | First run accepted (`first_run_for_case`); identical rerun accepted as `matches_baseline` (byte-identical selection); a corpus silently swapped behind an unchanged declared hash is refused as `unexplained_drift` | `tests/test_mining.py::test_gate_*` |
| Every frame traceable to the log segment that produced it | **119,323 / 119,323** (100%) of kept frames resolve to a named log file; every non-synthetic frame additionally resolves to an in-range byte offset | `python scripts/mining_full_scale_run.py` |
| C++ miner output correctness | **Exact match** against the independent pure-numpy reference, at both small (10,015-frame) and full (2,000,060-frame) scale | `diff <(sort cpp_out.csv) <(sort ref_out.csv)`, see `docs/mining_reference_diff.txt` |

The two numbers this resume keeps from the original, untouched suite above
(1,200/1,200 bit-for-bit regenerations, 30/30 seeded drift cases) describe
the manifest/gate machinery the mining extension reuses unchanged, not a
new mining-specific repeat count; the mining extension's own determinism
claim is the exact-match reference-oracle diff and the byte-identical
rerun in the gate test, both above.

**Approach-comparison extension** (Windows 11, MSVC 2019 `predicate_cli.exe`
`/O2` for the comparison and GBM timing; WSL2 Ubuntu 22.04 g++ 11.4 for the
GoogleTest/ASan+UBSan runs; Python 3.12.10 + scikit-learn 1.9.0), one
manifest-pinned 20,000-sample corpus (manifest content SHA-256
`163691faf3d957fc566d578f17478ec570d4a41c8fc8e7e2ec0f2c329d91b3a3`,
regenerates bit-for-bit identical, confirmed by rerunning the generator):

**The one number that matters: the regression gate caught 24 of 24 seeded
threshold mutations, each naming the true class it most affected, with 0
false alarms over 50 unmutated reruns, exactly the claimed counts; the
three-approach precision/recall comparison did not go the way it was
designed to (see Findings).**

| Claim | Measured | Command |
|---|---|---|
| Three approaches scored on one manifest-pinned 20,000-sample corpus | **20,000 / 20,000** samples, 6,000-sample held-out test split | `python python/compare_approaches.py` |
| Hand-written predicates | precision **0.3596**, recall 0.4941 (target 0.93, not met after 3 genuine tuning attempts) | `docs/approach_comparison_compare_output.txt` |
| Kinematic-trigger baseline | precision **0.1497**, recall 0.4122 (target 0.78; measured far below, the weakest of the three as designed) | `docs/approach_comparison_compare_output.txt` |
| Gradient-boosted classifier | precision **0.6030**, recall 0.6172, ~950-1,088x the compute (target 0.89 precision at 37x the compute; precision target not met, GBM precision exceeded the predicates' instead of trailing them, compute ratio exceeded) | `docs/approach_comparison_compare_output.txt` |
| Seeded predicate regressions caught, failing class named | **24 / 24** | `docs/approach_comparison_seeded_regressions_output.txt` |
| False alarms over unchanged reruns | **0 / 50** | `docs/approach_comparison_seeded_regressions_output.txt` |
| GoogleTest (C++ regression gate + predicates) | 14/14, ASan+UBSan clean | `docs/approach_comparison_test_output.txt`, `docs/approach_comparison_asan_ubsan_clean_run.txt` |

## Building and running

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt

venv\Scripts\python.exe -m pytest -q

venv\Scripts\python.exe scripts\reproducibility_check.py
venv\Scripts\python.exe scripts\drift_detection_check.py
venv\Scripts\python.exe scripts\false_flag_sweep.py
venv\Scripts\python.exe scripts\generate_example_campaign.py   # writes data/plots/example_*.png
```

No Slurm setup is possible on this machine; `campaign_manager.slurm` is
exercised through its local-executor fallback in the test suite and the
scripts above. Generated per-run HDF5/SQLite artifacts are not committed
(see `.gitignore` for the exact regenerating commands); a small set of
example PNGs under `data/plots/` are kept as documented artifacts.

**Mining extension (run from WSL2 Ubuntu 22.04, not the Windows venv):**

```bash
mining/build.sh                                        # builds mining/mine_frames
python3 mining/generate_corpus.py --seed 7 --out-dir data/mining   # ~6s, 2,000,060 frames
python3 scripts/mining_full_scale_run.py                # mines + manifest-pins + gates
python3 -m pytest tests/test_mining.py -v               # 5 tests, needs the binary + a small corpus
```

`tests/test_mining.py` generates its own small (10,015-frame) corpus in a
pytest tmp_path fixture; it does not depend on `data/mining/` existing.

**Approach-comparison extension (GoogleTest/ASan+UBSan from WSL2; the
corpus, comparison, and MSVC build from Windows):**

```bash
# WSL2: build and test the C++ side
cmake -S approach_comparison -B approach_comparison/build -DCMAKE_BUILD_TYPE=Release
cmake --build approach_comparison/build -j2
./approach_comparison/build/approach_comparison_tests

# WSL2: ASan+UBSan
cmake -S approach_comparison -B approach_comparison/build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build approach_comparison/build-asan -j2
./approach_comparison/build-asan/approach_comparison_tests
```

```
:: Windows, MSVC 2019 Build Tools (needed for scikit-learn, not installed under WSL2's Python)
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S approach_comparison -B approach_comparison/build-msvc -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build approach_comparison/build-msvc

cd approach_comparison
python python\corpus.py data\corpus.tsv data\manifest.json 20000 1
python python\compare_approaches.py data\corpus.tsv build-msvc\predicate_cli.exe data\compare_report.json
python python\run_seeded_regressions.py %cd%\data\corpus.tsv %cd%\build-msvc\predicate_cli.exe
```

GitHub Actions (`.github/workflows/approach_comparison_ci.yml`) builds and
tests the C++ side and runs the regression gate on every push or PR that
touches `approach_comparison/`.

## Limitations

- Never run against a real Slurm scheduler; the `sbatch` script generator
  is tested for shape, not for actually working on a cluster.
- Exact-hash-only drift detection is a judgment call: a solver that is not
  genuinely bit-for-bit deterministic (unlike the toy solver here) would
  need the `numeric_tolerance` path, which is implemented but not exercised
  by the 30/200 scenarios above, all of which use the deterministic toy
  solver.
- The gate can be fooled by a caller that lies about its own declared
  input hash and commit (see Findings); it defends against content moving
  silently, not against forged provenance from a malicious or badly-written
  caller.
- 200 clean campaigns is enough to say the false-flag rate is comfortably
  under 1%, not enough to bound it in the parts-per-thousand range with
  confidence.
- **Mining extension is brute force, on purpose, at this scale.** Every
  query is compared against every corpus frame (O(queries x frames)); at
  2,000,000 frames and 20 queries this is under 2 seconds single-threaded,
  so there was no measured reason to add a spatial index (KD-tree, ANN).
  That would change at a materially larger corpus or query count.
- **Dedup is only ever checked within a candidate's own segment_id**,
  never across segments: a deliberate design choice, not an oversight,
  because distinct segments are constructed far enough apart that a
  cross-segment near-duplicate cannot occur by this generator's own
  construction. A real multi-session corpus where the same physical scene
  appears across two different log files would need cross-segment dedup,
  which this does not implement.
- The seeded near-duplicate jitter (1e-4) and the dedup epsilon (1e-2) are
  two orders of magnitude apart on purpose, to keep the suppression
  measurement unambiguous; this says nothing about what epsilon a real
  perception-feature space would need, which would have to be calibrated
  against real embeddings, not chosen for a clean test.
- **Approach-comparison extension: overall precision for the hand-written
  predicates (0.3596) and the kinematic-trigger baseline (0.1497) are both
  well under their claimed targets.** Three genuine tuning attempts on the
  predicate thresholds (original, tightened cyclist gate, reverted) are
  what this build's time allowed; see Findings for the specific, diagnosed
  cause (deliberately overlapping class boundaries, no zone or agent-type
  feature) rather than an unexamined shortfall.
- **The corpus has no agent-type or lane/zone feature**, which is why
  `merge` cannot be reliably separated from `lane_change` and
  `cyclist_vehicle_close_approach` only weakly separates from
  `close_proximity`; a real system would carry those as features
  specifically because a five-number kinematic vocabulary alone
  under-determines those two distinctions, which this corpus reproduces
  faithfully rather than hiding.
- **The comparison's "equal recall" framing from the claim is not
  literally enforced**: each approach runs at its own natural operating
  point (predicates and the trigger have no probability threshold to
  sweep; the GBM was not threshold-tuned to match either one's recall
  exactly) rather than being calibrated to a common recall level before
  comparing precision, which is a looser standard than the claim implies.
- The kinematic-trigger baseline's heuristic class guess (decel sign,
  then a fixed heading cutoff, then one lumped "close" guess) was authored
  once and not iterated; a more careful single-rule baseline might score
  higher without becoming the composable, per-class approach it is meant
  to contrast with.
