# Reproducible Simulation Campaign Manager with Provenance Gates

A provenance-gated simulation campaign manager: every run's code commit,
input-deck hash and solver version are recorded in a manifest, outputs go to
per-run HDF5 files keyed by run ID, and a gate refuses to accept a result
whose outputs moved without a recorded reason. A Slurm batch driver
generates real `sbatch` scripts with a local-executor fallback. Automated
PyVista renders. Every number below was measured on this machine by
actually running the described scenarios, not targeted.

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
tests/           37 pytest cases covering all of the above
docs/            raw output from every run this README quotes
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
