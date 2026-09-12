"""Reproducible reproducibility check: many small cases x many repetitions,
alternating straight runs and checkpoint/restart runs, hash-compared bit for
bit against the first run of each case.

Run with:
    venv\\Scripts\\python.exe scripts\\reproducibility_check.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign_manager.manifest import hash_array_content
from campaign_manager.solver import run_solver, run_solver_with_restart


def build_cases(n_cases: int = 40, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for _ in range(n_cases):
        cases.append(
            {
                "n": rng.choice([8, 10, 12, 14, 16, 18, 20]),
                "iterations": rng.choice([50, 80, 100, 120, 150, 180, 200]),
                "boundary_temp": rng.choice([0.5, 1.0, 1.5, 2.0]),
                "source_amplitude": rng.choice([1.0, 3.0, 5.0, 7.0]),
                "seed": rng.randint(0, 10_000),
            }
        )
    return cases


def run_reproducibility_sweep(n_cases: int = 40, reps_per_case: int = 30):
    cases = build_cases(n_cases)
    total = 0
    mismatches = []
    per_case_hashes = []
    for idx, deck in enumerate(cases):
        results = []
        for rep in range(reps_per_case):
            if rep % 2 == 0:
                h = hash_array_content(run_solver(deck))
            else:
                checkpoint_at = max(1, deck["iterations"] // 2)
                h = hash_array_content(run_solver_with_restart(deck, checkpoint_at))
            results.append(h)
            total += 1
        reference = results[0]
        per_case_hashes.append(reference)
        for rep, h in enumerate(results):
            if h != reference:
                mismatches.append((idx, rep, deck))
    return {
        "n_cases": n_cases,
        "reps_per_case": reps_per_case,
        "total": total,
        "matched": total - len(mismatches),
        "mismatches": mismatches,
        "distinct_case_hashes": len(set(per_case_hashes)),
    }


if __name__ == "__main__":
    result = run_reproducibility_sweep()
    print(f"cases: {result['n_cases']}")
    print(f"reps per case: {result['reps_per_case']}")
    print(f"total runs executed: {result['total']}")
    print(f"bit-for-bit matched: {result['matched']} / {result['total']}")
    print(f"distinct case output hashes: {result['distinct_case_hashes']} (of {result['n_cases']} cases)")
    if result["mismatches"]:
        print(f"MISMATCHES: {len(result['mismatches'])}")
        for idx, rep, deck in result["mismatches"][:10]:
            print(f"  case {idx} rep {rep}: {deck}")
    else:
        print("no mismatches")
