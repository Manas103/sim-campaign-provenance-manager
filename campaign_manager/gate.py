"""The provenance gate: decide whether a new run becomes the accepted baseline.

Rule: compare the new run's manifest (commit, input-deck hash, solver
version) to the most recent accepted manifest for the same named case.

- No prior accepted run for this case: accept, it becomes the baseline.
- Manifest unchanged and content hash unchanged (within tolerance): accept,
  a non-event, still the same baseline.
- Manifest unchanged but content hash changed: REFUSE. Nothing recorded
  should explain a different result; flag as unexplained drift.
- Manifest changed (any of the three fields) regardless of whether content
  changed: accept, it becomes the new baseline. A legitimate code or input
  change is allowed to change the answer, and is also allowed to not change
  the answer (a refactor that is a no-op numerically is still a legitimate
  new baseline).

Content comparison is by exact hash equality of the canonicalized array
bytes (see manifest.hash_array_content), not a numeric tolerance, because
this toy solver is genuinely bit-for-bit deterministic end to end (fixed
iteration count, no threading, no unseeded randomness). A `numeric_tolerance`
knob is exposed for a max-abs-diff fallback comparison for solvers that are
not bit-for-bit deterministic, but it defaults to off; see README
limitations for why an exact-hash-only gate is a judgment call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import storage
from .manifest import Manifest, hash_array_content, hash_input_deck


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    baseline_run_id: Optional[str]
    manifest_changed: bool
    content_changed: bool


def evaluate_gate(
    conn,
    case_name: str,
    new_manifest: Manifest,
    new_content_hash: str,
    numeric_tolerance: Optional[float] = None,
    new_field: Optional[np.ndarray] = None,
) -> GateDecision:
    baseline = storage.get_latest_accepted(conn, case_name)
    if baseline is None:
        return GateDecision(
            accepted=True,
            reason="first_run_for_case",
            baseline_run_id=None,
            manifest_changed=True,
            content_changed=True,
        )

    manifest_changed = new_manifest.key() != baseline.manifest.key()

    if numeric_tolerance is not None and new_field is not None:
        baseline_field = storage.read_run_hdf5(Path(baseline.hdf5_path))
        content_changed = not np.allclose(
            new_field, baseline_field, atol=numeric_tolerance, rtol=0.0
        )
    else:
        content_changed = new_content_hash != baseline.content_hash

    if not manifest_changed and content_changed:
        return GateDecision(
            accepted=False,
            reason="unexplained_drift",
            baseline_run_id=baseline.run_id,
            manifest_changed=False,
            content_changed=True,
        )

    return GateDecision(
        accepted=True,
        reason="manifest_changed" if manifest_changed else "matches_baseline",
        baseline_run_id=baseline.run_id,
        manifest_changed=manifest_changed,
        content_changed=content_changed,
    )


def naive_reference_check(conn, case_name: str, run_id: str) -> GateDecision:
    """Deliberately slow, independently-coded re-derivation of the same decision.

    Does not call evaluate_gate and does not trust any cached hash column:
    it re-reads every run row for the case from scratch, re-reads the raw
    HDF5 file bytes off disk and rehashes them, re-parses the recorded input
    deck JSON and rehashes it, and does a linear scan (not the indexed
    "ORDER BY ... LIMIT 1" query the fast path uses) to find the true most
    recent prior accepted run strictly before this one. Used only in tests
    to cross-check the fast gate, never in the campaign driver itself.
    """
    all_rows = storage.all_runs_for_case(conn, case_name)
    target = None
    prior_accepted = None
    for row in all_rows:
        if row.run_id == run_id:
            target = row
            break
        if row.accepted:
            prior_accepted = row
    if target is None:
        raise ValueError(f"run {run_id} not found for case {case_name}")

    # Recompute everything from raw stored artifacts rather than trusting columns.
    recomputed_content_hash = hash_array_content(storage.read_run_hdf5(Path(target.hdf5_path)))
    recomputed_input_hash = hash_input_deck(json.loads(target.input_deck_json))
    recomputed_manifest = Manifest(
        commit=target.manifest.commit,
        input_hash=recomputed_input_hash,
        solver_version=target.manifest.solver_version,
    )

    if prior_accepted is None:
        return GateDecision(
            accepted=True,
            reason="first_run_for_case",
            baseline_run_id=None,
            manifest_changed=True,
            content_changed=True,
        )

    prior_content_hash = hash_array_content(storage.read_run_hdf5(Path(prior_accepted.hdf5_path)))
    manifest_changed = recomputed_manifest.key() != prior_accepted.manifest.key()
    content_changed = recomputed_content_hash != prior_content_hash

    if (not manifest_changed) and content_changed:
        return GateDecision(
            accepted=False,
            reason="unexplained_drift",
            baseline_run_id=prior_accepted.run_id,
            manifest_changed=False,
            content_changed=True,
        )
    return GateDecision(
        accepted=True,
        reason="manifest_changed" if manifest_changed else "matches_baseline",
        baseline_run_id=prior_accepted.run_id,
        manifest_changed=manifest_changed,
        content_changed=content_changed,
    )
