"""Orchestrator: run one case end to end (solve, manifest, store, gate)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from . import gate, storage
from .manifest import Manifest, hash_array_content, make_manifest
from .solver import SOLVER_VERSION, run_solver, run_solver_with_restart


def run_case(
    conn: sqlite3.Connection,
    case_name: str,
    deck: Mapping[str, Any],
    hdf5_dir: Path,
    repo_path: Optional[Path] = None,
    solver_version: str = SOLVER_VERSION,
    commit_override: Optional[str] = None,
    input_hash_override: Optional[str] = None,
    checkpoint_at: Optional[int] = None,
    run_id: Optional[str] = None,
) -> gate.GateDecision:
    """Run a case, write its HDF5 output, insert the run + manifest, then gate it.

    `input_hash_override` exists only for the seeded silent-drift test
    construction: it lets a test record a manifest that claims the hash of a
    deck OTHER than the one actually solved, exactly the real failure mode
    (input changed, the manifest was not updated to reflect it) the gate
    exists to catch. The normal path never uses it.
    """
    run_id = run_id or uuid.uuid4().hex[:12]

    if checkpoint_at is not None:
        field = run_solver_with_restart(deck, checkpoint_at)
    else:
        field = run_solver(deck)

    manifest = make_manifest(deck, solver_version, repo_path=repo_path, commit_override=commit_override)
    if input_hash_override is not None:
        manifest = Manifest(
            commit=manifest.commit, input_hash=input_hash_override, solver_version=manifest.solver_version
        )

    content_hash = hash_array_content(field)
    hdf5_path = Path(hdf5_dir) / f"{run_id}.h5"
    storage.write_run_hdf5(hdf5_path, field, run_id, case_name, manifest)

    decision = gate.evaluate_gate(conn, case_name, manifest, content_hash)

    storage.insert_run(
        conn,
        run_id=run_id,
        case_name=case_name,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        hdf5_path=str(hdf5_path),
        content_hash=content_hash,
        accepted=decision.accepted,
        status=decision.reason,
        manifest=manifest,
        input_deck_json=json.dumps(dict(deck), sort_keys=True),
    )

    return decision
