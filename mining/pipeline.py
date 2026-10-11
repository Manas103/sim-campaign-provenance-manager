"""Ties the C++ miner to the existing provenance gate (campaign_manager).

Mirrors campaign_manager.campaign.run_case exactly: the mined frame
selection is just another kind of "run" as far as storage.py and gate.py
are concerned (case_name="frame_mining"), reusing the identical
manifest/gate/HDF5 machinery the toy-solver campaigns already use rather
than inventing a second provenance mechanism. See IDEAMAKER design
principle 3 (reuse wins ties): the gate's accept/refuse rule for "manifest
unchanged but content moved" is exactly what "a dataset pinned by a
manifest" needs, with zero new gate logic.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign_manager import gate, storage
from campaign_manager.manifest import Manifest, hash_array_content, make_manifest

MINER_VERSION = "mine_frames_cpp_v1"
CASE_NAME = "frame_mining"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_mining_pass(
    miner_binary: Path,
    corpus_path: Path,
    queries_path: Path,
    dedup_epsilon: float,
    out_csv: Path,
) -> np.ndarray:
    """Runs mine_frames.cpp and returns the selected (frame_id, segment_id) rows."""
    result = subprocess.run(
        [
            str(miner_binary),
            f"--corpus={corpus_path}",
            f"--queries={queries_path}",
            f"--out={out_csv}",
            f"--dedup-epsilon={dedup_epsilon}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mine_frames failed: {result.stderr}")
    rows = np.loadtxt(out_csv, delimiter=",", skiprows=1, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows, result.stderr


def run_mining_case(
    conn,
    hdf5_dir: Path,
    miner_binary: Path,
    corpus_path: Path,
    queries_path: Path,
    dedup_epsilon: float,
    scratch_dir: Path,
    corpus_hash_override: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple:
    """End to end: mine, manifest-pin, gate. Returns (GateDecision, selected rows, miner_stderr)."""
    run_id = run_id or uuid.uuid4().hex[:12]
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_csv = scratch_dir / f"{run_id}.csv"

    selected, miner_stderr = run_mining_pass(
        miner_binary, corpus_path, queries_path, dedup_epsilon, out_csv
    )

    corpus_hash = corpus_hash_override or _hash_file(corpus_path)
    deck = {
        "corpus_hash": corpus_hash,
        "query_hash": _hash_file(queries_path),
        "dedup_epsilon": dedup_epsilon,
    }
    manifest = make_manifest(deck, MINER_VERSION)

    content_hash = hash_array_content(selected.astype(np.float64))
    hdf5_path = Path(hdf5_dir) / f"{run_id}.h5"
    storage.write_run_hdf5(hdf5_path, selected.astype(np.float64), run_id, CASE_NAME, manifest)

    decision = gate.evaluate_gate(conn, CASE_NAME, manifest, content_hash)

    storage.insert_run(
        conn,
        run_id=run_id,
        case_name=CASE_NAME,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        hdf5_path=str(hdf5_path),
        content_hash=content_hash,
        accepted=decision.accepted,
        status=decision.reason,
        manifest=manifest,
        input_deck_json=json.dumps(deck, sort_keys=True),
    )

    return decision, selected, miner_stderr
