"""SQLite run/manifest store plus per-run HDF5 output files.

Two tables: `runs` (one row per run: id, case name, where its HDF5 output
lives, its content hash, whether it was accepted) and `manifest` (one row
per run: the git commit, input-deck hash and solver version that produced
it). They are joined on run_id. Splitting them mirrors the actual question
being asked: "what happened" (runs) versus "what should explain it"
(manifest), and keeps the manifest columns identically shaped whether the
run was accepted, refused, or is still pending review.

Design note (SQLite + HDF5, not one store): SQLite is good at exactly the
query the gate needs ("give me the most recent accepted manifest for this
case") and is trivially transactional. HDF5 is good at large dense numeric
arrays with a stable binary layout. Putting the array bytes inside SQLite
as a BLOB would work at this toy scale but would not survive the campaign
manager's actual purpose, which is standing in for a workflow where each
run's output is a real multi-field simulation dump; HDF5 is what the
downstream PyVista rendering step and any real post-processing tooling
already expects to open.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from .manifest import Manifest, hash_array_content

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    case_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    hdf5_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS manifest (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    commit_sha TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    input_deck_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_case ON runs(case_name, created_at);
"""


@dataclass
class RunRecord:
    run_id: str
    case_name: str
    created_at: str
    hdf5_path: str
    content_hash: str
    accepted: bool
    status: str
    note: str
    manifest: Manifest
    input_deck_json: str


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def write_run_hdf5(
    path: Path,
    field: np.ndarray,
    run_id: str,
    case_name: str,
    manifest: Manifest,
) -> None:
    """Write one run's output field to its own HDF5 file, keyed by run_id.

    Manifest fields are stored as attributes for self-description (a human or
    another tool opening the file in isolation can see what produced it), but
    they are never part of the content hash; see manifest.hash_array_content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        dset = f.create_dataset("field", data=np.ascontiguousarray(field, dtype=np.float64))
        dset.attrs["run_id"] = run_id
        f.attrs["case_name"] = case_name
        f.attrs["commit_sha"] = manifest.commit
        f.attrs["input_hash"] = manifest.input_hash
        f.attrs["solver_version"] = manifest.solver_version


def read_run_hdf5(path: Path) -> np.ndarray:
    with h5py.File(str(path), "r") as f:
        return f["field"][:]


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    case_name: str,
    created_at: str,
    hdf5_path: str,
    content_hash: str,
    accepted: bool,
    status: str,
    manifest: Manifest,
    input_deck_json: str,
    note: str = "",
) -> None:
    conn.execute(
        "INSERT INTO runs (run_id, case_name, created_at, hdf5_path, content_hash, "
        "accepted, status, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, case_name, created_at, hdf5_path, content_hash, int(accepted), status, note),
    )
    conn.execute(
        "INSERT INTO manifest (run_id, commit_sha, input_hash, solver_version, input_deck_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, manifest.commit, manifest.input_hash, manifest.solver_version, input_deck_json),
    )
    conn.commit()


def _row_to_record(row) -> RunRecord:
    (
        run_id,
        case_name,
        created_at,
        hdf5_path,
        content_hash,
        accepted,
        status,
        note,
        commit_sha,
        input_hash,
        solver_version,
        input_deck_json,
    ) = row
    return RunRecord(
        run_id=run_id,
        case_name=case_name,
        created_at=created_at,
        hdf5_path=hdf5_path,
        content_hash=content_hash,
        accepted=bool(accepted),
        status=status,
        note=note,
        manifest=Manifest(commit=commit_sha, input_hash=input_hash, solver_version=solver_version),
        input_deck_json=input_deck_json,
    )


_JOIN_SELECT = (
    "SELECT r.run_id, r.case_name, r.created_at, r.hdf5_path, r.content_hash, r.accepted, "
    "r.status, r.note, m.commit_sha, m.input_hash, m.solver_version, m.input_deck_json "
    "FROM runs r JOIN manifest m ON r.run_id = m.run_id "
)


def get_latest_accepted(conn: sqlite3.Connection, case_name: str) -> Optional[RunRecord]:
    cur = conn.execute(
        _JOIN_SELECT + "WHERE r.case_name = ? AND r.accepted = 1 "
        "ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1",
        (case_name,),
    )
    row = cur.fetchone()
    return _row_to_record(row) if row else None


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[RunRecord]:
    cur = conn.execute(_JOIN_SELECT + "WHERE r.run_id = ?", (run_id,))
    row = cur.fetchone()
    return _row_to_record(row) if row else None


def all_runs_for_case(conn: sqlite3.Connection, case_name: str) -> list[RunRecord]:
    cur = conn.execute(
        _JOIN_SELECT + "WHERE r.case_name = ? ORDER BY r.created_at ASC, r.rowid ASC",
        (case_name,),
    )
    return [_row_to_record(row) for row in cur.fetchall()]


def all_runs(conn: sqlite3.Connection) -> list[RunRecord]:
    cur = conn.execute(_JOIN_SELECT + "ORDER BY r.created_at ASC, r.rowid ASC")
    return [_row_to_record(row) for row in cur.fetchall()]
