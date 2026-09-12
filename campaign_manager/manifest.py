"""Provenance manifest: code commit, canonicalized input-deck hash, solver version.

The one rule that matters here, found the hard way (see README "what broke"):
a content hash must be computed only from the actual numeric content that is
supposed to define the result. Anything else that can vary run to run without
the result actually changing (a timestamp, a PID, a free-space map inside a
container file format) must never leak into a hash that is later compared for
equality, or every run looks like drift.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def canonicalize_input_deck(deck: Mapping[str, Any]) -> bytes:
    """Deterministic byte representation of an input deck, independent of key order,
    float formatting, or dict construction path."""
    return json.dumps(deck, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_input_deck(deck: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonicalize_input_deck(deck)).hexdigest()


def hash_array_content(arr: np.ndarray) -> str:
    """Content-only hash of a numeric array: fixed dtype, fixed memory order.

    This is deliberately NOT a hash of any file the array happens to be stored
    in. See README findings: hashing raw HDF5 file bytes picks up a per-write
    creation timestamp attribute and gives two byte-identical-content runs two
    different hashes, which is a false drift flag waiting to happen.
    """
    canonical = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def get_code_commit(repo_path: Optional[Path] = None) -> str:
    """Short git SHA of HEAD in the campaign manager's own repo.

    Returns "unknown" if not run inside a git repo (e.g. an isolated tmp
    fixture in a unit test that constructs manifests directly instead).
    """
    cwd = str(repo_path) if repo_path is not None else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Manifest:
    commit: str
    input_hash: str
    solver_version: str

    def key(self) -> tuple:
        return (self.commit, self.input_hash, self.solver_version)


def make_manifest(
    deck: Mapping[str, Any],
    solver_version: str,
    repo_path: Optional[Path] = None,
    commit_override: Optional[str] = None,
) -> Manifest:
    commit = commit_override if commit_override is not None else get_code_commit(repo_path)
    return Manifest(
        commit=commit,
        input_hash=hash_input_deck(deck),
        solver_version=solver_version,
    )
