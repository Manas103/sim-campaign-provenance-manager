"""Toy deterministic stencil solver used as a stand-in for a real physics solver.

This is NOT one of the physics solvers in the other repos in this portfolio;
it is a small fixed-iteration-count Jacobi heat-equation stencil deliberately
kept cheap and deterministic so the thing actually under test is the campaign
manager (provenance manifest, storage, gate), not a real solver's numerics.

The solver is Markovian: the only state that carries across iterations is the
field array itself. This is what makes checkpoint/restart bit-for-bit
reproducible: resuming from a saved field at iteration k and continuing to N
produces exactly the same floating point bytes as running straight from 0 to
N, because there is no other hidden state (no RNG draws inside the loop, no
per-step randomness) once the initial condition is built.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SOLVER_VERSION = "toy-jacobi-heat-v1.0"

REQUIRED_KEYS = ("n", "iterations", "boundary_temp", "source_amplitude", "seed")


def validate_deck(deck: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in deck]
    if missing:
        raise ValueError(f"input deck missing required keys: {missing}")
    if deck["n"] < 2:
        raise ValueError("n must be >= 2")
    if deck["iterations"] < 0:
        raise ValueError("iterations must be >= 0")


def _initial_field(deck: Mapping[str, Any]) -> np.ndarray:
    """Build the initial field: fixed Dirichlet boundary plus optional seeded noise."""
    n = int(deck["n"])
    boundary_temp = float(deck["boundary_temp"])
    seed = deck.get("seed")

    field = np.zeros((n + 2, n + 2), dtype=np.float64)
    field[0, :] = boundary_temp
    field[-1, :] = boundary_temp
    field[:, 0] = boundary_temp
    field[:, -1] = boundary_temp

    if seed is not None:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(loc=0.0, scale=0.01, size=(n, n))
        field[1:-1, 1:-1] += noise

    return field


def _step(field: np.ndarray, source_amplitude: float, boundary_temp: float) -> np.ndarray:
    """One Jacobi update: average of four neighbors plus a fixed interior source term."""
    new = field.copy()
    new[1:-1, 1:-1] = 0.25 * (
        field[0:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, 0:-2] + field[1:-1, 2:]
    )
    new[1:-1, 1:-1] += source_amplitude * 0.01
    new[0, :] = boundary_temp
    new[-1, :] = boundary_temp
    new[:, 0] = boundary_temp
    new[:, -1] = boundary_temp
    return new


@dataclass
class Checkpoint:
    """Serializable solver state: the field array and the iteration count reached."""

    field: np.ndarray
    iteration: int
    deck: dict


def run_solver(deck: Mapping[str, Any]) -> np.ndarray:
    """Run the full deck from iteration 0 to deck['iterations'] and return the field."""
    validate_deck(deck)
    field = _initial_field(deck)
    iterations = int(deck["iterations"])
    source_amplitude = float(deck["source_amplitude"])
    boundary_temp = float(deck["boundary_temp"])
    for _ in range(iterations):
        field = _step(field, source_amplitude, boundary_temp)
    return field


def run_solver_to_checkpoint(deck: Mapping[str, Any], stop_at: int) -> Checkpoint:
    """Run from iteration 0 up to (not including further steps past) stop_at and checkpoint."""
    validate_deck(deck)
    if not (0 <= stop_at <= int(deck["iterations"])):
        raise ValueError("stop_at must be within [0, deck['iterations']]")
    field = _initial_field(deck)
    source_amplitude = float(deck["source_amplitude"])
    boundary_temp = float(deck["boundary_temp"])
    for _ in range(stop_at):
        field = _step(field, source_amplitude, boundary_temp)
    return Checkpoint(field=field, iteration=stop_at, deck=dict(deck))


def resume_from_checkpoint(checkpoint: Checkpoint) -> np.ndarray:
    """Resume a checkpointed run and finish the remaining iterations."""
    deck = checkpoint.deck
    total = int(deck["iterations"])
    remaining = total - checkpoint.iteration
    field = checkpoint.field.copy()
    source_amplitude = float(deck["source_amplitude"])
    boundary_temp = float(deck["boundary_temp"])
    for _ in range(remaining):
        field = _step(field, source_amplitude, boundary_temp)
    return field


def run_solver_with_restart(deck: Mapping[str, Any], checkpoint_at: int) -> np.ndarray:
    """Convenience wrapper: run to a checkpoint, serialize/deserialize it, then resume.

    The deep copy plus checkpoint round trip simulates a real checkpoint file being
    written to disk and read back, so the reproducibility test exercises the same
    code path a genuine restart would use, not just an in-memory continuation.
    """
    ckpt = run_solver_to_checkpoint(deck, checkpoint_at)
    # Simulate a checkpoint file round trip (serialize to plain data, deserialize).
    serialized = {
        "field": ckpt.field.copy(),
        "iteration": ckpt.iteration,
        "deck": copy.deepcopy(ckpt.deck),
    }
    restored = Checkpoint(
        field=serialized["field"], iteration=serialized["iteration"], deck=serialized["deck"]
    )
    return resume_from_checkpoint(restored)
