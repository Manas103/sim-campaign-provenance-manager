import subprocess

from campaign_manager.manifest import (
    canonicalize_input_deck,
    get_code_commit,
    hash_array_content,
    hash_input_deck,
    make_manifest,
)
from campaign_manager.solver import SOLVER_VERSION, run_solver

from .conftest import base_deck


def test_canonicalization_is_key_order_independent():
    deck_a = {"n": 8, "iterations": 10, "boundary_temp": 1.0, "source_amplitude": 2.0, "seed": 1}
    deck_b = {"seed": 1, "source_amplitude": 2.0, "boundary_temp": 1.0, "iterations": 10, "n": 8}
    assert canonicalize_input_deck(deck_a) == canonicalize_input_deck(deck_b)
    assert hash_input_deck(deck_a) == hash_input_deck(deck_b)


def test_hash_changes_when_content_changes():
    deck_a = base_deck()
    deck_b = base_deck(seed=43)
    assert hash_input_deck(deck_a) != hash_input_deck(deck_b)


def test_array_content_hash_is_dtype_and_order_independent():
    import numpy as np

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[1, 2], [3, 4]], dtype=np.float64)
    assert hash_array_content(a) == hash_array_content(b)

    c = np.asfortranarray(b)
    assert hash_array_content(b) == hash_array_content(c)


def test_array_content_hash_detects_real_difference():
    import numpy as np

    a = np.zeros((4, 4))
    b = np.zeros((4, 4))
    b[2, 2] = 1e-9
    assert hash_array_content(a) != hash_array_content(b)


def test_get_code_commit_matches_real_git_head(repo_path):
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if expected.returncode != 0:
        import pytest

        pytest.skip("repo has no commits yet")
    assert get_code_commit(repo_path) == expected.stdout.strip()


def test_get_code_commit_unknown_outside_repo(tmp_path):
    assert get_code_commit(tmp_path) == "unknown"


def test_make_manifest_reflects_deck_and_version(repo_path):
    deck = base_deck()
    m = make_manifest(deck, SOLVER_VERSION, repo_path=repo_path)
    assert m.input_hash == hash_input_deck(deck)
    assert m.solver_version == SOLVER_VERSION
    assert m.commit != ""


def test_solver_is_deterministic_across_two_direct_calls():
    deck = base_deck()
    f1 = run_solver(deck)
    f2 = run_solver(deck)
    assert hash_array_content(f1) == hash_array_content(f2)
