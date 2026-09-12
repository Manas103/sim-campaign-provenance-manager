import pytest

from campaign_manager.manifest import hash_array_content
from campaign_manager.solver import (
    run_solver,
    run_solver_with_restart,
    validate_deck,
)

from .conftest import base_deck


def test_validate_deck_rejects_missing_keys():
    with pytest.raises(ValueError):
        validate_deck({"n": 4})


def test_validate_deck_rejects_bad_n():
    with pytest.raises(ValueError):
        validate_deck(base_deck(n=1))


def test_restart_matches_direct_run_bit_for_bit():
    deck = base_deck(n=16, iterations=100)
    direct = run_solver(deck)
    restarted = run_solver_with_restart(deck, checkpoint_at=37)
    assert hash_array_content(direct) == hash_array_content(restarted)
    assert (direct == restarted).all()


@pytest.mark.parametrize("checkpoint_at", [0, 1, 50, 99, 100])
def test_restart_at_various_points_matches_direct_run(checkpoint_at):
    deck = base_deck(n=10, iterations=100, seed=7)
    direct = run_solver(deck)
    restarted = run_solver_with_restart(deck, checkpoint_at=checkpoint_at)
    assert hash_array_content(direct) == hash_array_content(restarted)


def test_different_seed_gives_different_output():
    deck_a = base_deck(seed=1)
    deck_b = base_deck(seed=2)
    assert hash_array_content(run_solver(deck_a)) != hash_array_content(run_solver(deck_b))


def test_boundary_stays_fixed_at_boundary_temp():
    deck = base_deck(boundary_temp=3.5, iterations=20)
    field = run_solver(deck)
    assert (field[0, :] == 3.5).all()
    assert (field[-1, :] == 3.5).all()
    assert (field[:, 0] == 3.5).all()
    assert (field[:, -1] == 3.5).all()
