"""Unit tests for the gate with constructed manifests, exercising both branches
directly, independent of the solver or the full campaign orchestrator."""
import numpy as np

from campaign_manager import campaign, gate, storage
from campaign_manager.manifest import Manifest, hash_array_content, hash_input_deck

from .conftest import base_deck


def _record(conn, hdf5_dir, case_name, run_id, commit, input_hash, solver_version, field, accepted):
    manifest = Manifest(commit=commit, input_hash=input_hash, solver_version=solver_version)
    path = hdf5_dir / f"{run_id}.h5"
    storage.write_run_hdf5(path, field, run_id, case_name, manifest)
    storage.insert_run(
        conn,
        run_id=run_id,
        case_name=case_name,
        created_at=f"2026-09-{10 + int(run_id[-1]):02d}T00:00:00+00:00",
        hdf5_path=str(path),
        content_hash=hash_array_content(field),
        accepted=accepted,
        status="accepted" if accepted else "unexplained_drift",
        manifest=manifest,
        input_deck_json="{}",
    )


def test_first_run_for_case_is_always_accepted(db_conn, hdf5_dir):
    field = np.zeros((4, 4))
    manifest = Manifest(commit="abc123", input_hash="h1", solver_version="v1")
    decision = gate.evaluate_gate(db_conn, "case_a", manifest, hash_array_content(field))
    assert decision.accepted
    assert decision.reason == "first_run_for_case"


def test_same_manifest_same_content_is_accepted_non_event(db_conn, hdf5_dir):
    field = np.ones((4, 4))
    _record(db_conn, hdf5_dir, "case_b", "run1", "abc123", "h1", "v1", field, accepted=True)

    manifest = Manifest(commit="abc123", input_hash="h1", solver_version="v1")
    decision = gate.evaluate_gate(db_conn, "case_b", manifest, hash_array_content(field))
    assert decision.accepted
    assert decision.reason == "matches_baseline"
    assert not decision.content_changed


def test_same_manifest_different_content_is_refused_as_drift(db_conn, hdf5_dir):
    field = np.ones((4, 4))
    _record(db_conn, hdf5_dir, "case_c", "run1", "abc123", "h1", "v1", field, accepted=True)

    drifted = field.copy()
    drifted[0, 0] = 99.0
    manifest = Manifest(commit="abc123", input_hash="h1", solver_version="v1")
    decision = gate.evaluate_gate(db_conn, "case_c", manifest, hash_array_content(drifted))
    assert not decision.accepted
    assert decision.reason == "unexplained_drift"
    assert decision.manifest_changed is False
    assert decision.content_changed is True


def test_manifest_changed_different_content_is_accepted_new_baseline(db_conn, hdf5_dir):
    field = np.ones((4, 4))
    _record(db_conn, hdf5_dir, "case_d", "run1", "abc123", "h1", "v1", field, accepted=True)

    changed = field.copy()
    changed[0, 0] = 42.0
    manifest = Manifest(commit="def456", input_hash="h1", solver_version="v1")
    decision = gate.evaluate_gate(db_conn, "case_d", manifest, hash_array_content(changed))
    assert decision.accepted
    assert decision.reason == "manifest_changed"
    assert decision.manifest_changed is True


def test_input_hash_changed_alone_counts_as_manifest_changed(db_conn, hdf5_dir):
    field = np.ones((4, 4))
    _record(db_conn, hdf5_dir, "case_e", "run1", "abc123", "h1", "v1", field, accepted=True)

    manifest = Manifest(commit="abc123", input_hash="h2", solver_version="v1")
    decision = gate.evaluate_gate(db_conn, "case_e", manifest, hash_array_content(field))
    assert decision.accepted
    assert decision.reason == "manifest_changed"


def test_solver_version_changed_alone_counts_as_manifest_changed(db_conn, hdf5_dir):
    field = np.ones((4, 4))
    _record(db_conn, hdf5_dir, "case_f", "run1", "abc123", "h1", "v1", field, accepted=True)

    changed = field.copy()
    changed[1, 1] = -3.0
    manifest = Manifest(commit="abc123", input_hash="h1", solver_version="v2")
    decision = gate.evaluate_gate(db_conn, "case_f", manifest, hash_array_content(changed))
    assert decision.accepted
    assert decision.reason == "manifest_changed"


def test_refused_run_does_not_become_new_baseline(db_conn, hdf5_dir):
    deck = base_deck()
    d1 = campaign.run_case(db_conn, "case_g", deck, hdf5_dir, commit_override="c1")
    assert d1.accepted

    # Silent drift: same manifest claimed, but a different deck actually solved.
    drifted_deck = base_deck(seed=999)
    d2 = campaign.run_case(
        db_conn,
        "case_g",
        drifted_deck,
        hdf5_dir,
        commit_override="c1",
        input_hash_override=hash_input_deck(deck),
    )
    assert not d2.accepted

    all_rows = storage.all_runs_for_case(db_conn, "case_g")
    accepted_ids = [r.run_id for r in all_rows if r.accepted]
    assert len(accepted_ids) == 1
