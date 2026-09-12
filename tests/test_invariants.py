"""Invariant checks that hold regardless of which gate decisions were made:

1. Every accepted run's recorded input hash actually matches a fresh
   recomputation of the hash of its own recorded input deck JSON.
2. The SQLite `runs` table and the HDF5 files on disk never disagree about
   which runs exist: every row's hdf5_path exists on disk, and every .h5
   file under the run directory has a corresponding row.
"""
import json
from pathlib import Path

from campaign_manager import campaign, storage
from campaign_manager.manifest import hash_input_deck

from .conftest import base_deck


def test_recorded_input_hash_matches_recomputed_hash_from_stored_deck(db_conn, hdf5_dir):
    for i in range(15):
        deck = base_deck(seed=i, n=8 + (i % 5))
        campaign.run_case(db_conn, f"invariant_case_{i}", deck, hdf5_dir, commit_override="c1")

    for row in storage.all_runs(db_conn):
        stored_deck = json.loads(row.input_deck_json)
        recomputed = hash_input_deck(stored_deck)
        assert recomputed == row.manifest.input_hash, (
            f"run {row.run_id}: recorded input_hash does not match a fresh "
            f"recompute from its own stored input deck"
        )


def test_sqlite_runs_and_hdf5_files_never_disagree(db_conn, hdf5_dir):
    for i in range(15):
        deck = base_deck(seed=i)
        campaign.run_case(db_conn, f"disk_case_{i}", deck, hdf5_dir, commit_override="c1")
        # A second run for some cases, occasionally drifted, to also populate refused runs.
        if i % 3 == 0:
            campaign.run_case(
                db_conn,
                f"disk_case_{i}",
                base_deck(seed=i, n=99),
                hdf5_dir,
                commit_override="c1",
                input_hash_override=hash_input_deck(deck),
            )

    rows = storage.all_runs(db_conn)
    for row in rows:
        assert Path(row.hdf5_path).exists(), f"run {row.run_id} has no HDF5 file on disk"

    disk_files = {p.stem for p in Path(hdf5_dir).glob("*.h5")}
    db_run_ids = {row.run_id for row in rows}
    assert disk_files == db_run_ids, (
        f"disk/db disagreement: on disk only {disk_files - db_run_ids}, "
        f"in db only {db_run_ids - disk_files}"
    )
