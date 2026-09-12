"""Cross-check the fast gate path against the independently-coded, deliberately
slow naive oracle (full re-read of raw HDF5 bytes, rehash from scratch, linear
scan instead of an indexed query) across a mixed batch of clean and drifted
runs. Both must agree on every single decision."""
from campaign_manager import campaign, gate, storage
from campaign_manager.manifest import hash_input_deck

from .conftest import base_deck


def test_naive_oracle_agrees_with_fast_gate_on_clean_and_drifted_runs(db_conn, hdf5_dir):
    mismatches = []
    checked = 0

    for i in range(10):
        case_name = f"oracle_case_{i}"
        deck = base_deck(seed=i)

        d1 = campaign.run_case(db_conn, case_name, deck, hdf5_dir, commit_override="c1")
        r1 = storage.get_latest_accepted(db_conn, case_name)
        checked += 1
        fast = d1
        naive = gate.naive_reference_check(db_conn, case_name, r1.run_id)
        if (fast.accepted, fast.reason) != (naive.accepted, naive.reason):
            mismatches.append((case_name, "run1", fast, naive))

        if i % 2 == 0:
            # A legitimate change: new commit, real new deck, honestly hashed.
            deck2 = base_deck(seed=i, boundary_temp=1.0 + 0.1 * i)
            d2 = campaign.run_case(db_conn, case_name, deck2, hdf5_dir, commit_override="c2")
        else:
            # Silent drift: solved a different deck but claimed the original hash.
            deck2 = base_deck(seed=i, boundary_temp=5.0 + i)
            d2 = campaign.run_case(
                db_conn,
                case_name,
                deck2,
                hdf5_dir,
                commit_override="c1",
                input_hash_override=hash_input_deck(deck),
            )

        run2_id = storage.all_runs_for_case(db_conn, case_name)[-1].run_id
        checked += 1
        naive2 = gate.naive_reference_check(db_conn, case_name, run2_id)
        if (d2.accepted, d2.reason) != (naive2.accepted, naive2.reason):
            mismatches.append((case_name, "run2", d2, naive2))

    assert not mismatches, f"fast gate and naive oracle disagreed: {mismatches}"
    assert checked == 20
