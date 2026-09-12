"""200 clean campaigns: every manifest change is a genuine, honestly recorded
change (new commit and/or new input deck and/or new solver version, matching
what was actually solved), or a genuine exact rerun with nothing changed at
all. The gate must never refuse a legitimate result as drift.

100 campaigns make a real legitimate change every step (new commit id and/or
a real edited input deck, hashed correctly). 100 campaigns are plain exact
reruns of an unchanged case (same deck, same commit, same solver version).

Run with:
    venv\\Scripts\\python.exe scripts\\false_flag_sweep.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign_manager import campaign, storage

BASE_DECK = {
    "n": 12,
    "iterations": 90,
    "boundary_temp": 1.0,
    "source_amplitude": 4.0,
    "seed": 7,
}


def build_legit_change_deck(base: dict, rng: random.Random) -> dict:
    deck = dict(base)
    knob = rng.choice(["boundary_temp", "source_amplitude", "seed", "iterations", "n"])
    if knob == "boundary_temp":
        deck["boundary_temp"] += rng.uniform(0.05, 0.5)
    elif knob == "source_amplitude":
        deck["source_amplitude"] += rng.uniform(0.1, 2.0)
    elif knob == "seed":
        deck["seed"] = rng.randint(0, 100_000)
    elif knob == "iterations":
        deck["iterations"] += rng.randint(1, 20)
    elif knob == "n":
        deck["n"] += rng.choice([-2, -1, 1, 2])
    return deck


def run_false_flag_sweep(n_legit_change: int = 100, n_exact_rerun: int = 100, seed: int = 2):
    rng = random.Random(seed)
    false_flags = []
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        hdf5_dir = tmp_path / "runs"
        db_path = tmp_path / "campaign.sqlite"
        with storage.connect(db_path) as conn:
            # Track state per case so each campaign's manifest genuinely reflects
            # what changed, the way a real campaign log would.
            for i in range(n_legit_change):
                case_name = f"clean_change_case_{i}"
                commit_1 = f"commit-{i}-a"
                commit_2 = f"commit-{i}-b"
                deck_1 = dict(BASE_DECK)
                d1 = campaign.run_case(conn, case_name, deck_1, hdf5_dir, commit_override=commit_1)
                total += 1
                if not d1.accepted:
                    false_flags.append((case_name, "initial", d1.reason))

                deck_2 = build_legit_change_deck(deck_1, rng)
                # A real code or input change: new commit AND the manifest is built
                # honestly from the deck actually solved (no override).
                d2 = campaign.run_case(conn, case_name, deck_2, hdf5_dir, commit_override=commit_2)
                total += 1
                if not d2.accepted:
                    false_flags.append((case_name, "legit_change", d2.reason))

            for i in range(n_exact_rerun):
                case_name = f"clean_rerun_case_{i}"
                commit = f"commit-rerun-{i}"
                deck = dict(BASE_DECK)
                deck["seed"] = i
                d1 = campaign.run_case(conn, case_name, deck, hdf5_dir, commit_override=commit)
                total += 1
                if not d1.accepted:
                    false_flags.append((case_name, "initial", d1.reason))

                # Exact rerun: identical deck, identical commit, identical solver version.
                d2 = campaign.run_case(conn, case_name, dict(deck), hdf5_dir, commit_override=commit)
                total += 1
                if not d2.accepted:
                    false_flags.append((case_name, "exact_rerun", d2.reason))

    return {
        "total_campaigns": n_legit_change + n_exact_rerun,
        "total_runs": total,
        "false_flags": false_flags,
    }


if __name__ == "__main__":
    result = run_false_flag_sweep()
    print(f"clean campaigns: {result['total_campaigns']}")
    print(f"total runs executed: {result['total_runs']}")
    print(f"false drift flags: {len(result['false_flags'])} / {result['total_campaigns']}")
    if result["false_flags"]:
        print("FALSE FLAGS:")
        for case_name, phase, reason in result["false_flags"][:10]:
            print(f"  {case_name} ({phase}): {reason}")
    else:
        print("no false flags")
