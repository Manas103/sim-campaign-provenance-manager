"""30 seeded silent-drift scenarios: an input deck is changed in a way that
genuinely changes the solved output, but the manifest recorded for that run
still claims the ORIGINAL deck's input hash (commit and solver version also
held fixed). This is the real failure mode the gate exists to catch: the
thing tracking what changed silently fell out of sync with what actually
ran. The gate must refuse every one of these as unexplained drift.

Run with:
    venv\\Scripts\\python.exe scripts\\drift_detection_check.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign_manager import campaign, storage
from campaign_manager.manifest import hash_input_deck

BASE_DECK = {
    "n": 14,
    "iterations": 100,
    "boundary_temp": 1.0,
    "source_amplitude": 5.0,
    "seed": 42,
}

TWEAK_KINDS = [
    "boundary_temp_small",
    "boundary_temp_large",
    "source_amplitude_small",
    "source_amplitude_large",
    "seed_change",
    "iterations_off_by_one",
    "iterations_off_by_few",
    "n_change",
]


def make_tweaked_deck(kind: str, rng: random.Random) -> dict:
    deck = dict(BASE_DECK)
    if kind == "boundary_temp_small":
        deck["boundary_temp"] += rng.uniform(1e-3, 1e-2)
    elif kind == "boundary_temp_large":
        deck["boundary_temp"] += rng.uniform(0.1, 1.0)
    elif kind == "source_amplitude_small":
        deck["source_amplitude"] *= 1.0 + rng.uniform(1e-3, 1e-2)
    elif kind == "source_amplitude_large":
        deck["source_amplitude"] *= 1.0 + rng.uniform(0.1, 0.5)
    elif kind == "seed_change":
        deck["seed"] = deck["seed"] + rng.randint(1, 1000)
    elif kind == "iterations_off_by_one":
        deck["iterations"] += 1
    elif kind == "iterations_off_by_few":
        deck["iterations"] += rng.randint(2, 10)
    elif kind == "n_change":
        deck["n"] += rng.choice([-2, -1, 1, 2])
    else:
        raise ValueError(kind)
    return deck


def build_scenarios(n_scenarios: int = 30, seed: int = 1):
    rng = random.Random(seed)
    scenarios = []
    for i in range(n_scenarios):
        kind = TWEAK_KINDS[i % len(TWEAK_KINDS)]
        tweaked = make_tweaked_deck(kind, rng)
        scenarios.append({"scenario_id": i, "kind": kind, "tweaked_deck": tweaked})
    return scenarios


def run_drift_detection_sweep(n_scenarios: int = 30):
    scenarios = build_scenarios(n_scenarios)
    caught = []
    missed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        hdf5_dir = tmp_path / "runs"
        db_path = tmp_path / "campaign.sqlite"
        with storage.connect(db_path) as conn:
            for scenario in scenarios:
                case_name = f"drift_case_{scenario['scenario_id']}"
                baseline_decision = campaign.run_case(
                    conn, case_name, BASE_DECK, hdf5_dir, commit_override="c-fixed"
                )
                assert baseline_decision.accepted

                # Silent drift: solve the tweaked deck, but record the ORIGINAL deck's hash.
                drift_decision = campaign.run_case(
                    conn,
                    case_name,
                    scenario["tweaked_deck"],
                    hdf5_dir,
                    commit_override="c-fixed",
                    input_hash_override=hash_input_deck(BASE_DECK),
                )
                if not drift_decision.accepted and drift_decision.reason == "unexplained_drift":
                    caught.append(scenario)
                else:
                    missed.append(scenario)
    return {"total": len(scenarios), "caught": len(caught), "missed": missed}


if __name__ == "__main__":
    result = run_drift_detection_sweep()
    print(f"seeded drift scenarios: {result['total']}")
    print(f"caught: {result['caught']} / {result['total']}")
    if result["missed"]:
        print("MISSED:")
        for scenario in result["missed"]:
            print(f"  scenario {scenario['scenario_id']} kind={scenario['kind']}")
    else:
        print("all seeded scenarios caught")
