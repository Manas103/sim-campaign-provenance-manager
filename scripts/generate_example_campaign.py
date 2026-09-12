"""Generate a small example campaign: a few runs recorded through the real
pipeline (solver, manifest, SQLite, HDF5), plus two example PyVista renders
committed under data/plots/ for the README.

Run with:
    venv\\Scripts\\python.exe scripts\\generate_example_campaign.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from campaign_manager import campaign, storage
from campaign_manager.render import render_field_png
from campaign_manager.storage import read_run_hdf5

DB_PATH = REPO_ROOT / "data" / "example_campaign.sqlite"
HDF5_DIR = REPO_ROOT / "data" / "runs"
PLOTS_DIR = REPO_ROOT / "data" / "plots"


def main():
    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with storage.connect(DB_PATH) as conn:
        deck_a = {"n": 40, "iterations": 300, "boundary_temp": 1.0, "source_amplitude": 5.0, "seed": 1}
        deck_b = {"n": 40, "iterations": 300, "boundary_temp": 2.5, "source_amplitude": 8.0, "seed": 2}

        d1 = campaign.run_case(conn, "example_case_a", deck_a, HDF5_DIR, repo_path=REPO_ROOT)
        print("example_case_a run 1:", d1.accepted, d1.reason)
        d2 = campaign.run_case(conn, "example_case_b", deck_b, HDF5_DIR, repo_path=REPO_ROOT)
        print("example_case_b run 1:", d2.accepted, d2.reason)

        row_a = storage.get_latest_accepted(conn, "example_case_a")
        row_b = storage.get_latest_accepted(conn, "example_case_b")

        field_a = read_run_hdf5(Path(row_a.hdf5_path))
        field_b = read_run_hdf5(Path(row_b.hdf5_path))

        render_field_png(field_a, PLOTS_DIR / "example_case_a.png", title="example_case_a")
        render_field_png(field_b, PLOTS_DIR / "example_case_b.png", title="example_case_b")
        print("wrote", PLOTS_DIR / "example_case_a.png")
        print("wrote", PLOTS_DIR / "example_case_b.png")


if __name__ == "__main__":
    main()
