import json

from campaign_manager.cli import main

from .conftest import base_deck


def test_cli_run_case_accepts_first_run(tmp_path):
    deck = base_deck(n=8, iterations=20)
    deck_path = tmp_path / "deck.json"
    deck_path.write_text(json.dumps(deck))

    exit_code = main(
        [
            "run-case",
            "--case-name",
            "cli_case",
            "--input-deck",
            str(deck_path),
            "--run-id",
            "cli_run_1",
            "--db-path",
            str(tmp_path / "c.sqlite"),
            "--hdf5-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "runs" / "cli_run_1.h5").exists()
