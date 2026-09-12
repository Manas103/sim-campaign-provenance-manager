"""Slurm batch driver tests.

`sbatch` is not installed on this machine, so `is_slurm_available()` is
expected to be False here and the local-executor fallback path is what
actually gets exercised end to end. The slurm-branch command construction is
still tested by forcing `force_backend="slurm"` with a stubbed `sbatch` on
PATH, but that never talks to a real scheduler; it only proves the command
line built for `subprocess.run(["sbatch", ...])` is well formed.
"""
import json
from pathlib import Path

from campaign_manager import slurm

from .conftest import base_deck


def test_slurm_is_genuinely_unavailable_on_this_machine():
    assert slurm.is_slurm_available() is False, (
        "expected sbatch to be absent on this machine; if this now fails, "
        "the honest local-fallback claim in the README needs updating"
    )


def test_generate_sbatch_script_has_required_directives(tmp_path):
    cmd = slurm.build_run_command(
        case_name="demo",
        input_deck_path=tmp_path / "deck.json",
        run_id="abc123",
        db_path=tmp_path / "c.sqlite",
        hdf5_dir=tmp_path / "runs",
        repo_path=tmp_path,
    )
    script = slurm.generate_sbatch_script("demo_job", cmd, log_dir=tmp_path)
    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=demo_job" in script
    assert "#SBATCH --partition=" in script
    assert "#SBATCH --time=" in script
    assert "#SBATCH --output=" in script
    assert "campaign_manager.cli" in script
    assert "run-case" in script


def test_submit_run_local_fallback_actually_executes(tmp_path, repo_path):
    deck = base_deck(n=8, iterations=20)
    deck_path = tmp_path / "deck.json"
    deck_path.write_text(json.dumps(deck))
    hdf5_dir = tmp_path / "runs"
    db_path = tmp_path / "c.sqlite"

    cmd = slurm.build_run_command(
        case_name="local_fallback_case",
        input_deck_path=deck_path,
        run_id="local1",
        db_path=db_path,
        hdf5_dir=hdf5_dir,
        repo_path=repo_path,
    )
    result = slurm.submit_run(cmd)
    assert result.backend == "local"
    assert result.returncode == 0, result.stderr
    assert (hdf5_dir / "local1.h5").exists()


def test_submit_run_slurm_branch_builds_sbatch_command(tmp_path, monkeypatch):
    """Force the slurm branch to prove the command construction. `subprocess.run`
    is stubbed so this never depends on a real `sbatch` binary being resolvable
    on PATH (Windows CreateProcess extension resolution is not something this
    test should depend on); it proves `submit_run` builds and issues exactly
    the `["sbatch", <script>]` command, nothing more."""
    import subprocess as subprocess_module

    captured = {}

    class FakeCompletedProcess:
        returncode = 0
        stdout = "Submitted batch job 999\n"
        stderr = ""

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    script_path = tmp_path / "job.sh"
    script_path.write_text("#!/bin/bash\necho hi\n")

    result = slurm.submit_run(["echo", "hi"], sbatch_script_path=script_path, force_backend="slurm")
    assert result.backend == "slurm"
    assert result.returncode == 0
    assert "999" in result.stdout
    assert captured["cmd"] == ["sbatch", str(script_path)]
