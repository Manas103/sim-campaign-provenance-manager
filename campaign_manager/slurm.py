"""Slurm batch job script generation, with a local subprocess fallback.

This machine does not have Slurm installed (no cluster, no `sbatch` on PATH).
That is the honest, expected state for a laptop-scale portfolio project, not
a bug. `is_slurm_available()` reflects reality: on this machine it is always
False, so `submit_run()` always takes the local-executor fallback path. The
sbatch script this module generates is a real, syntactically valid Slurm
batch script (directives, job name, output redirection, the actual run
command), and a test constructs it and checks its shape, but it has never
been handed to a real `sbatch` on a real scheduler. Say so plainly in the
README rather than implying otherwise.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


def is_slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def build_run_command(
    case_name: str,
    input_deck_path: Path,
    run_id: str,
    db_path: Path,
    hdf5_dir: Path,
    repo_path: Path,
    checkpoint_at: Optional[int] = None,
    python_exe: Optional[str] = None,
) -> list[str]:
    exe = python_exe or sys.executable
    cmd = [
        exe,
        "-m",
        "campaign_manager.cli",
        "run-case",
        "--case-name",
        case_name,
        "--input-deck",
        str(input_deck_path),
        "--run-id",
        run_id,
        "--db-path",
        str(db_path),
        "--hdf5-dir",
        str(hdf5_dir),
        "--repo-path",
        str(repo_path),
    ]
    if checkpoint_at is not None:
        cmd += ["--checkpoint-at", str(checkpoint_at)]
    return cmd


def generate_sbatch_script(
    job_name: str,
    run_command: Sequence[str],
    log_dir: Path,
    partition: str = "compute",
    time_limit: str = "00:10:00",
    cpus: int = 1,
) -> str:
    """Return the text of a real sbatch-compatible batch script."""
    log_dir = Path(log_dir)
    quoted_cmd = " ".join(_shell_quote(part) for part in run_command)
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --output={log_dir.as_posix()}/{job_name}.%j.out\n"
        f"#SBATCH --error={log_dir.as_posix()}/{job_name}.%j.err\n"
        "\n"
        f"{quoted_cmd}\n"
    )


def _shell_quote(part: str) -> str:
    if any(c in part for c in " \t\"'"):
        return '"' + part.replace('"', '\\"') + '"'
    return part


@dataclass
class SubmitResult:
    backend: str  # "slurm" or "local"
    returncode: int
    stdout: str
    stderr: str


def submit_run(
    run_command: Sequence[str],
    sbatch_script_path: Optional[Path] = None,
    force_backend: Optional[str] = None,
) -> SubmitResult:
    """Submit a run either via real `sbatch` (if present) or a local subprocess.

    `force_backend` is for tests only, to exercise the slurm-branch code path
    (command construction) via a monkeypatched `shutil.which` without ever
    invoking a real scheduler.
    """
    use_slurm = is_slurm_available() if force_backend is None else (force_backend == "slurm")

    if use_slurm:
        if sbatch_script_path is None:
            raise ValueError("sbatch_script_path is required when submitting via slurm")
        result = subprocess.run(
            ["sbatch", str(sbatch_script_path)], capture_output=True, text=True
        )
        return SubmitResult(
            backend="slurm",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    result = subprocess.run(list(run_command), capture_output=True, text=True)
    return SubmitResult(
        backend="local",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
