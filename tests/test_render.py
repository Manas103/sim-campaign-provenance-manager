"""Automated PyVista off-screen render of a run's HDF5 output to PNG.

Confirmed to genuinely work headless on this machine (Windows 11, VTK's
software/hardware off-screen path, no visible window). If PyVista's
off-screen renderer cannot initialize at all on a given machine, this test
skips cleanly rather than silently passing on the documented matplotlib
substitute; see README for that fallback.
"""
import pytest

from campaign_manager.render import RenderUnavailable, render_field_png
from campaign_manager.solver import run_solver
from campaign_manager.storage import write_run_hdf5, read_run_hdf5
from campaign_manager.manifest import Manifest

from .conftest import base_deck


def test_pyvista_offscreen_render_produces_a_real_png(tmp_path):
    deck = base_deck(n=20, iterations=150)
    field = run_solver(deck)

    manifest = Manifest(commit="c1", input_hash="h1", solver_version="v1")
    hdf5_path = tmp_path / "render_case.h5"
    write_run_hdf5(hdf5_path, field, "render_case", "render_case", manifest)
    loaded = read_run_hdf5(hdf5_path)

    png_path = tmp_path / "render_case.png"
    try:
        render_field_png(loaded, png_path, title="render_case")
    except RenderUnavailable as exc:
        pytest.skip(f"pyvista off-screen renderer unavailable on this machine: {exc}")

    assert png_path.exists()
    assert png_path.stat().st_size > 1000
