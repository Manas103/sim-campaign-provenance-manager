"""Automated rendering of a run's HDF5 output to a PNG snapshot.

Primary path is PyVista in off-screen mode (`pyvista.OFF_SCREEN = True`),
which was confirmed to genuinely produce a rendered PNG on this machine
(Windows 11, VTK's software/hardware off-screen path) without a visible
window or a virtual framebuffer. If PyVista's off-screen renderer cannot
initialize on a given machine (no GPU driver, no software rasterizer
available at all), `render_field_png` raises `RenderUnavailable` and callers
(or tests) can fall back to `render_field_png_matplotlib`, a plain 2D
imshow-based substitute documented here as a fallback, not silently used to
replace the PyVista claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class RenderUnavailable(RuntimeError):
    pass


def render_field_png(field: np.ndarray, png_path: Path, title: str = "") -> Path:
    import pyvista as pv

    pv.OFF_SCREEN = True
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        grid = pv.ImageData()
        n0, n1 = field.shape
        grid.dimensions = (n1, n0, 1)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data["temperature"] = field.ravel(order="C")

        plotter = pv.Plotter(off_screen=True, window_size=(480, 480))
        plotter.add_mesh(grid, scalars="temperature", cmap="viridis", show_scalar_bar=True)
        plotter.view_xy()
        if title:
            plotter.add_text(title, font_size=10)
        plotter.screenshot(str(png_path))
        plotter.close()
    except Exception as exc:  # pragma: no cover - only exercised on a headless-render failure
        raise RenderUnavailable(f"pyvista off-screen render failed: {exc}") from exc

    return png_path


def render_field_png_matplotlib(field: np.ndarray, png_path: Path, title: str = "") -> Path:
    """Documented fallback substitute path, used only if PyVista is unavailable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(field, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.savefig(str(png_path), dpi=100)
    plt.close(fig)
    return png_path
