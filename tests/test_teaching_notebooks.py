"""Integrity checks for the executed, data-safe learning sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXPECTED_CONCEPTS = {
    "00_project_map.ipynb": ("tensor contracts", "ADE", "evidence integrity"),
    "01_geometry_splits_and_baselines.ipynb": ("anchor-local", "CTRV", "recording"),
    "02_neural_ode_and_multiple_shooting.ipynb": ("RK4", "multiple shooting", "hybrid"),
    "03_fourier_operators.ipynb": ("Fourier transform", "causality", "CTRV"),
    "04_road_lane_segmentation.ipynb": ("multilabel", "U-Net", "tolerant"),
    "05_lidar_bev_detection_and_tracking.ipynb": ("bird's-eye", "oriented", "Kalman"),
    "06_interview_capstone.ipynb": ("bootstrap", "failure", "promotion"),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_executed_notebook_sequence() -> None:
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == set(EXPECTED_CONCEPTS)
    for filename, concepts in EXPECTED_CONCEPTS.items():
        notebook = _load(NOTEBOOKS / filename)
        cells = notebook["cells"]
        assert len(cells) >= 14, filename
        markdown = "\n".join(
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
        )
        for concept in concepts:
            assert concept.casefold() in markdown.casefold(), (filename, concept)
        code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
        assert code_cells, filename
        assert all(cell["execution_count"] is not None for cell in code_cells), filename
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, filename
        image_outputs = sum(
            "image/png" in output.get("data", {})
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
        assert image_outputs >= 2, filename


def test_notebooks_do_not_embed_private_locator_or_machine_paths() -> None:
    # Assemble sentinels so the release archive itself does not contain a token
    # that a conservative byte-level privacy scanner would flag.
    forbidden = (
        "rl" + "key=",
        "dropbox." + "com/scl/",
        "@gmail." + "com",
        "d:" + "\\datasets\\",
    )
    for path in NOTEBOOKS.glob("*.ipynb"):
        lowered = path.read_text(encoding="utf-8").casefold()
        assert all(value not in lowered for value in forbidden), path.name
