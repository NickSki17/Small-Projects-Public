
"""Tests for the optimization orchestration and true-physics validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import scripts.optimize_design as design_script
from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    OptimizationOutcome,
)
from src.surrogate import SurrogateMLP


def make_physics_outputs(
    values: list[float] | None = None,
) -> dict[str, float]:
    """Create outputs using the production target names and order."""
    if values is None:
        values = [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]

    return dict(zip(TARGET_COLUMNS, values, strict=True))


def make_outcome() -> OptimizationOutcome:
    """Create a deterministic baseline-to-candidate optimization outcome."""
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)

    baseline = 0.5 * (lower + upper)
    optimized = baseline.copy()
    optimized[:3] = lower[:3] + 0.25 * (upper[:3] - lower[:3])

    baseline_outputs = np.array(
        [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]
    )
    optimized_outputs = np.array(
        [11.0, 21.0, 31.0, 0.18, 0.27, 3.8, 0.09]
    )

    return OptimizationOutcome(
        baseline_design=baseline,
        baseline_outputs=baseline_outputs,
        optimized_design=optimized,
        optimized_outputs=optimized_outputs,
        solver_result=None,
        surrogate_feasible=True,
    )


def test_physics_output_vector_uses_target_order(monkeypatch):
    expected = np.array([10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1])

    monkeypatch.setattr(
        design_script,
        "evaluate_design",
        lambda design: make_physics_outputs(expected.tolist()),
    )

    result = design_script.physics_output_vector(
        np.asarray(LOWER_BOUNDS, dtype=float)
    )

    np.testing.assert_allclose(result, expected)


def test_physics_output_vector_rejects_wrong_design_shape():
    with pytest.raises(ValueError, match="exactly six"):
        design_script.physics_output_vector(np.ones(5))


def test_physics_output_vector_rejects_nonfinite_design():
    design = np.asarray(LOWER_BOUNDS, dtype=float)
    design[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        design_script.physics_output_vector(design)


def test_physics_output_vector_reports_missing_output(monkeypatch):
    monkeypatch.setattr(
        design_script,
        "evaluate_design",
        lambda design: {"omega1": 1.0},
    )

    with pytest.raises(KeyError, match="missing required outputs"):
        design_script.physics_output_vector(
            np.asarray(LOWER_BOUNDS, dtype=float)
        )


def test_physics_output_vector_rejects_nonfinite_result(monkeypatch):
    outputs = make_physics_outputs()
    outputs["omega2"] = float("nan")

    monkeypatch.setattr(
        design_script,
        "evaluate_design",
        lambda design: outputs,
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        design_script.physics_output_vector(
            np.asarray(LOWER_BOUNDS, dtype=float)
        )


def test_run_design_study_validates_baseline_and_candidate(monkeypatch):
    outcome = make_outcome()
    calls: list[np.ndarray] = []

    monkeypatch.setattr(
        design_script,
        "optimize_design",
        lambda model, **kwargs: outcome,
    )

    def fake_evaluate(design):
        calls.append(np.array(design, dtype=float, copy=True))
        if np.allclose(design, outcome.baseline_design):
            return make_physics_outputs(
                [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]
            )
        return make_physics_outputs(
            [11.0, 21.0, 31.0, 0.18, 0.27, 3.8, 0.09]
        )

    monkeypatch.setattr(design_script, "evaluate_design", fake_evaluate)

    result = design_script.run_design_study(model=object())

    assert len(calls) == 2
    assert result.outcome is outcome
    assert result.true_physics_feasible
    assert result.true_constraint_margins.shape == (len(TARGET_COLUMNS),)


def test_run_design_study_detects_true_physics_constraint_violation(
    monkeypatch,
):
    outcome = make_outcome()

    monkeypatch.setattr(
        design_script,
        "optimize_design",
        lambda model, **kwargs: outcome,
    )

    def fake_evaluate(design):
        if np.allclose(design, outcome.baseline_design):
            return make_physics_outputs(
                [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]
            )

        # omega1 is below baseline, and theta2_max is above baseline.
        return make_physics_outputs(
            [9.0, 21.0, 31.0, 0.22, 0.27, 3.8, 0.09]
        )

    monkeypatch.setattr(design_script, "evaluate_design", fake_evaluate)

    result = design_script.run_design_study(model=object())

    assert not result.true_physics_feasible
    assert np.any(result.true_constraint_margins < 0.0)


def test_run_design_study_rejects_nonpositive_baseline_outputs(monkeypatch):
    outcome = make_outcome()

    monkeypatch.setattr(
        design_script,
        "optimize_design",
        lambda model, **kwargs: outcome,
    )
    monkeypatch.setattr(
        design_script,
        "evaluate_design",
        lambda design: make_physics_outputs(
            [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.0]
        ),
    )

    with pytest.raises(ValueError, match="must be positive"):
        design_script.run_design_study(model=object())


def test_load_surrogate_reads_saved_metadata_and_weights(tmp_path):
    model_dir = tmp_path / "surrogate"
    model_dir.mkdir()

    model = SurrogateMLP(
        input_dim=len(INPUT_COLUMNS),
        output_dim=len(TARGET_COLUMNS),
        hidden_dims=(16, 16),
    )

    metadata = {
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "input_dim": len(INPUT_COLUMNS),
        "output_dim": len(TARGET_COLUMNS),
        "hidden_dims": [16, 16],
        "seed": 42,
    }

    (model_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), model_dir / "model_state.pt")

    loaded_model, loaded_metadata = design_script.load_surrogate(model_dir)

    assert isinstance(loaded_model, SurrogateMLP)
    assert loaded_model.training is False
    assert loaded_metadata == metadata

    sample = torch.zeros((2, len(INPUT_COLUMNS)), dtype=torch.float32)
    with torch.no_grad():
        predictions = loaded_model(sample)

    assert tuple(predictions.shape) == (2, len(TARGET_COLUMNS))


def test_load_surrogate_rejects_incompatible_target_columns(tmp_path):
    model_dir = tmp_path / "surrogate"
    model_dir.mkdir()

    metadata = {
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": ["wrong_target"],
        "input_dim": len(INPUT_COLUMNS),
        "output_dim": len(TARGET_COLUMNS),
        "hidden_dims": [16, 16],
    }

    (model_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Supply valid weights so the test reaches metadata validation.
    model = design_script.SurrogateMLP(
        input_dim=len(INPUT_COLUMNS),
        output_dim=len(TARGET_COLUMNS),
        hidden_dims=(16, 16),
    )
    torch.save(model.state_dict(), model_dir / "model_state.pt")

    with pytest.raises(ValueError, match="target_columns"):
        design_script.load_surrogate(model_dir)



def test_build_comparison_table_has_baseline_and_optimized_rows():
    outcome = make_outcome()

    # Build the study result directly to isolate table formatting.
    study = design_script.DesignStudyResult(
        outcome=outcome,
        baseline_physics_outputs=np.array(
            [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]
        ),
        optimized_physics_outputs=np.array(
            [11.0, 21.0, 31.0, 0.18, 0.27, 3.8, 0.09]
        ),
        true_constraint_margins=np.zeros(len(TARGET_COLUMNS)),
        true_physics_feasible=True,
    )

    table = design_script.build_comparison_table(study)

    assert table["case"].tolist() == ["baseline", "optimized"]
    assert "J1" in table.columns
    assert "omega1_surrogate" in table.columns
    assert "omega1_physics" in table.columns
    assert "total_inertia" in table.columns



def test_save_results_writes_csv_json_and_figures(tmp_path):
    outcome = make_outcome()
    study = design_script.DesignStudyResult(
        outcome=outcome,
        baseline_physics_outputs=np.array(
            [10.0, 20.0, 30.0, 0.2, 0.3, 4.0, 0.1]
        ),
        optimized_physics_outputs=np.array(
            [11.0, 21.0, 31.0, 0.18, 0.27, 3.8, 0.09]
        ),
        true_constraint_margins=np.array(
            [0.1, 0.05, 0.03, 0.1, 0.1, 0.05, 0.1]
        ),
        true_physics_feasible=True,
    )

    design_script.save_results(study, tmp_path)

    csv_path = tmp_path / "design_comparison.csv"
    json_path = tmp_path / "optimization_summary.json"
    design_figure = tmp_path / "figures" / "design_comparison.png"
    output_figure = tmp_path / "figures" / "output_comparison.png"

    assert csv_path.is_file()
    assert json_path.is_file()
    assert design_figure.is_file()
    assert output_figure.is_file()

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["surrogate_feasible"] is True
    assert summary["true_physics_feasible"] is True
    assert "optimized_design" in summary
    assert "optimized_physics_outputs" in summary
