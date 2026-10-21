import numpy as np
import pytest
import torch

import src.optimization as optimization_module
from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization import (
    SLSQP_FINITE_DIFFERENCE_STEP,
    conservative_allowance_envelope,
    constraint_margins,
    scale_design,
    total_inertia,
    unscale_design,
)


def test_midpoint_scales_to_half():
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)
    midpoint = 0.5 * (lower + upper)

    np.testing.assert_allclose(scale_design(midpoint), 0.5)


def test_scale_and_unscale_round_trip():
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    upper = np.asarray(UPPER_BOUNDS, dtype=float)
    design = lower + 0.37 * (upper - lower)

    recovered = unscale_design(scale_design(design))

    np.testing.assert_allclose(recovered, design)


def test_total_inertia():
    design = np.array([0.02, 0.03, 0.04, 10.0, 20.0, 30.0])

    assert total_inertia(design) == pytest.approx(0.09)


@pytest.mark.parametrize(
    "design",
    [
        np.ones(5),
        np.full(6, np.nan),
        np.full(6, np.inf),
    ],
)
def test_invalid_design_is_rejected(design):
    with pytest.raises(ValueError):
        scale_design(design)


def test_out_of_bounds_design_is_rejected():
    lower = np.asarray(LOWER_BOUNDS, dtype=float)
    design = lower.copy()
    design[0] -= 0.001

    with pytest.raises(ValueError):
        scale_design(design)


def test_constraint_margins_are_zero_at_baseline():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])

    np.testing.assert_allclose(
        constraint_margins(baseline, baseline),
        np.zeros(7),
    )


def test_constraint_margins_detect_improvements():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    predicted = baseline.copy()

    predicted[:3] *= 1.1
    predicted[3:] *= 0.9

    margins = constraint_margins(predicted, baseline)

    assert np.all(margins > 0.0)


def test_constraint_margins_detect_violations():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    predicted = baseline.copy()

    predicted[0] *= 0.9
    predicted[3] *= 1.1

    margins = constraint_margins(predicted, baseline)

    assert margins[0] < 0.0
    assert margins[3] < 0.0


def test_directional_allowances_correct_all_seven_outputs():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    predicted = baseline.copy()
    allowance = np.array([1.0, 2.0, 3.0, 0.001, 0.002, 0.5, 0.0003])
    margins = constraint_margins(predicted, baseline, allowance)
    expected = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
    np.testing.assert_allclose(margins, expected)


def test_directional_allowances_have_no_violation_at_corrected_threshold():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    allowance = np.array([1.0, 2.0, 3.0, 0.001, 0.002, 0.5, 0.0003])
    predicted = baseline.copy()
    predicted[:3] += allowance[:3]
    predicted[3:] -= allowance[3:]
    np.testing.assert_allclose(
        constraint_margins(predicted, baseline, allowance), 0.0, atol=1e-14
    )


def test_invalid_allowances_are_rejected():
    baseline = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    with pytest.raises(ValueError, match="allowances"):
        constraint_margins(baseline, baseline, np.full(7, np.nan))


def test_optimization_uses_continuous_envelope_of_regional_allowances():
    global_allowance = np.array([1.0, 4.0, 3.0, 0.8, 0.4, 5.0, 0.2])
    low_inertia_allowance = np.array([1.0, 2.0, 2.0, 0.6, 0.3, 4.0, 0.1])

    selected = conservative_allowance_envelope(
        global_allowance, low_inertia_allowance
    )

    np.testing.assert_array_equal(selected, global_allowance)
    assert np.all(selected >= global_allowance)
    assert np.all(selected >= low_inertia_allowance)


def test_optimization_allowance_envelope_rejects_invalid_regional_values():
    with pytest.raises(ValueError, match="seven values"):
        conservative_allowance_envelope(np.ones(6), np.ones(7))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        conservative_allowance_envelope(np.ones(7), np.array([0.0] * 6 + [-1.0]))


def test_invalid_baseline_outputs_are_rejected():
    outputs = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    outputs[0] = 0.0

    with pytest.raises(ValueError):
        constraint_margins(outputs, outputs)


def test_slsqp_uses_float32_appropriate_difference_step_and_records_candidate(monkeypatch):
    captured = {}

    class ConstantModel(torch.nn.Module):
        def forward(self, inputs):
            outputs = inputs.new_tensor([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
            return outputs.repeat(inputs.shape[0], 1)

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        captured.update(options=options, method=method)
        return optimization_module.OptimizeResult(
            x=np.asarray(x0).copy(), success=False, message="controlled test",
            nit=1, nfev=9, fun=float(fun(x0)),
        )

    monkeypatch.setattr(optimization_module, "minimize", fake_minimize)
    outcome = optimization_module.optimize_design(
        ConstantModel(), n_starts=1, solver="SLSQP"
    )

    assert captured["method"] == "SLSQP"
    assert captured["options"]["eps"] == SLSQP_FINITE_DIFFERENCE_STEP == 1e-5
    assert len(outcome.candidates) == 1
    np.testing.assert_allclose(outcome.candidates[0]["outputs"],
                               [10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    np.testing.assert_array_equal(outcome.candidates[0]["allowances"], np.zeros(7))
