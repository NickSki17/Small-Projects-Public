import numpy as np
import pytest

from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization import (
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


def test_invalid_baseline_outputs_are_rejected():
    outputs = np.array([10.0, 20.0, 30.0, 0.01, 0.02, 5.0, 0.003])
    outputs[0] = 0.0

    with pytest.raises(ValueError):
        constraint_margins(outputs, outputs)