import pytest

from scripts.diagnose_validation_region import dangerous_error_definition
from src.optimization import constraint_margins
import numpy as np


@pytest.mark.parametrize("target", ["omega1", "omega2", "omega3"])
def test_frequency_overprediction_is_dangerous(target):
    assert dangerous_error_definition(target) == "frequency_overprediction"


@pytest.mark.parametrize("target", ["theta2_max", "theta3_max", "alpha_max", "theta_rms"])
def test_response_underprediction_is_dangerous(target):
    assert dangerous_error_definition(target) == "maximum_response_underprediction"


def test_signed_directions_match_baseline_relative_feasibility():
    base = np.ones(7)
    allowances = np.full(7, 0.1)
    frequency_over = base.copy(); frequency_over[:3] += 0.1
    response_under = base.copy(); response_under[3:] -= 0.1
    np.testing.assert_allclose(constraint_margins(frequency_over, base, allowances)[:3], 0)
    np.testing.assert_allclose(constraint_margins(response_under, base, allowances)[3:], 0)
    assert np.all(constraint_margins(base + np.array([-.01, 0, 0, 0, 0, 0, 0]), base, allowances)[:1] < 0)
    assert np.all(constraint_margins(base + np.array([0, 0, 0, .01, 0, 0, 0]), base, allowances)[3:4] < 0)
