import numpy as np

from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization_adapter import classify_candidate, select_candidate


def _margins(value=1.0):
    return np.full(7, value)


def test_feasible_candidate_is_eligible_even_when_solver_did_not_converge():
    design = np.asarray(LOWER_BOUNDS, dtype=float) + 0.2 * (
        np.asarray(UPPER_BOUNDS) - np.asarray(LOWER_BOUNDS)
    )
    result = classify_candidate(design, _margins(), _margins(), LOWER_BOUNDS,
                                UPPER_BOUNDS, baseline_inertia=1.0,
                                solver_success=False)
    assert result["accepted_for_selection"]
    assert result["solver_success"] is False


def test_solver_success_does_not_override_physics_failure():
    design = np.asarray(LOWER_BOUNDS, dtype=float) + 0.2 * (
        np.asarray(UPPER_BOUNDS) - np.asarray(LOWER_BOUNDS)
    )
    physical = _margins()
    physical[4] = -1e-3
    result = classify_candidate(design, _margins(), physical, LOWER_BOUNDS,
                                UPPER_BOUNDS, baseline_inertia=1.0,
                                solver_success=True)
    assert not result["accepted_for_selection"]
    assert "independent_physics_failed" in result["rejection_reasons"]


def test_out_of_bounds_and_nonfinite_designs_are_rejected():
    design = np.asarray(LOWER_BOUNDS, dtype=float) - 1.0
    result = classify_candidate(design, _margins(), _margins(), LOWER_BOUNDS,
                                UPPER_BOUNDS, baseline_inertia=1.0)
    assert not result["in_bounds"]
    assert not result["accepted_for_selection"]
    design[0] = np.nan
    result = classify_candidate(design, _margins(), _margins(), LOWER_BOUNDS,
                                UPPER_BOUNDS, baseline_inertia=1.0)
    assert "invalid_or_nonfinite_candidate" in result["rejection_reasons"]


def test_select_lowest_inertia_accepted_or_fallback_none():
    candidates = {
        "more": {"accepted_for_selection": True, "total_inertia": 0.8},
        "best": {"accepted_for_selection": True, "total_inertia": 0.4},
        "rejected": {"accepted_for_selection": False, "total_inertia": 0.1},
    }
    assert select_candidate(candidates) == "best"
    assert select_candidate({"bad": candidates["rejected"]}) is None
