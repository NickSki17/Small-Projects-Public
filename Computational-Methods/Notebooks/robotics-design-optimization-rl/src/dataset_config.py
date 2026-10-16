"""
Configuration for synthetic dataset generation for a 3-DOF
rotational torsional vibration system.

Parameter order:
[J1, J2, J3, k_theta1, k_theta2, k_theta3]

All parameter ranges are provisional until pilot validation passes.
"""

from __future__ import annotations

SEED = 42
PILOT_SAMPLES = 20
DATASET_SAMPLES = 5_000

# Design variable order:
# [J1, J2, J3, k_theta1, k_theta2, k_theta3]
#
# Moments of inertia: kg*m^2
# Torsional stiffnesses: N*m/rad
LOWER_BOUNDS = (
    0.01,
    0.01,
    0.01,
    5.0,
    5.0,
    5.0,
)

UPPER_BOUNDS = (
    0.05,
    0.05,
    0.05,
    50.0,
    50.0,
    50.0,
)

# Fixed rotational damping coefficients [c_theta1, c_theta2, c_theta3].
# Units: N*m*s/rad
DAMPINGS = (0.02, 0.02, 0.02)

# Passive free-vibration initial condition:
# [theta1, theta2, theta3, omega1, omega2, omega3]
#
# Angles: rad
# Angular velocities: rad/s
INITIAL_STATE = (0.01, 0.0, 0.0, 0.0, 0.0, 0.0)

T_START = 0.0
T_END = 8.0

# Coarse and refined output sampling intervals, in seconds.
TIME_STEP = 0.002
REFINED_TIME_STEP = 0.001

# Numerical integration tolerances.
RTOL = 1e-8
ATOL = 1e-10

# Maximum permitted relative difference between coarse and refined
# response metrics in the pilot study.
METRIC_RELATIVE_TOLERANCE = 0.01