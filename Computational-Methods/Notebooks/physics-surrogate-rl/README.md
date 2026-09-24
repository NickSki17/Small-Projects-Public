# Physics Surrogate RL

Project 2 in the Computational Methods repository: a physics-based design
optimization, surrogate modeling, and reinforcement-learning study.

This directory currently contains the project scaffold only. Implementation
will be added incrementally, with the physical model and simulation pipeline
validated before surrogate modeling, design optimization, and control are
introduced.

## Planned workflow

Design parameters → physical model → numerical simulation → engineering
metrics → synthetic dataset → PyTorch surrogate → constrained design
optimization → true-physics validation → Gymnasium control environment → PPO
and PID comparison.

## Planned modules

- `src/dynamics.py`: 3-DOF mass-spring-damper model and system matrices.
- `src/modal.py`: generalized eigenvalue modal analysis.
- `src/simulation.py`: time-domain simulation with SciPy `solve_ivp`.
- `src/metrics.py`: engineering response metrics.
- `src/surrogate.py`: PyTorch neural surrogate.
- `src/optimization.py`: constrained SLSQP design optimization.
- `src/environment.py`: Gymnasium control environment.
- `src/controllers.py`: PID and other conventional controllers.

The dependency direction will remain one-way: physics and simulation form the
foundation, followed by metrics and data generation, then surrogate and design
optimization, with control components depending on the validated physical
model rather than on machine-learning modules.

## Status

Scaffold only. No physics, machine-learning, optimization, or reinforcement-
learning implementation has been added yet.
