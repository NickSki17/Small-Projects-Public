# Computational Methods

Computational engineering projects focused on numerical optimization, vibration analysis, and data-driven design. The work emphasizes transparent assumptions, reproducible computation, and validation appropriate to each model's scope.

## Projects

### Numerical Optimization and Vibration Analysis

**Location:** [`Notebooks/optimization-project/`](Notebooks/optimization-project/)

An implemented collection of numerical-method benchmarks and a simplified mechanical vibration study. The Python analyses compare Gradient Descent, Newton's method, and SciPy BFGS on five test functions, and examine modal frequencies and time/frequency-domain responses for an undamped fixed-fixed mass-spring chain.

The scripts, assumptions, commands, and known numerical limitations are described in the [project README](Notebooks/optimization-project/optimization-project-code/README.md). An exploratory notebook is also retained at [`Notebooks/optimization_and_vibration_analysis(original).ipynb`](Notebooks/optimization_and_vibration_analysis(original).ipynb).

### Reduced-Order Rotational-Torsional Design Optimization

**Location:** [`Notebooks/robotics-design-optimization-rl/`](Notebooks/robotics-design-optimization-rl/)

This project studies a **reduced-order 3-DOF rotational-torsional chain**—not a full rigid-body robot manipulator. Its non-RL workflow connects physics-based dataset generation, neural-network surrogate training and selection, validation-only directional error calibration, constrained total-inertia optimization, and independent physics verification.

A recorded local run reduced total inertia from `0.09` to approximately `0.0323192 kg·m²` (about 64.09%) and passed the project's independent physics checks. This is a verified result for that run, not evidence of global optimality or physical-hardware performance. The dataset is local/ignored, and a clean-checkout reproduction has not been established. See the [project README](Notebooks/robotics-design-optimization-rl/README.md) for the run evidence, reproduction steps, limitations, and current status.

Reinforcement learning is **not implemented**: the related files are placeholders, and there is no connected control environment, training/evaluation pipeline, or RL-versus-baseline result.

## Engineering approach

These projects distinguish numerical correctness from physical validation. Results should be interpreted within the equations, assumptions, discretization, data provenance, and verification methods documented for each project. Simulation and surrogate agreement alone do not establish real-world performance.
