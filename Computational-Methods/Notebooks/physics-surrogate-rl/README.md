# Physics-Based Design Optimization and Reinforcement Learning

Project 2 in the Computational Methods repository.

This project develops a computational engineering pipeline combining numerical simulation, mechanical design optimization, neural-network surrogate modeling, and reinforcement-learning control.

The objective is to investigate whether a learned surrogate can accelerate mechanical design exploration while maintaining acceptable agreement with the original physics model. The project will then extend the optimized system into a closed-loop control problem and compare reinforcement learning against a conventional PID controller.

## Quick Start: Run Tests

**Run tests from the `physics-surrogate-rl` project root**, not from inside `tests/`.

From the repository root in PowerShell:

```powershell
cd Computational-Methods\Notebooks\physics-surrogate-rl
python -m pytest -v
```

If your terminal is already in the `physics-surrogate-rl` directory, run only:

```powershell
python -m pytest -v
```

The last verified run, on October 16, 2026, reported **56 passed in 2.13 seconds**.

A successful test run confirms that the collected tests passed. Run the suite again after changing code.

## Project Workflow

Mechanical design parameters  
→ Physics model  
→ Modal analysis and numerical simulation  
→ Engineering response metrics  
→ Synthetic dataset  
→ Neural-network surrogate  
→ Constrained design optimization  
→ True-physics validation  
→ Gymnasium control environment  
→ PPO versus PID evaluation  
→ Performance and robustness analysis

**Core principle:** The original physics model remains the source of truth. Surrogate predictions and controller performance must be evaluated against defined engineering criteria.

## Mechanical System

The project uses a three-degree-of-freedom (3-DOF) linear mass-spring-damper system:

```text
Wall ── k1, c1 ── m1 ── k2, c2 ── m2 ── k3, c3 ── m3
```

The governing equation is

\[
M\ddot{x}+C\dot{x}+Kx=Bu
\]

where:

- \(M\): mass matrix
- \(C\): damping matrix
- \(K\): stiffness matrix
- \(B\): force-input matrix
- \(u\): applied force input
- \(x\): displacement vector

Passive vibration analysis uses zero external force. Active control will be introduced later through the force-input term.

The initial design variables are

\[
\theta=[m_1,m_2,m_3,k_1,k_2,k_3]
\]

Damping coefficients will initially remain fixed to keep the design study focused.

## Implementation Status

### Phase 1 — Physics Foundation: Complete

All four modules have been implemented and the current test suite passes.

- [x] `src/dynamics.py` — Physical model and system matrices
- [x] `src/modal.py` — Generalized eigenvalue modal analysis
- [x] `src/simulation.py` — Time-domain integration using SciPy `solve_ivp`
- [x] `src/metrics.py` — Engineering response metrics and FFT diagnostic

### Implemented Functionality

**`src/dynamics.py`**

- Constructs mass, stiffness, and damping matrices.
- Computes the first-order state derivative.
- Supports passive dynamics and constant force inputs.
- Validates matrix-building inputs and state/input dimensions.

**`src/modal.py`**

- Solves the generalized eigenvalue problem \(K\phi=\lambda M\phi\).
- Calculates natural angular frequencies in rad/s.
- Returns mass-normalized mode shapes.
- Converts angular frequencies to Hz.
- Checks matrix dimensions, symmetry, mass-matrix positive definiteness, and positive modal eigenvalues.

**`src/simulation.py`**

- Integrates the six-state system using `scipy.integrate.solve_ivp`.
- Uses the state vector

  \[
  z=[x_1,x_2,x_3,\dot{x}_1,\dot{x}_2,\dot{x}_3]
  \]

- Supports zero input or a constant force input.
- Returns the simulation time vector and state history.
- Validates simulation inputs and reports integration failures.

**`src/metrics.py`**

- Reconstructs acceleration from the governing mechanical equation.
- Calculates maximum absolute displacement (`xmax`).
- Calculates maximum absolute acceleration (`amax`).
- Calculates system-level RMS displacement (`x_rms`), weighting the three DOFs equally.
- Provides `dominant_frequency()` for uniformly sampled response signals.

### Test Status

| Test file | Tests passed |
|---|---:|
| `tests/test_dynamics.py` | 17 |
| `tests/test_modal.py` | 13 |
| `tests/test_simulation.py` | 12 |
| `tests/test_metrics.py` | 14 |
| **Total** | **56/56** |

The tests cover matrix construction, physical equations, modal properties, numerical integration, response metrics, force inputs, and invalid-input cases.

The current verified checkpoint is **56 passing tests**. These tests validate the implemented behaviors they cover; additional tests will be needed as the project expands.

## Planned Engineering Metrics and Dataset

The next stage is to generate synthetic data from physically valid mechanical designs.

Each sample will contain six design inputs:

\[
[m_1,m_2,m_3,k_1,k_2,k_3]
\]

and six target outputs:

\[
[\omega_1,\omega_2,\omega_3,x_{\max},a_{\max},x_{\mathrm{RMS}}]
\]

The initial dataset target is approximately 3,000–10,000 designs, subject to simulation cost and data quality.

Before generating the full dataset, define and test:

- Physically valid parameter bounds.
- A reproducible random seed.
- Initial conditions for passive vibration simulations.
- Fixed damping coefficients.
- Simulation duration and time resolution.
- Handling of failed simulations and invalid samples.

The excitation protocol must be nontrivial. A zero initial state with zero external force produces a zero response and therefore provides no useful displacement or acceleration data.

## Planned Surrogate Model

The planned model is a PyTorch multilayer perceptron:

```text
6 design inputs
      ↓
64 neurons
      ↓
64 neurons
      ↓
6 predicted outputs
```

The model will use standardized inputs and outputs, separate training/validation/test sets, and regression metrics including MAE, RMSE, and \(R^2\).

The surrogate will be evaluated on held-out data. Its predictions will not replace the original numerical physics model for final design validation.

## Planned Design Optimization

The design study will minimize total mass subject to mechanical response constraints.

The planned approach uses SciPy's SLSQP constrained optimizer, with physical bounds and constraints such as:

- Minimum first natural frequency.
- Maximum allowable displacement.
- Positive masses and stiffnesses.

The optimized design must be reevaluated using the original numerical physics model. Results will include baseline-versus-optimized comparisons and surrogate-versus-physics errors.

## Planned Reinforcement-Learning Control

After design optimization and physics validation, the project will introduce a Gymnasium environment using the same physical model.

The initial control study will focus on controlling the first mass rather than attempting full 3-DOF control immediately.

Planned evaluation:

- PPO reinforcement-learning controller.
- Conventional PID controller.
- Position-tracking RMSE.
- Settling time and overshoot.
- Control effort and peak force.
- Robustness to approximately ±10% mass or payload perturbations.

The goal is an evidence-based comparison. PPO will not be assumed to outperform PID.

## Planned Repository Structure

```text
physics-surrogate-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── dynamics.py
│   ├── modal.py
│   ├── simulation.py
│   ├── metrics.py
│   ├── surrogate.py
│   ├── optimization.py
│   ├── environment.py
│   └── controllers.py
├── scripts/
│   ├── generate_dataset.py
│   ├── train_surrogate.py
│   ├── optimize_design.py
│   └── train_rl.py
├── tests/
│   ├── test_dynamics.py
│   ├── test_modal.py
│   ├── test_simulation.py
│   └── test_metrics.py
├── data/
└── figures/
```

This is the target structure. Some planned modules, scripts, or directories may not have been created yet.

## Next Steps

1. **Define dataset-generation settings:** parameter bounds, initial conditions, damping, simulation duration, time resolution, and random seed.
2. Implement and test `scripts/generate_dataset.py`.
3. Generate reproducible synthetic data and inspect its distributions and physical validity.
4. Implement and evaluate the neural surrogate.
5. Implement constrained design optimization and true-physics validation.
6. Build the Gymnasium environment and train PPO.
7. Implement the PID baseline and compare tracking performance.
8. Evaluate robustness, computational cost, and limitations.

**Immediate restart point:** Step 5 — synthetic dataset generation. The physics foundation is complete and has 56 passing tests. Do not start surrogate training until the dataset generator and its outputs have been validated.

## Reproducibility and Limitations

- Use fixed random seeds for dataset generation and model experiments.
- Keep the numerical physics model independent of the surrogate and controller implementations.
- Record model settings, simulation settings, and evaluation metrics.
- Avoid committing large datasets or model checkpoints unnecessarily; document how to regenerate them.
- The initial model is linear and simulated. Results do not establish real-world performance without further modeling and experimental validation.
- Surrogate accuracy depends on the sampled design space and training-data coverage.
- Reinforcement-learning performance depends on the reward function, training conditions, and evaluation protocol.

## Technology Stack

- Python
- NumPy
- SciPy
- Matplotlib
- Pandas
- PyTorch
- scikit-learn
- Gymnasium
- Stable-Baselines3
- pytest

The project demonstrates computational mechanics, numerical simulation, data-driven approximation, constrained engineering optimization, and control-system evaluation in a reproducible workflow.