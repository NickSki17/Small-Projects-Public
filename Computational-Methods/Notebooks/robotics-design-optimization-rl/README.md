# Physics-Based Robotic Arm Optimization and Reinforcement Learning

Project 2 in the Computational Methods repository.

This project investigates how computational mechanics, numerical simulation, neural-network surrogate modeling, constrained design optimization, and reinforcement learning can be combined in a robotic-joint engineering study.

The objective is to develop a reproducible pipeline that predicts the dynamic response of a mechanical system, uses a learned surrogate to accelerate design exploration, validates optimized designs against the original physics model, and ultimately compares a PPO reinforcement-learning controller with a conventional PID controller.

The current physics and dataset pipeline uses a **three-degree-of-freedom rotational-torsional system** as a reduced-order model. It is not a complete six-axis robotic manipulator, and the project does not claim hardware-level validation.

## Quick Start

Run commands from the `robotics-design-optimization-rl/` project root.

Run the automated tests:

```powershell
python -m pytest -v
```

Generate a small dataset to verify the data-generation pipeline:

```powershell
python -m scripts.generate_dataset --samples 20 --output data/generator_smoke_test.csv
```

Generate the full dataset:

```powershell
python -m scripts.generate_dataset
```

Train the neural-network surrogate after the full dataset has been generated and validated:

```powershell
python -m scripts.train_surrogate
```

The training script defaults to `data/full_dataset.csv` and saves model weights, metadata, evaluation metrics, training history, and plots under `artifacts/surrogate/`.

The full dataset can take substantial time to generate because every design requires a numerical simulation. Confirm that generation finishes successfully before starting model training.

## Project Workflow

1. Define the rotational-torsional mechanical model and its physical parameters.
2. Compute natural frequencies and simulate time-domain responses.
3. Extract engineering response metrics.
4. Generate and validate a reproducible synthetic dataset.
5. Train and evaluate a neural-network surrogate.
6. Perform constrained design optimization using the surrogate.
7. Validate candidate designs with the original physics model.
8. Develop a compatible control environment.
9. Compare PPO against a conventional PID controller.
10. Evaluate predictive accuracy, computational cost, control performance, and robustness.

**Core principle:** the original physics model remains the reference for engineering validation. Surrogate predictions and learned-controller performance must be assessed using explicit metrics rather than assumed to be accurate.

## Mechanical Model

The current application is a three-degree-of-freedom rotational-torsional system. Its design variables include three rotational inertias and three torsional stiffnesses.

The general matrix equation of motion is

\[
\mathbf{J}\ddot{\boldsymbol{\theta}}
+\mathbf{C}\dot{\boldsymbol{\theta}}
+\mathbf{K}\boldsymbol{\theta}
=\boldsymbol{\tau}
\]

where:

- \(\mathbf{J}\) is the rotational inertia matrix.
- \(\mathbf{C}\) is the damping matrix.
- \(\mathbf{K}\) is the torsional stiffness matrix.
- \(\boldsymbol{\theta}\) contains angular displacements in radians.
- \(\boldsymbol{\tau}\) represents applied joint torques.

The precise matrix definitions, parameter ranges, damping assumptions, initial conditions, and applied inputs are determined by the implementation. They should be documented alongside the corresponding source code.

This is a reduced-order engineering model. It does not represent the full dynamics of a six-joint manipulator, including all actuator, transmission, gravity, and inter-joint effects.

## Synthetic Dataset

The dataset generator samples physical design parameters, runs the numerical model, and extracts response quantities.

The current generator configuration reported in the console is:

| Setting | Value |
|---|---:|
| Number of designs | 5,000 |
| Random seed | 42 |
| Output time interval | 0.002 s |
| Simulation duration | 8.0 s |
| Input variables | 6 |
| Surrogate target variables | 7 |

The default output file is `data/full_dataset.csv`.

### Surrogate inputs

- `J1`, `J2`, `J3`
- `k_theta1`, `k_theta2`, `k_theta3`

### Surrogate targets

- `omega1`, `omega2`, `omega3`: natural angular frequencies in rad/s
- `theta2_max`, `theta3_max`: peak absolute angular responses in rad
- `alpha_max`: peak angular acceleration in rad/s²
- `theta_rms`: angular-response RMS metric in rad

The dataset also contains diagnostic quantities, including `theta_max` and `theta1_max`. The generator's pilot run found that `theta1_max` was constant across the tested samples, so it is not included as a surrogate target. Constant targets provide no useful regression signal.

The dataset should be checked for missing values, non-finite values, duplicate or constant targets, and physically invalid samples before training.

## Neural-Network Surrogate

The surrogate is implemented in `src/surrogate.py` using PyTorch.

The current default architecture is a fully connected multilayer perceptron:

\[
6 \rightarrow 64 \rightarrow 64 \rightarrow 7
\]

It uses ReLU activations in its hidden layers. Input and output normalization parameters are stored with the model, allowing inference to return predictions in the original physical units.

The training workflow is implemented in `scripts/train_surrogate.py`. It includes:

- Reproducible random seeds.
- A 70% training, 15% validation, and 15% test split.
- Normalization fitted using training data only.
- AdamW optimization and mean-squared-error loss.
- Validation-based early stopping.
- Per-target MAE, RMSE, and \(R^2\) evaluation.
- Training-history and actual-versus-predicted plots.
- Saved model weights and training metadata.

The held-out test set is intended for final evaluation, not repeated architecture selection. Deeper networks may be evaluated against the current two-hidden-layer baseline, but any improvement must be demonstrated experimentally.

A successful software test does not establish predictive accuracy. Model performance must be reported using the completed dataset and held-out results.

## Constrained Design Optimization

The next stage will use SciPy to formulate a clearly defined engineering design problem.

The objective and constraints must be compatible with the actual parameterization and physical model. Potential criteria include reducing total rotational inertia or another justified design cost while maintaining specified natural-frequency and dynamic-response requirements.

The optimized candidate must be simulated using the original physics model. Surrogate predictions alone are insufficient to establish feasibility.

Optimization results should report:

- Baseline and optimized design variables.
- Objective value and constraint satisfaction.
- Predicted versus physics-calculated responses.
- Computational cost and any relevant model errors.

## Reinforcement Learning: PPO versus PID

The planned control stage will use Gymnasium and Stable-Baselines3 to compare PPO with a conventional PID baseline on a consistent rotational-system control task.

Both controllers must use the same plant model, command profile, actuator limits, and evaluation conditions.

Planned evaluation metrics include:

- Position-tracking RMSE.
- Settling time and overshoot.
- Peak control torque and control effort.
- Performance under selected parameter perturbations.

PPO is not assumed to outperform PID. The objective is a fair quantitative comparison, including cases in which the conventional controller performs better.

The control environment and its final actuation and observation definitions must be documented before results are presented.

## Implementation Status

### Implemented components

- Rotational-torsional dataset generation through `scripts/generate_dataset.py`.
- Neural-network model definition in `src/surrogate.py`.
- Surrogate unit tests in `tests/test_surrogate.py`.
- Training and evaluation workflow in `scripts/train_surrogate.py`.
- Dataset validation and pilot CSV generation.

The 20-design smoke test completed successfully and reported no missing values in the generated CSV. The full 5,000-design dataset was shown in progress in the latest console output; verify its completion before training.

### Remaining work

- [ ] Complete and validate the full dataset.
- [ ] Run the full test suite against the current code.
- [ ] Train the baseline neural network and inspect held-out metrics.
- [ ] Compare two-, three-, and four-hidden-layer architectures if useful.
- [ ] Establish surrogate inference speed versus numerical simulation.
- [ ] Implement and test constrained optimization.
- [ ] Validate the optimized candidate using the original physics model.
- [ ] Implement and test the control environment and PID baseline.
- [ ] Train PPO and compare both controllers under identical conditions.
- [ ] Produce final figures, numerical results, and limitations.

## Reproducibility and Data Management

- Use fixed seeds for dataset generation, data splitting, and training.
- Record model parameters, units, solver settings, and evaluation conditions.
- Keep the physics implementation independent of the surrogate.
- Retain a held-out test set for final evaluation.
- Document how datasets and model checkpoints can be regenerated.
- Avoid committing large generated datasets and checkpoints unless there is a clear portfolio benefit.
- Distinguish numerical verification from validation against physical hardware.

## Limitations

The model is a simplified rotational-torsional system, not a complete robotic manipulator. The dataset is synthetic, and the surrogate's accuracy depends on its coverage of the parameter space. Optimization results depend on the selected objective and constraints. Reinforcement-learning results depend on the environment, reward formulation, training budget, and evaluation protocol.

No claims about full-robot performance or real-hardware validation should be made without additional evidence.

## Technology Stack

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- PyTorch
- scikit-learn
- Gymnasium
- Stable-Baselines3
- pytest

The intended outcome is a reproducible portfolio project demonstrating numerical mechanics, scientific machine learning, constrained design optimization, and data-driven control in a robotic-joint application.