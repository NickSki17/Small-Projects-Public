# Optimization Project: Benchmarks and Vibration Analysis

Two independent computational mechanics scripts, separated and cleaned up from the original tangled implementation:

```text
optimization-project/
├── README.md
├── requirements.txt
├── src/
│   ├── optimization.py          # GD / Newton / BFGS on 5 benchmark functions
│   └── vibration_analysis.py    # n-mass spring chain: eigenvalues + FFT
├── figures/                     # generated convergence, trajectory, and spectrum plots
└── results/                     # generated optimization summary CSV
```

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run either analysis directly:

```bash
python src/optimization.py
python src/vibration_analysis.py
```

Generated figures are written to `figures/`, and the optimization summary is written to `results/optimization_summary.csv`.

## What Changed from the Original Scripts

* **Separated the two computational problems** into independent modules:
  `optimization.py` contains the optimization benchmarks, while
  `vibration_analysis.py` contains the mass-spring vibration analysis.

* **Removed duplicated implementations** that previously existed in the original scripts and established a single implementation for each method.

* **Fixed a return-value bug** in the vibration simulation. The original conditional return expression could produce inconsistent return types when integration failed. The simulation now consistently returns the solver time and state arrays.

* **Fixed an ODE performance issue** by constructing the constant system matrix once per simulation rather than rebuilding it on every right-hand-side evaluation.

* **Improved Newton's method robustness** by explicitly symmetrizing the finite-difference Hessian before solving the Newton system.

* **Improved reproducibility** by using a seeded `numpy.random.default_rng` for vibration initial conditions and `np.linspace` for deterministic simulation time grids.

* **Added structured optimization results** using `IterationRecord` and `OptimizationResult` dataclasses instead of ad hoc dictionaries.

* **Added automated reporting**, including convergence plots for each optimization benchmark, vibration trajectory and spectrum plots, and a CSV summary of optimization results.

* **Improved optimization convergence reporting** by using the gradient norm directly as the stopping metric for the custom Gradient Descent and Newton implementations. This avoids allowing a large objective magnitude to artificially make the convergence metric appear small.

* **Added divergence detection to Gradient Descent** so runaway iterates or non-finite values are reported as `converged=False` rather than being allowed to continue indefinitely.

## Optimization Benchmarks

Five benchmark functions are used to compare:

* Fixed-step Gradient Descent
* Newton's Method with a finite-difference Hessian
* SciPy BFGS

The benchmark set includes:

* `f1`: Sphere function
* `f2`: Convex quadratic with a cross term
* `f3`: Rosenbrock function
* `f4`: Quartic function with a degenerate minimum
* `f5`: Positive Gaussian bump

The benchmarks intentionally expose different algorithmic behaviors rather than being designed for every method to converge successfully.

### Observed Behavior

* **Sphere and quadratic functions:** all three methods converge to the expected minima.

* **Rosenbrock:** fixed-step Gradient Descent diverges from the selected starting point and is correctly reported as `converged=False`, while Newton's Method and BFGS converge to the minimum near `[1, 1]`.

* **Quartic benchmark:** fixed-step Gradient Descent does not reach the convergence tolerance within the iteration limit, while Newton's Method and BFGS converge much more effectively.

* **Gaussian bump:** the objective approaches zero away from the finite maximum. Since the problem is posed as an unconstrained minimization, Gradient Descent does not reach the stopping criterion within the iteration limit. Newton's Method can take a very large step into a region where the Gaussian underflows numerically to zero, producing an apparent `converged=True` result at approximately `x = 2.7e7`. This is a numerical failure mode of the basic Newton implementation, not a valid finite minimizer.

These behaviors are intentionally retained because they demonstrate practical limitations of the algorithms rather than hiding them through additional line-search, damping, regularization, or other convergence enhancements.

## Vibration Analysis

The vibration model represents an undamped fixed-fixed chain:

```text
wall -- k -- m1 -- k -- m2 -- ... -- mn -- k -- wall
```

with uniform mass `m` and stiffness `k`.

The equations of motion are written in state-space form and solved using `scipy.integrate.solve_ivp`.

Natural frequencies are obtained from the eigenvalues of the first-order state-space system. For an undamped conservative system, eigenvalues occur in conjugate pairs:

```text
λ = ±iω
```

The positive imaginary parts correspond to the system's natural angular frequencies in rad/s.

The vibration simulation also provides an FFT-based frequency-domain cross-check of the time-domain response.

### Vibration Assumptions

The baseline model assumes:

* uniform mass
* uniform spring stiffness
* no damping
* no external forcing
* fixed boundary conditions at both ends
* free vibration from randomly generated initial conditions

The random initial conditions are deterministic because a fixed random seed is used.

### FFT Resolution

For the baseline `n = 20` simulation, the frequency resolution is:

```text
Δf = 1 / T = 0.01 Hz
```

Therefore, closely spaced high-order modes may appear blended in the FFT spectrum. The FFT is used as a numerical cross-check rather than as the primary method for computing the natural frequencies.

## Known Limitations

* **Gradient Descent uses a fixed step length.** It does not use line search, momentum, or adaptive step-size control. This intentionally simple implementation can converge slowly or diverge on poorly conditioned or strongly curved problems.

* **BFGS uses SciPy's own stopping criteria.** Its convergence test is not identical to the gradient-norm criterion used by the custom Gradient Descent and Newton implementations, so iteration counts should be treated as informative rather than as perfectly equivalent measures of efficiency.

* **Newton's Method is not globally convergent.** The Gaussian-bump benchmark demonstrates how a small or numerically underflowed gradient can cause the basic implementation to report convergence far from a meaningful minimizer.

* **The vibration model is intentionally simplified.** It uses uniform properties, fixed-fixed boundaries, and undamped free vibration. It is intended as a computational baseline rather than a general structural dynamics solver.

* **FFT resolution is finite.** Closely spaced modes cannot necessarily be resolved as distinct peaks at the available simulation duration.

## Reproducibility

Simulation parameters are defined as named constants in `vibration_analysis.py`, including:

* `DEFAULT_K`
* `DEFAULT_M`
* `DEFAULT_SEED`
* `BASELINE_N_SIM`
* `BASELINE_T`
* `BASELINE_DT`

The fixed random seed makes the baseline vibration simulation reproducible across runs, while the optimization benchmarks use deterministic starting conditions.

## Purpose

This project serves as a verified computational baseline for numerical optimization and vibration analysis. It emphasizes transparent algorithm behavior, reproducibility, numerical diagnostics, and explicit reporting of failure modes.

It provides the foundation for the more advanced computational engineering work developed in subsequent projects.
