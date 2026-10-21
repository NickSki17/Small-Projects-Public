# Numerical Optimization and Vibration Analysis

This project brings together two compact computational-mechanics studies: first-order and second-order optimization on benchmark functions, and modal/time-history analysis of an idealized mass-spring chain. The work focuses on algorithm behavior, numerical failure modes, and reproducible analysis—not on presenting every solver result as physically meaningful.

## Project layout

```text
Notebooks/optimization-project/
├── optimization-project-code/
│   ├── optimization.py
│   ├── vibration_analysis.py
│   └── README.md
├── figures/                  # generated plots
└── results/                  # generated optimization summary
```

The original exploratory notebook is retained separately at `Notebooks/optimization_and_vibration_analysis(original).ipynb`.

## Run the analyses

Install the packages required by the scripts in your Python environment, then run from the code directory:

```bash
python optimization.py
python vibration_analysis.py
```

Both scripts resolve generated output paths relative to their source files. Optimization plots and the summary CSV are written under `Notebooks/optimization-project/figures/` and `results/`; vibration trajectory and spectrum plots are written under `figures/`.

## Optimization benchmarks

`optimization.py` compares fixed-step Gradient Descent, a finite-difference Newton method, and SciPy BFGS on five functions: a sphere, a convex quadratic, Rosenbrock, a quartic with a degenerate minimum, and a positive Gaussian bump. Results include convergence diagnostics, plots, and a structured CSV summary.

The examples are selected to expose different behavior, including divergence or failure to reach tolerance. In particular, the Gaussian-bump objective approaches zero far from its finite maximum; Newton's method can take an enormous step and encounter numerical underflow, yielding an apparent convergence far from a meaningful finite minimizer. This is intentionally reported as a numerical limitation of the basic method, not presented as a valid solution.

## Vibration model

`vibration_analysis.py` models a uniform, undamped, fixed-fixed chain of point masses connected by springs. It computes natural frequencies from the state-space eigenvalues and simulates free response using `scipy.integrate.solve_ivp`. An FFT provides a frequency-domain cross-check of the simulated response.

Assumptions include uniform mass and stiffness, no damping or external forcing, fixed end conditions, and seeded random initial conditions. The FFT resolution is finite, so nearby modes may not appear as distinct peaks. The FFT is a cross-check; modal frequencies are obtained from the model eigenproblem.

Named constants in `vibration_analysis.py` include `DEFAULT_K`, `DEFAULT_M`, `DEFAULT_SEED`, `BASELINE_N_SIM`, `BASELINE_T`, and `BASELINE_DT`. The fixed seed and deterministic time grid make the baseline simulation repeatable under the same numerical environment.

## Interpretation and limitations

Gradient Descent uses a fixed step size and can be slow or diverge. Newton's method is not globally convergent and can be sensitive to the local curvature and finite-difference Hessian. BFGS uses SciPy's stopping criteria, which are not identical to the custom methods' gradient-norm criteria. Iteration counts therefore should not be interpreted as directly comparable measures of efficiency.

The vibration study is an idealized computational example, not a general structural-dynamics solver. Numerical outputs depend on the stated assumptions, discretization, solver tolerances, and finite observation horizon.
