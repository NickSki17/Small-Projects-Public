# Optimization Project: Benchmarks and Vibration Analysis

Two independent, previously-tangled scripts, separated and cleaned up:

```
optimization-project/
├── README.md
├── requirements.txt
├── src/
│   ├── optimization.py        # GD / Newton / BFGS on 5 benchmark functions
│   └── vibration_analysis.py  # n-mass spring chain: eigenvalues + FFT
├── figures/                   # generated on run (convergence, trajectory, spectrum plots)
└── results/                   # generated on run (optimization_summary.csv)
```

Run either module directly:

```bash
pip install numpy scipy matplotlib
python optimization.py
python vibration_analysis.py
```

## What changed from the original scripts

- **Split into two modules** by concern (optimization vs. vibration), each with
  a `main()` entry point. The original files interleaved both problems and
  had two duplicated, slightly-diverging copies of the same functions.
- **Fixed a return-value bug** in the vibration script:
  `return sol.t, sol.y if sol.success else (sol.t, sol.y)` evaluated (due to
  operator precedence) to always return `sol.y`'s normal value on success,
  but on *failure* returned a nested tuple as the second element instead of
  an array — a silent shape mismatch that would have surfaced downstream as
  a confusing crash or, worse, a silent bad plot.
- **Fixed a performance bug**: the ODE right-hand-side function was
  rebuilding the (constant) system matrix on every single call — and
  `solve_ivp` calls it many thousands of times per integration. The matrix
  is now built once per simulation and passed in.
- **Fixed a correctness fragility** in Newton's method: the finite-difference
  Hessian is not guaranteed symmetric, but the code inverted it directly,
  relying on the true Hessian's symmetry to make this "work out." It is now
  explicitly symmetrized before inversion.
- **Deterministic simulation**: initial conditions now use a seeded
  `numpy.random.default_rng` instead of unseeded `np.random.rand`, and the
  time grid is built with `np.linspace` (exact point count, both endpoints
  guaranteed) instead of `np.arange` plus a manual endpoint patch.
- **Structured results**: iteration histories are `IterationRecord`/
  `OptimizationResult` dataclasses instead of ad hoc dicts; a CSV summary
  and PNG convergence/trajectory/spectrum plots are saved automatically
  instead of only printed or shown interactively.

## Known limitations (stated, not hidden)

- **Optimization algorithms are simplistic by design**: gradient descent
  uses a fixed step length (no line search or momentum), so it converges
  slowly or not at all on curved problems (Rosenbrock, f3). This is
  reported as `converged=False`, not disguised.
- **BFGS's stopping test is not identical** to the normalized criterion
  used for GD/Newton (see `optimization.py` module docstring). Iteration
  counts across methods are informative, not perfectly comparable.
- **Newton's method is not globally convergent.** On `f5` (a Gaussian
  bump, where the gradient vanishes far from the optimum), a single unit
  Newton step overshoots to `x ≈ 2.7e7` and the loop reports
  `converged=True` because the (numerically zero) gradient there trivially
  satisfies the stopping test. This is real, reportable behavior of the
  method — not something to average away or omit from a report.
- **The vibration model assumes** uniform mass/stiffness, no damping, and
  free (unforced) vibration. Natural-frequency deduplication rounds to 8
  decimal places, which is a practical necessity but would also merge two
  genuinely distinct frequencies that happened to agree that closely
  (not expected at this system size, but worth stating explicitly).
- **FFT frequency resolution** for the `n = 20` baseline case is
  `1/T = 0.01 Hz`; the closely-spaced high-order modes (see the spectrum
  plot) blur together at that resolution and should not be read as
  exact single-frequency peaks.

## Reproducibility

All simulation parameters are named constants at the top of
`vibration_analysis.py` (`DEFAULT_K`, `DEFAULT_M`, `DEFAULT_SEED`,
`BASELINE_N_SIM`, `BASELINE_T`, `BASELINE_DT`) rather than scattered
literals, and the random seed is fixed, so the baseline results are
regenerated identically on every run.
