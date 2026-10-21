# Reduced-Order Rotational-Torsional Design Optimization

This project develops and tests a physics-based design-optimization workflow for a **reduced-order 3-DOF rotational-torsional chain**. It is not a full rigid-body robot manipulator, and its simulated results are not validation against physical hardware.

The implemented non-RL pipeline connects mechanics simulation, synthetic data generation, neural-network surrogate selection, validation-only error calibration, constrained design optimization, and independent physics verification.

## Engineering model

The design vector is `[J1, J2, J3, k_theta1, k_theta2, k_theta3]`. The model parameters, physical bounds, damping assumptions, and simulation settings are defined in `src/dataset_config.py`; system matrices and dynamics are in `src/dynamics.py`; modal analysis and time integration are implemented in `src/modal.py` and `src/simulation.py`. `src/metrics.py` computes the three modal frequencies (`omega1`–`omega3`, rad/s), sampled-history angular-response maxima (`theta2_max`, `theta3_max`, rad), maximum angular acceleration (`alpha_max`, rad/s²), and RMS angular response (`theta_rms`, rad).

The objective is total inertia, `J1 + J2 + J3` (kg·m²). Stiffness is bounded but has no direct cost in the current formulation. That is a modeling choice, not a claim that stiffness has no engineering or economic cost; a different objective requires an explicit engineering requirement. Time-history maxima are sampled-grid values, not guaranteed continuous-time extrema.

## Current evidence and status

A recorded local optimization run is available at `artifacts/surrogate_optimization/run_20261021T052354.523939Z`. It selected a candidate with total inertia approximately `0.0323192 kg·m²`, compared with a `0.09 kg·m²` baseline—about a 64.09% reduction. The recorded independent-physics verification reports positive margins for all seven constraints, and the candidate is within the configured design bounds. This is evidence for that saved run only; it does not establish global optimality or physical-hardware performance.

The associated verification command is:

```powershell
python -m scripts.verify_optimization_run --run-dir artifacts/surrogate_optimization/run_20261021T052354.523939Z
```

The full test suite completed with **176 passed** after the SLSQP finite-difference regression was added. A clean-checkout end-to-end reproduction has **not** been completed. The dataset is ignored by Git, and the prior isolated reproduction attempt was interrupted during dataset generation. Treat the saved result as a local verified baseline, not a clean-checkout reproduction.

An SLSQP diagnosis found that SciPy's default finite-difference step is too small for reliable differences through this float32 surrogate. The optimizer now uses an absolute step of `1e-5` in unit-scaled design coordinates. In a controlled run from the verified candidate, SLSQP then terminated successfully after 96 iterations and 1,330 function evaluations; its returned design remained inside the bounds, passed both the calibrated surrogate screen and independent physics checks, and changed inertia by only about `−9.85e−9 kg·m²` relative to the starting candidate. This controlled single-start result is not a replacement for the saved COBYQA baseline; a full multi-start comparison with the updated SLSQP step was not completed.

The selected model in the existing benchmark is `MLP_very_wide_deep`, trained with seed 42 and architecture `[512, 512, 512, 256]`. The local dataset, selected model, calibration, and optimization run must remain identity-matched; do not substitute a newer diagnostic directory based on its timestamp alone.

## Reproducing the workflow

Run commands from this project directory so Python can resolve the `src` and `scripts` modules. The dataset and generated run outputs are local artifacts; the dataset is not committed. A fresh reproduction requires regenerating data and then training/selecting a model and calibrating that exact selected model against that exact dataset. Use unique output locations to preserve existing evidence.

The following PowerShell sequence illustrates the intended order. Choose a fresh run ID and ensure output paths do not already exist:

```powershell
$RunId = Get-Date -Format 'yyyyMMddTHHmmss'
$Dataset = "data/reproduction/full_dataset_$RunId.csv"
$BaseModel = "artifacts/reproduction/$RunId/base_model"
$SelectedModel = "artifacts/reproduction/$RunId/selected_model"
$Calibration = "artifacts/validation_region_diagnostic/run_$RunId"

python -m scripts.generate_dataset --samples 5000 --output $Dataset
python -m scripts.train_surrogate --data $Dataset --output-dir $BaseModel
python -m scripts.benchmark_surrogates --data $Dataset --baseline-model-dir $BaseModel --output-dir $SelectedModel
python -m scripts.diagnose_validation_region --data $Dataset --model-dir $SelectedModel --output-dir $Calibration
python -m src.optimization_adapter --dataset $Dataset --model-dir $SelectedModel --calibration-dir $Calibration --n-starts 10 --max-iterations 10000
```

Use the new run directory printed by optimization to verify the result:

```powershell
python -m scripts.verify_optimization_run --run-dir artifacts/surrogate_optimization/run_<timestamp>
python -m pytest -q
python -m compileall -q src scripts tests
```

This sequence is a documented workflow, not a claim that the full sequence has already been reproduced from a clean checkout. The benchmark script refuses a non-empty default output directory unless explicitly directed otherwise; prefer a new output directory rather than replacing prior evidence. The generator, model fitting, and benchmark may take substantial time. Do not regenerate or retrain merely to verify an existing local run.

## Calibration and acceptance

For residual `e = prediction − actual`, dangerous frequency error is overprediction, while dangerous response error is underprediction. The directional allowance is therefore `max(e, 0)` for each frequency and `max(−e, 0)` for each response. The surrogate screen uses the baseline-relative forms:

```text
frequency: (prediction − allowance) / physical_baseline − 1 ≥ −1e−5
response:  1 − (prediction + allowance) / physical_baseline ≥ −1e−5
```

The baseline is independently evaluated using the physics model at the configured baseline design. The low-inertia allowance applies when candidate total inertia is at or below the validation-set 10th-percentile cutoff; the global validation allowance applies otherwise. In the recorded split there were 750 validation rows and 75 in the low-inertia subset. These observed directional maxima are empirical safeguards, not confidence bounds or guarantees about unseen designs. The test partition must not be used to estimate or tune allowances.

The optimization workflow keeps solver termination, design-bound checks, calibrated surrogate screening, independent physical feasibility, and objective improvement distinct. Solver success does not establish physical feasibility; solver failure alone does not disqualify a candidate that satisfies the documented acceptance checks. If no feasible candidate improves on the feasible baseline, the baseline is retained. Local solver termination is not proof of a global optimum.

## Tests and artifacts

Run tests from the project directory with `python -m pytest -q`. New optimization runs are written to timestamped directories under `artifacts/surrogate_optimization/`; the directory contains a JSON summary, design comparison, independent physics verification, and run manifest. The verifier checks consistency of the saved run and its recorded inputs. Keep each dataset, selected checkpoint and metadata, calibration directory, and optimization run together when preserving evidence.

`data/full_dataset.csv` is ignored by Git. Run manifests can record file hashes and available configuration/runtime details, but hashes do not make unavailable inputs reproducible. The dependency file constrains supported package versions; it is not a platform-specific lockfile. Exact bitwise training reproducibility across hardware and library builds is not claimed.

## Reinforcement-learning status

Reinforcement learning is not implemented. `src/environment.py`, `src/controllers.py`, and `scripts/train_rl.py` are placeholders; there is no executable state/action interface, reward definition, training/evaluation protocol, checkpoint workflow, or RL-versus-controller comparison. Defining the control objective and safety/evaluation requirements is a prerequisite to that work.
