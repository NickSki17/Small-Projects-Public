
"""Benchmark surrogate models for the 3-DOF torsional system.

Run from the project root:
    python -m scripts.benchmark_surrogates

The existing trained MLP is evaluated as a reference, not retrained.
Candidate models are selected using validation performance.
The test set is evaluated only after candidate selection.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import random
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.train_surrogate import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    TARGET_UNITS,
    load_dataset,
    split_dataset,
    make_arrays,
    fit_normalization,
    train_model,
    predict,
)
from src.surrogate import SurrogateMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts/surrogate"
DATA_PATH = PROJECT_ROOT / "data/full_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/surrogate_benchmark"
PLOT_DIR = OUTPUT_DIR / "plots"
DEVICE = torch.device("cpu")
DEFAULT_SEED = 42


def load_existing_mlp():
    """Load the previously trained baseline MLP and its metadata."""
    metadata_path = ARTIFACT_DIR / "metadata.json"
    weights_path = ARTIFACT_DIR / "model_state.pt"

    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "Baseline MLP artifacts are missing. Run "
            "'python -m scripts.train_surrogate' first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata["input_columns"] != INPUT_COLUMNS:
        raise ValueError("Baseline input columns do not match.")
    if metadata["target_columns"] != TARGET_COLUMNS:
        raise ValueError("Baseline target columns do not match.")

    model = SurrogateMLP(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_dims=tuple(metadata["hidden_dims"]),
    ).to(DEVICE)

    state = torch.load(
        weights_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def predict_mlp(
    model: SurrogateMLP,
    inputs: np.ndarray,
) -> np.ndarray:
    """Predict physical outputs using an MLP."""
    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        predictions = model(tensor).cpu().numpy()

    if not np.isfinite(predictions).all():
        raise RuntimeError("MLP produced non-finite predictions.")

    return predictions


def normalized_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Calculate per-target RMSE normalized by actual target ranges."""
    rmse = np.sqrt(np.mean((predicted - actual) ** 2, axis=0))
    ranges = np.ptp(actual, axis=0)

    if np.any(ranges <= 0):
        raise ValueError("Every target must have a nonzero range.")

    return rmse / ranges


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    partition: str,
) -> list[dict]:
    """Calculate regression metrics for each target."""
    records = []

    for i, target in enumerate(TARGET_COLUMNS):
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        target_range = float(np.ptp(y_true))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        records.append({
            "model": model_name,
            "partition": partition,
            "target": target,
            "units": TARGET_UNITS[target],
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": rmse,
            "R2": float(r2_score(y_true, y_pred)),
            "max_absolute_error": float(abs_errors.max()),
            "P95_absolute_error": float(np.percentile(abs_errors, 95)),
            "mean_signed_error": float(errors.mean()),
            "NRMSE_range_percent": (
                100.0 * rmse / target_range
                if target_range > 0
                else float("nan")
            ),
        })

    return records


def save_selected_model(
    selected_name: str,
    candidates: dict,
    alternative_mlps: dict,
    existing_mlp: SurrogateMLP,
    existing_metadata: dict,
    mlp_architectures: dict,
    seed: int,
    validation_score: float,
    dataset: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict:
    """Save the selected model and metadata for later inference."""
    metadata = {
        "model": selected_name,
        "seed": seed,
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_units": TARGET_UNITS,
        "input_dim": len(INPUT_COLUMNS),
        "output_dim": len(TARGET_COLUMNS),
        "selection_metric": "validation_mean_NRMSE",
        "validation_mean_NRMSE": validation_score,
        "dataset_sha256": _sha256_file(DATA_PATH),
        "dataset_rows": int(len(dataset)),
        "validation_rows": int(len(validation_df)),
        "validation_design_ids_sha256": hashlib.sha256(
            ",".join(map(str, validation_df["design_id"].tolist())).encode("utf-8")
        ).hexdigest(),
        "source_revision": _git_revision(),
        "python_version": __import__("sys").version,
    }

    if selected_name in candidates:
        model_path = OUTPUT_DIR / "selected_model.joblib"
        joblib.dump(candidates[selected_name], model_path)
        metadata["model_type"] = "sklearn"
        metadata["model_file"] = model_path.name
        metadata["estimator_class"] = type(
            candidates[selected_name]
        ).__name__

    else:
        if selected_name == "MLP_existing":
            model = existing_mlp
            hidden_dims = list(existing_metadata["hidden_dims"])
        else:
            model = alternative_mlps[selected_name]
            hidden_dims = list(mlp_architectures[selected_name])

        model_path = OUTPUT_DIR / "selected_model_state.pt"
        torch.save(model.state_dict(), model_path)

        metadata["model_type"] = "pytorch"
        metadata["model_file"] = model_path.name
        metadata["hidden_dims"] = hidden_dims

    metadata_path = OUTPUT_DIR / "selected_model_metadata.json"
    metadata["checkpoint_sha256"] = _sha256_file(model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def generate_benchmark_plots(
    validation_summary: pd.DataFrame,
    validation_predictions: dict[str, np.ndarray],
    y_validation: np.ndarray,
    selected_name: str,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """Save validation comparisons, test diagnostics, and MLP histories."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, filename: str) -> None:
        fig.savefig(
            PLOT_DIR / f"{filename}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            PLOT_DIR / f"{filename}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # 1. Validation model comparison.
    comparison = validation_summary.sort_values(
        "validation_mean_NRMSE"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        comparison["model"],
        100 * comparison["validation_mean_NRMSE"],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized RMSE (%)")
    ax.set_title("Surrogate Model Comparison — Validation Set")
    ax.grid(axis="x", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f", padding=4)
    fig.tight_layout()
    save_plot(fig, "validation_model_comparison")

    # 2. Validation error heatmap.
    model_names = list(validation_predictions)
    errors = np.array([
        100 * normalized_rmse(
            y_validation, validation_predictions[name]
        )
        for name in model_names
    ])

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.6 * len(model_names)))
    )
    image = ax.imshow(errors, aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(TARGET_COLUMNS)),
        [
            f"{target}\n({TARGET_UNITS[target]})"
            for target in TARGET_COLUMNS
        ],
    )
    ax.set_yticks(range(len(model_names)), model_names)
    ax.set_title("Validation NRMSE by Target")
    fig.colorbar(image, ax=ax, label="Normalized RMSE (%)")
    fig.tight_layout()
    save_plot(fig, "validation_error_heatmap")

    # 3. Test parity plots for the selected model only.
    fig, axes = plt.subplots(
        1,
        len(TARGET_COLUMNS),
        figsize=(5 * len(TARGET_COLUMNS), 4),
        squeeze=False,
    )

    for i, target in enumerate(TARGET_COLUMNS):
        ax = axes[0, i]
        actual = y_test[:, i]
        predicted = y_test_pred[:, i]

        ax.scatter(actual, predicted, alpha=0.55, s=18)
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        ax.plot(
            [lower, upper],
            [lower, upper],
            "k--",
            linewidth=1.5,
            label="Ideal prediction",
        )
        ax.set_xlabel(f"Actual ({TARGET_UNITS[target]})")
        ax.set_ylabel(f"Predicted ({TARGET_UNITS[target]})")
        ax.set_title(target)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Test Predictions — {selected_name}")
    fig.tight_layout()
    save_plot(fig, "test_parity_plots")

    # 4. Test residual plots for the selected model only.
    fig, axes = plt.subplots(
        1,
        len(TARGET_COLUMNS),
        figsize=(5 * len(TARGET_COLUMNS), 4),
        squeeze=False,
    )

    for i, target in enumerate(TARGET_COLUMNS):
        ax = axes[0, i]
        actual = y_test[:, i]
        residual = y_test_pred[:, i] - actual

        ax.scatter(actual, residual, alpha=0.55, s=18)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"Actual ({TARGET_UNITS[target]})")
        ax.set_ylabel(f"Prediction error ({TARGET_UNITS[target]})")
        ax.set_title(target)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Test Residuals — {selected_name}")
    fig.tight_layout()
    save_plot(fig, "test_residual_plots")

    # 5. Training curves for the five alternative MLPs.
    history_files = sorted(
        OUTPUT_DIR.glob("*_training_history.csv")
    )

    for path in history_files:
        history = pd.read_csv(path)

        train_col = next(
            (
                column
                for column in ("training_loss", "train_loss")
                if column in history.columns
            ),
            None,
        )
        val_col = next(
            (
                column
                for column in ("validation_loss", "val_loss")
                if column in history.columns
            ),
            None,
        )

        if train_col is None or val_col is None:
            print(f"Skipping unrecognized history: {path.name}")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["epoch"], history[train_col], label="Training")
        ax.plot(history["epoch"], history[val_col], label="Validation")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_title(path.stem.replace("_training_history", ""))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_plot(fig, path.stem)

    print(f"\nPlots saved to: {PLOT_DIR.resolve()}")


def main() -> None:
    global ARTIFACT_DIR, DATA_PATH, OUTPUT_DIR, PLOT_DIR
    parser = argparse.ArgumentParser(description="Benchmark surrogates using a validation-only selection split.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--baseline-model-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in the selected output directory.")
    args = parser.parse_args()
    DATA_PATH = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    ARTIFACT_DIR = args.baseline_model_dir if args.baseline_model_dir.is_absolute() else PROJECT_ROOT / args.baseline_model_dir
    OUTPUT_DIR = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    PLOT_DIR = OUTPUT_DIR / "plots"
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Benchmark output directory is not empty: {OUTPUT_DIR}. Choose a new directory or pass --overwrite.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(DATA_PATH)
    existing_mlp, existing_metadata = load_existing_mlp()
    if existing_metadata.get("dataset_sha256") and existing_metadata["dataset_sha256"] != _sha256_file(DATA_PATH):
        raise ValueError("Baseline model was trained with a different dataset fingerprint.")
    seed = int(existing_metadata["seed"])

    train_df, validation_df, test_df = split_dataset(data, seed)

    x_train, y_train = make_arrays(train_df)
    x_validation, y_validation = make_arrays(validation_df)
    x_test, y_test = make_arrays(test_df)

    if not (
        len(train_df) + len(validation_df) + len(test_df) == len(data)
    ):
        raise RuntimeError("Dataset split sizes do not add up.")

    print(
        f"Dataset: {len(data)} designs | "
        f"Train: {len(train_df)} | "
        f"Validation: {len(validation_df)} | "
        f"Test: {len(test_df)}"
    )

    candidates = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    mlp_architectures = {
        "MLP_small": (64, 64),
        "MLP_wide_deep": (256, 256, 128),
        "MLP_deeper": (256, 256, 256, 128),
        "MLP_wider": (384, 384, 256),
        "MLP_very_wide_deep": (512, 512, 512, 256),
    }

    alternative_mlps = {}
    validation_predictions = {}
    fit_times = {}

    # Existing MLP: evaluate on validation only.
    if int(existing_metadata["seed"]) != seed:
        raise ValueError("Unexpected baseline seed mismatch.")

    validation_predictions["MLP_existing"] = predict_mlp(
        existing_mlp, x_validation
    )
    fit_times["MLP_existing"] = 0.0

    # Fit tree candidates on training data only.
    for name, estimator in candidates.items():
        print(f"\nTraining {name}...")
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_times[name] = time.perf_counter() - start
        validation_predictions[name] = estimator.predict(x_validation)

    # Train alternative MLP architectures.
    for name, hidden_dims in mlp_architectures.items():
        print(f"\nTraining {name}: {hidden_dims}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = SurrogateMLP(
            input_dim=len(INPUT_COLUMNS),
            output_dim=len(TARGET_COLUMNS),
            hidden_dims=hidden_dims,
        ).to(DEVICE)

        fit_normalization(model, x_train, y_train)

        start = time.perf_counter()
        model, history, best_epoch = train_model(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
            epochs=2000,
            patience=100,
            batch_size=128,
            learning_rate=1e-3,
        )

        alternative_mlps[name] = model
        fit_times[name] = time.perf_counter() - start
        validation_predictions[name] = predict(model, x_validation)

        history.to_csv(
            OUTPUT_DIR / f"{name.lower()}_training_history.csv",
            index=False,
        )

        (OUTPUT_DIR / f"{name.lower()}_metadata.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "hidden_dims": list(hidden_dims),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "training_seconds": fit_times[name],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Select by validation performance only.
    results = []

    for name, prediction in validation_predictions.items():
        per_target_nrmse = normalized_rmse(y_validation, prediction)
        results.append({
            "model": name,
            "validation_mean_NRMSE": float(per_target_nrmse.mean()),
            "validation_max_target_NRMSE": float(per_target_nrmse.max()),
            "training_seconds": fit_times[name],
        })

    validation_summary = pd.DataFrame(results).sort_values(
        "validation_mean_NRMSE"
    )

    validation_summary.to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    selected_name = str(validation_summary.iloc[0]["model"])
    validation_score = float(
        validation_summary.iloc[0]["validation_mean_NRMSE"]
    )

    print("\nVALIDATION MODEL COMPARISON")
    print(validation_summary.to_string(index=False))
    print(f"\nSelected by validation: {selected_name}")

    # Persist the selected artifact before final test evaluation.
    selected_metadata = save_selected_model(
        selected_name=selected_name,
        candidates=candidates,
        alternative_mlps=alternative_mlps,
        existing_mlp=existing_mlp,
        existing_metadata=existing_metadata,
        mlp_architectures=mlp_architectures,
        seed=seed,
        validation_score=validation_score,
        dataset=data,
        validation_df=validation_df,
    )

    # Evaluate the test partition only after model selection.
    if selected_metadata["model_type"] == "sklearn":
        y_test_pred = candidates[selected_name].predict(x_test)
    elif selected_name == "MLP_existing":
        y_test_pred = predict_mlp(existing_mlp, x_test)
    else:
        y_test_pred = predict(alternative_mlps[selected_name], x_test)

    if not np.isfinite(y_test_pred).all():
        raise RuntimeError("Selected model produced non-finite predictions.")

    test_records = calculate_metrics(
        y_test, y_test_pred, selected_name, "test"
    )

    pd.DataFrame(test_records).to_csv(
        OUTPUT_DIR / "selected_model_test_metrics.csv",
        index=False,
    )

    print("\nFINAL TEST METRICS — SELECTED MODEL ONLY")
    print(pd.DataFrame(test_records).to_string(index=False))
    print(f"\nBenchmark outputs: {OUTPUT_DIR.resolve()}")

    generate_benchmark_plots(
        validation_summary=validation_summary,
        validation_predictions=validation_predictions,
        y_validation=y_validation,
        selected_name=selected_name,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )


if __name__ == "__main__":
    main()
