
"""Evaluate selected-surrogate errors on the held-out validation set.

Run from the repository root:
    python -m scripts.diagnose_validation_region

This script does not train a model, alter the dataset, or evaluate the test set.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_surrogates import predict_mlp
from scripts.train_surrogate import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    TARGET_UNITS,
    load_dataset,
    split_dataset,
    make_arrays,
)
from src.surrogate import SurrogateMLP


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "full_dataset.csv"
MODEL_DIR = ROOT / "artifacts" / "surrogate_benchmark"
OUTPUT_DIR = ROOT / "artifacts" / "validation_region_diagnostic"

LOW_INERTIA_FRACTION = 0.10
DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dangerous_error_definition(target: str) -> str:
    """Return dangerous signed-error direction for baseline-relative limits."""
    if target.startswith("omega"):
        return "frequency_overprediction"
    return "maximum_response_underprediction"


def load_selected_model(model_dir: Path = MODEL_DIR) -> tuple[SurrogateMLP, dict]:
    metadata_path = model_dir / "selected_model_metadata.json"
    weights_path = model_dir / "selected_model_state.pt"

    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "Selected PyTorch model artifacts were not found. "
            "This diagnostic expects the currently selected "
            "MLP_very_wide_deep model."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_file", weights_path.name) != weights_path.name:
        raise ValueError("Selected-model metadata references a different checkpoint file.")
    expected_checkpoint_hash = metadata.get("checkpoint_sha256")
    if expected_checkpoint_hash and expected_checkpoint_hash != sha256_file(weights_path):
        raise ValueError("Selected-model checkpoint fingerprint does not match metadata.")

    if metadata["model"] != "MLP_very_wide_deep":
        raise ValueError(
            f"Expected MLP_very_wide_deep, found {metadata['model']!r}."
        )

    if metadata["model_type"] != "pytorch":
        raise ValueError(
            f"Expected a PyTorch model, found {metadata['model_type']!r}."
        )

    if metadata["input_columns"] != INPUT_COLUMNS:
        raise ValueError("Selected model input columns do not match.")

    if metadata["target_columns"] != TARGET_COLUMNS:
        raise ValueError("Selected model target columns do not match.")

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


def summarize_errors(
    records: list[dict],
    actual: np.ndarray,
    predicted: np.ndarray,
    region_name: str,
) -> None:
    """Append per-target signed and absolute error statistics."""
    errors = predicted - actual

    for index, target in enumerate(TARGET_COLUMNS):
        signed = errors[:, index]
        absolute = np.abs(signed)

        records.append(
            {
                "region": region_name,
                "target": target,
                "units": TARGET_UNITS[target],
                "sample_count": int(len(actual)),
                "MAE": float(absolute.mean()),
                "RMSE": float(np.sqrt(np.mean(signed**2))),
                "P90_absolute_error": float(
                    np.percentile(absolute, 90)
                ),
                "P95_absolute_error": float(
                    np.percentile(absolute, 95)
                ),
                "P99_absolute_error": float(
                    np.percentile(absolute, 99)
                ),
                "max_absolute_error": float(absolute.max()),
                "mean_signed_error_pred_minus_actual": float(
                    signed.mean()
                ),
                "max_overprediction": float(signed.max()),
                "max_underprediction": float(signed.min()),
                "underprediction_count": int((signed < 0).sum()),
                "overprediction_count": int((signed > 0).sum()),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate empirical constraint safeguards from validation data only.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    model_dir = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = OUTPUT_DIR / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S.%fZ")
    elif not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    dataset = load_dataset(data_path)
    model, metadata = load_selected_model(model_dir)
    if metadata.get("dataset_sha256") and metadata["dataset_sha256"] != sha256_file(data_path):
        raise ValueError("Selected model was trained/selected using a different dataset fingerprint.")

    seed = int(metadata["seed"])
    _, validation_df, _ = split_dataset(dataset, seed)

    x_validation, y_validation = make_arrays(validation_df)

    predictions = predict_mlp(model, x_validation)

    if predictions.shape != y_validation.shape:
        raise RuntimeError(
            f"Prediction shape {predictions.shape} does not match "
            f"validation target shape {y_validation.shape}."
        )

    # Preserve row identity and input design values for auditing.
    row_data = validation_df.reset_index(drop=True).copy()

    for index, target in enumerate(TARGET_COLUMNS):
        row_data[f"{target}_predicted"] = predictions[:, index]
        row_data[f"{target}_error_pred_minus_actual"] = (
            predictions[:, index] - y_validation[:, index]
        )
        row_data[f"{target}_absolute_error"] = np.abs(
            predictions[:, index] - y_validation[:, index]
        )

    # Define the low-inertia subset from validation designs only.
    # Inertia objective is J1 + J2 + J3.
    row_data["total_inertia"] = row_data[["J1", "J2", "J3"]].sum(axis=1)

    cutoff = float(
        row_data["total_inertia"].quantile(LOW_INERTIA_FRACTION)
    )
    low_mask = row_data["total_inertia"] <= cutoff

    if int(low_mask.sum()) < 20:
        raise RuntimeError(
            "Too few low-inertia validation designs for a useful "
            f"regional diagnostic: {int(low_mask.sum())}."
        )

    error_records: list[dict] = []

    summarize_errors(
        error_records,
        y_validation,
        predictions,
        "all_validation",
    )

    summarize_errors(
        error_records,
        y_validation[low_mask.to_numpy()],
        predictions[low_mask.to_numpy()],
        "lowest_inertia_validation_10pct",
    )

    error_summary = pd.DataFrame(error_records)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Positive error means prediction - actual is positive. With the
    # candidate compared to an independent physical baseline, overprediction
    # can hide an insufficient frequency and underprediction can hide an
    # excessive response.
    directional_records = []

    for target in TARGET_COLUMNS:
        errors = row_data[f"{target}_error_pred_minus_actual"].to_numpy()

        dangerous_label = dangerous_error_definition(target)
        dangerous = errors > 0 if target.startswith("omega") else errors < 0

        for region, mask in (
            ("all_validation", np.ones(len(row_data), dtype=bool)),
            (
                "lowest_inertia_validation_10pct",
                low_mask.to_numpy(),
            ),
        ):
            region_errors = errors[mask]
            region_dangerous = dangerous[mask]

            directional_records.append(
                {
                    "region": region,
                    "target": target,
                    "units": TARGET_UNITS[target],
                    "dangerous_error_direction": dangerous_label,
                    "sample_count": int(mask.sum()),
                    "dangerous_error_count": int(region_dangerous.sum()),
                    "dangerous_error_fraction": float(
                        region_dangerous.mean()
                    ),
                    "worst_dangerous_error_magnitude": (
                        float(np.max(np.abs(region_errors[region_dangerous])))
                        if region_dangerous.any()
                        else 0.0
                    ),
                }
            )

    error_summary.to_csv(
        output_dir / "validation_error_summary.csv",
        index=False,
    )

    pd.DataFrame(directional_records).to_csv(
        output_dir / "validation_constraint_direction.csv",
        index=False,
    )

    row_data.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    metadata_out = {
        "model": metadata["model"],
        "model_type": metadata["model_type"],
        "model_file": metadata["model_file"],
        "seed": seed,
        "dataset_rows": int(len(dataset)),
        "validation_rows": int(len(validation_df)),
        "low_inertia_fraction": LOW_INERTIA_FRACTION,
        "low_inertia_cutoff": cutoff,
        "low_inertia_validation_rows": int(low_mask.sum()),
        "inertia_definition": "J1 + J2 + J3",
        "error_sign_convention": "prediction - actual",
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_units": TARGET_UNITS,
        "test_partition_evaluated": False,
        "dataset_sha256": sha256_file(data_path),
        "checkpoint_sha256": sha256_file(model_dir / metadata["model_file"]),
        "model_metadata_sha256": sha256_file(model_dir / "selected_model_metadata.json"),
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "validation_design_ids_sha256": hashlib.sha256(
            ",".join(map(str, validation_df["design_id"].tolist())).encode("utf-8")
        ).hexdigest(),
        "split": {"method": "two-stage sklearn train_test_split", "seed": seed,
                  "train_fraction": 0.70, "validation_fraction": 0.15, "test_fraction": 0.15},
        "source_revision": None,
        "source_tree_dirty": None,
        "note": (
            "These are empirical validation errors, not formal "
            "guarantees for unseen designs."
        ),
    }

    try:
        metadata_out["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        metadata_out["source_tree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        pass

    for filename, key in (
        ("validation_error_summary.csv", "error_summary_sha256"),
        ("validation_constraint_direction.csv", "constraint_direction_sha256"),
        ("validation_predictions.csv", "validation_predictions_sha256"),
    ):
        metadata_out[key] = sha256_file(output_dir / filename)

    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata_out, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION-ONLY SURROGATE ERROR DIAGNOSTIC")
    print("=" * 65)
    print(f"Selected model: {metadata['model']}")
    print(f"Dataset rows: {len(dataset)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Low-inertia cutoff: {cutoff:.10f}")
    print(f"Low-inertia validation rows: {int(low_mask.sum())}")
    print("Test partition evaluated: False")

    print("\nERROR SUMMARY")
    print(
        error_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print("\nCONSTRAINT-DIRECTION SUMMARY")
    print(
        pd.DataFrame(directional_records).to_string(
            index=False,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
