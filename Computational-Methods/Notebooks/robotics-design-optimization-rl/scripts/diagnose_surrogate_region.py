from pathlib import Path

import numpy as np
import pandas as pd

from src.surrogate_inference import load_surrogate

DATA_PATH = Path("data/full_dataset.csv")
MODEL_DIR = Path("artifacts/surrogate_benchmark")

INPUT_COLUMNS = [
    "J1", "J2", "J3",
    "k_theta1", "k_theta2", "k_theta3",
]

TARGET_COLUMNS = [
    "omega1", "omega2", "omega3",
    "theta2_max", "theta3_max",
    "alpha_max", "theta_rms",
]


def main():
    df = pd.read_csv(DATA_PATH)

    required = INPUT_COLUMNS + TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y_true = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all() or not np.isfinite(y_true).all():
        raise ValueError("Dataset contains non-finite values.")

    inference = load_surrogate(MODEL_DIR)
    y_pred = np.asarray(inference.predict_array(X), dtype=np.float64)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Prediction shape {y_pred.shape} does not match "
            f"target shape {y_true.shape}."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError("Surrogate produced non-finite predictions.")

    # Positive signed error means the surrogate overpredicts the target.
    errors = y_pred - y_true
    inertia = df[["J1", "J2", "J3"]].sum(axis=1).to_numpy()

    # Analyze the lowest-inertia 10% of the existing dataset.
    cutoff = float(np.quantile(inertia, 0.10))
    low_inertia = inertia <= cutoff

    print(f"Dataset rows: {len(df)}")
    print(f"Model directory: {MODEL_DIR.resolve()}")
    print(f"Lowest-10% inertia cutoff: {cutoff:.10f}")
    print(f"Rows in lowest-inertia subset: {low_inertia.sum()}")
    print(f"Minimum dataset inertia: {inertia.min():.10f}")
    print(f"Maximum dataset inertia: {inertia.max():.10f}")

    for label, mask in [
        ("ALL DATASET DESIGNS", np.ones(len(df), dtype=bool)),
        ("LOWEST-INERTIA 10%", low_inertia),
    ]:
        print(f"\n===== {label} =====")
        records = []

        for i, target in enumerate(TARGET_COLUMNS):
            e = errors[mask, i]
            abs_e = np.abs(e)

            records.append({
                "target": target,
                "MAE": np.mean(abs_e),
                "P95_abs_error": np.percentile(abs_e, 95),
                "max_abs_error": np.max(abs_e),
                "mean_signed_error": np.mean(e),
                "overprediction_count": np.sum(e > 0),
                "underprediction_count": np.sum(e < 0),
            })

        print(pd.DataFrame(records).to_string(
            index=False, float_format=lambda value: f"{value:.8g}"
        ))

    # Inspect the lowest-inertia rows with the largest omega1 errors.
    selected_indices = np.flatnonzero(low_inertia)
    ranked = selected_indices[
        np.argsort(np.abs(errors[low_inertia, 0]))[::-1]
    ][:10]

    columns = INPUT_COLUMNS + ["omega1"]
    examples = df.iloc[ranked][columns].copy()
    examples["inertia"] = inertia[ranked]
    examples["omega1_predicted"] = y_pred[ranked, 0]
    examples["omega1_signed_error"] = errors[ranked, 0]
    examples["omega1_absolute_error"] = np.abs(errors[ranked, 0])

    print("\n===== 10 LOW-INERTIA DESIGNS WITH LARGEST OMEGA1 ERRORS =====")
    print(examples.to_string(index=False))

if __name__ == "__main__":
    main()
