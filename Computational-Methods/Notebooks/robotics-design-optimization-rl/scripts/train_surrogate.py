
"""
Train and evaluate a neural-network surrogate for the rotational
torsional vibration system.

Run from the project root:
    python -m scripts.train_surrogate

Optional:
    python -m scripts.train_surrogate --epochs 2000 --patience 100

The test partition is held out until final evaluation.
Normalization statistics are fitted using training data only.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.surrogate import (
    DEFAULT_HIDDEN_DIMS,
    INPUT_DIM,
    OUTPUT_DIM,
    SurrogateMLP,
)


INPUT_COLUMNS = [
    "J1",
    "J2",
    "J3",
    "k_theta1",
    "k_theta2",
    "k_theta3",
]

TARGET_COLUMNS = [
    "omega1",
    "omega2",
    "omega3",
    "theta2_max",
    "theta3_max",
    "alpha_max",
    "theta_rms",
]

TARGET_UNITS = {
    "omega1": "rad/s",
    "omega2": "rad/s",
    "omega3": "rad/s",
    "theta2_max": "rad",
    "theta3_max": "rad",
    "alpha_max": "rad/s^2",
    "theta_rms": "rad",
}

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the rotational torsional neural-network surrogate."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/full_dataset.csv"),
        help="Path to the generated dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/surrogate"),
        help="Directory for saved model, metrics, history, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible model training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate all required design and target columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}\n"
            "Generate it first with: python -m scripts.generate_dataset"
        )

    data = pd.read_csv(path)

    required_columns = INPUT_COLUMNS + TARGET_COLUMNS
    missing = sorted(set(required_columns) - set(data.columns))

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    if len(data) < 30:
        raise ValueError(
            "At least 30 rows are required for train/validation/test splitting."
        )

    numeric_data = data[required_columns].to_numpy(dtype=float)

    if not np.isfinite(numeric_data).all():
        raise ValueError(
            "Required input and target columns contain NaN or infinite values."
        )

    if np.any(data[INPUT_COLUMNS].to_numpy(dtype=float) <= 0.0):
        raise ValueError("All inertia and stiffness inputs must be positive.")

    if np.any(data[TARGET_COLUMNS].to_numpy(dtype=float) < 0.0):
        raise ValueError("The selected frequency and response targets must be nonnegative.")

    if any(data[column].nunique() < 2 for column in TARGET_COLUMNS):
        raise ValueError(
            "At least one target is constant. Inspect the dataset and "
            "remove constant targets from TARGET_COLUMNS."
        )

    return data


def split_dataset(
    data: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible 70% / 15% / 15% partitions."""
    train_data, temporary_data = train_test_split(
        data,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def make_arrays(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract input and target arrays in the prescribed column order."""
    inputs = data[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    targets = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return inputs, targets


def fit_normalization(
    model: SurrogateMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalization parameters using training data only."""
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)

    # Prevent division by zero if a feature or target has negligible variance.
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_scale = np.where(y_scale > 1e-12, y_scale, 1.0)

    model.set_normalization(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )

    return x_mean, x_scale, y_mean, y_scale


def train_model(
    model: SurrogateMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[SurrogateMLP, pd.DataFrame, int]:
    """Train using validation loss for early stopping."""
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)

    x_validation_tensor = torch.as_tensor(
        x_validation,
        dtype=torch.float32,
    )
    y_validation_tensor = torch.as_tensor(
        y_validation,
        dtype=torch.float32,
    )

    y_mean = model.y_mean
    y_scale = model.y_scale

    y_train_scaled = (y_train_tensor - y_mean) / y_scale
    y_validation_scaled = (y_validation_tensor - y_mean) / y_scale

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_sample_count = 0

        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = model.forward_scaled(inputs)
            loss = loss_function(predictions, targets)

            loss.backward()
            optimizer.step()

            batch_count = inputs.shape[0]
            training_loss_sum += loss.item() * batch_count
            training_sample_count += batch_count

        training_loss = training_loss_sum / training_sample_count

        model.eval()
        with torch.no_grad():
            validation_predictions = model.forward_scaled(
                x_validation_tensor
            )
            validation_loss = loss_function(
                validation_predictions,
                y_validation_scaled,
            ).item()

        history.append(
            {
                "epoch": float(epoch),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train MSE: {training_loss:.6e} | "
                f"validation MSE: {validation_loss:.6e}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    return model, pd.DataFrame(history), best_epoch


def predict(
    model: SurrogateMLP,
    inputs: np.ndarray,
) -> np.ndarray:
    """Predict physical outputs, not normalized targets."""
    model.eval()

    with torch.no_grad():
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        predictions = model(tensor).cpu().numpy()

    if not np.isfinite(predictions).all():
        raise RuntimeError("Model produced non-finite predictions.")

    return predictions


def evaluate_predictions(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> pd.DataFrame:
    """Calculate held-out metrics separately for each physical target."""
    records: list[dict[str, float | str]] = []

    for index, target in enumerate(TARGET_COLUMNS):
        actual = y_true[:, index]
        predicted = y_predicted[:, index]

        records.append(
            {
                "target": target,
                "units": TARGET_UNITS[target],
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(
                    np.sqrt(mean_squared_error(actual, predicted))
                ),
                "R2": float(r2_score(actual, predicted)),
            }
        )

    return pd.DataFrame(records)


def save_plots(
    output_dir: Path,
    history: pd.DataFrame,
    y_test: np.ndarray,
    y_predicted: np.ndarray,
) -> None:
    """Save the training curve and per-target prediction comparisons."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        history["epoch"],
        history["training_loss"],
        label="Training",
    )
    axis.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized MSE")
    axis.set_title("Surrogate Training History")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_history.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))
    axes_flat = axes.ravel()

    for index, target in enumerate(TARGET_COLUMNS):
        axis = axes_flat[index]
        actual = y_test[:, index]
        predicted = y_predicted[:, index]

        axis.scatter(actual, predicted, alpha=0.65, s=18)

        lower = min(float(actual.min()), float(predicted.min()))
        upper = max(float(actual.max()), float(predicted.max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--")

        axis.set_xlabel(f"Actual {target} ({TARGET_UNITS[target]})")
        axis.set_ylabel(f"Predicted {target} ({TARGET_UNITS[target]})")
        axis.set_title(f"{target}: actual vs. predicted")
        axis.grid(True, alpha=0.3)

    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive.")

    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    set_seed(args.seed)

    # This baseline is CPU-compatible and avoids requiring a GPU.
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Loading dataset: {args.data}")

    data = load_dataset(args.data)
    train_data, validation_data, test_data = split_dataset(
        data,
        args.seed,
    )

    print(f"Total designs: {len(data)}")
    print(
        f"Train: {len(train_data)} | "
        f"Validation: {len(validation_data)} | "
        f"Test: {len(test_data)}"
    )

    x_train, y_train = make_arrays(train_data)
    x_validation, y_validation = make_arrays(validation_data)
    x_test, y_test = make_arrays(test_data)

    if x_train.shape[1] != INPUT_DIM or y_train.shape[1] != OUTPUT_DIM:
        raise ValueError("Dataset dimensions do not match the surrogate model.")

    model = SurrogateMLP().to(device)

    x_mean, x_scale, y_mean, y_scale = fit_normalization(
        model,
        x_train,
        y_train,
    )

    start_time = time.perf_counter()

    model, history, best_epoch = train_model(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    training_seconds = time.perf_counter() - start_time

    # Evaluate the test partition only after training and model selection.
    y_predicted = predict(model, x_test)
    metrics = evaluate_predictions(y_test, y_predicted)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "model_state.pt"
    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "test_metrics.csv"
    history_path = args.output_dir / "training_history.csv"

    # Save weights and normalization buffers using PyTorch's state_dict.
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_class": "SurrogateMLP",
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "target_units": TARGET_UNITS,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
        "seed": args.seed,
        "dataset_path": str(args.data),
        "dataset_rows": len(data),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_scale": x_scale.tolist(),
            "y_mean": y_mean.tolist(),
            "y_scale": y_scale.tolist(),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    metrics.to_csv(metrics_path, index=False)
    history.to_csv(history_path, index=False)

    save_plots(
        args.output_dir,
        history,
        y_test,
        y_predicted,
    )

    print("\nHELD-OUT TEST METRICS")
    print("=" * 80)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    print(f"\nBest epoch: {best_epoch}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Model weights: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Test metrics: {metrics_path}")
    print(f"Training history: {history_path}")
    print(f"Plots: {args.output_dir / 'figures'}")

    print(
        "\nInterpretation: R2 values near 1 indicate strong predictive "
        "performance on the held-out test data. Inspect each target's "
        "MAE and RMSE in its physical units; do not rely on R2 alone."
    )


if __name__ == "__main__":
    main()
