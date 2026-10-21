
"""Load and query the selected torsional-system surrogate.

Run from the project root.

Example:
    from src.surrogate_inference import load_surrogate

    surrogate = load_surrogate()
    prediction = surrogate.predict_one({
        "J1": 0.12,
        "J2": 0.08,
        "J3": 0.05,
        "k_theta1": 120.0,
        "k_theta2": 90.0,
        "k_theta3": 60.0,
    })
    print(prediction)

Input values must use the same physical units and conventions as the
dataset used to train the model.

This module loads a saved model. It does not train or select models.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import torch

from src.surrogate import SurrogateMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "surrogate_benchmark"

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

DEVICE = torch.device("cpu")


class SurrogateInference:
    """Validated inference interface for a saved surrogate model."""

    def __init__(self, artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR):
        artifact_path = Path(artifact_dir)
        self.artifact_dir = (artifact_path if artifact_path.is_absolute() else PROJECT_ROOT / artifact_path).resolve()
        metadata_path = self.artifact_dir / "selected_model_metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Selected-model metadata not found: {metadata_path}. "
                "Run 'python -m scripts.benchmark_surrogates' first."
            )

        self.metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if self.metadata.get("input_columns") != INPUT_COLUMNS:
            raise ValueError("Saved model input columns do not match.")

        if self.metadata.get("target_columns") != TARGET_COLUMNS:
            raise ValueError("Saved model target columns do not match.")

        if self.metadata.get("target_units") != TARGET_UNITS:
            raise ValueError("Saved model target units do not match.")

        model_file = self.metadata.get("model_file")
        model_type = self.metadata.get("model_type")

        if not model_file or Path(model_file).name != model_file:
            raise ValueError("Invalid model filename in metadata.")

        model_path = self.artifact_dir / model_file

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Selected model artifact not found: {model_path}"
            )
        expected_digest = self.metadata.get("checkpoint_sha256")
        if expected_digest:
            digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError("Selected model checkpoint fingerprint does not match metadata.")

        self.model_type = model_type
        self.model_name = self.metadata["model"]
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load either a scikit-learn model or a PyTorch MLP."""
        if self.model_type == "sklearn":
            model = joblib.load(model_path)

            expected_class = self.metadata.get("estimator_class")
            if expected_class and type(model).__name__ != expected_class:
                raise ValueError(
                    "Loaded estimator type does not match metadata."
                )

            return model

        if self.model_type == "pytorch":
            hidden_dims = tuple(self.metadata["hidden_dims"])

            model = SurrogateMLP(
                input_dim=len(INPUT_COLUMNS),
                output_dim=len(TARGET_COLUMNS),
                hidden_dims=hidden_dims,
            ).to(DEVICE)

            state = torch.load(
                model_path,
                map_location=DEVICE,
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            model.eval()
            return model

        raise ValueError(
            f"Unsupported model type: {self.model_type!r}"
        )

    @staticmethod
    def _validate_inputs(
        inputs: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Convert one design or a batch of designs to a validated array."""
        if isinstance(inputs, Mapping):
            missing = [name for name in INPUT_COLUMNS if name not in inputs]
            extra = [name for name in inputs if name not in INPUT_COLUMNS]

            if missing or extra:
                raise ValueError(
                    f"Invalid input keys. Missing: {missing}; extra: {extra}"
                )

            array = np.asarray(
                [inputs[name] for name in INPUT_COLUMNS],
                dtype=np.float64,
            )

        else:
            try:
                array = np.asarray(inputs, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Inputs must be numeric."
                ) from exc

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2 or array.shape[1] != len(INPUT_COLUMNS):
            raise ValueError(
                "Inputs must have shape (6,) or (n_samples, 6), "
                f"in this column order: {INPUT_COLUMNS}"
            )

        if array.shape[0] == 0:
            raise ValueError("At least one design must be supplied.")

        if not np.isfinite(array).all():
            raise ValueError("Inputs must contain only finite values.")

        if np.any(array <= 0.0):
            raise ValueError(
                "All inertia and stiffness inputs must be positive."
            )

        return array

    def predict_array(
        self,
        inputs: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return physical predictions with shape (n_samples, 7)."""
        array = self._validate_inputs(inputs)

        if self.model_type == "sklearn":
            predictions = np.asarray(
                self.model.predict(array),
                dtype=np.float64,
            )
        else:
            self.model.eval()
            with torch.inference_mode():
                tensor = torch.as_tensor(
                    array,
                    dtype=torch.float32,
                    device=DEVICE,
                )
                predictions = self.model(tensor).cpu().numpy()

        predictions = np.asarray(predictions, dtype=np.float64)

        expected_shape = (array.shape[0], len(TARGET_COLUMNS))
        if predictions.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected prediction shape {predictions.shape}; "
                f"expected {expected_shape}."
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError("Surrogate produced non-finite predictions.")

        return predictions

    def predict_one(
        self,
        inputs: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> dict[str, dict[str, float | str]]:
        """Predict one design and return each output with its physical unit."""
        predictions = self.predict_array(inputs)

        if predictions.shape[0] != 1:
            raise ValueError(
                "predict_one() accepts exactly one design. "
                "Use predict_array() for a batch."
            )

        values = predictions[0]

        return {
            name: {
                "value": float(values[i]),
                "unit": TARGET_UNITS[name],
            }
            for i, name in enumerate(TARGET_COLUMNS)
        }

    def predict_batch(
        self,
        inputs: Sequence[Sequence[float]] | np.ndarray,
    ) -> list[dict[str, dict[str, float | str]]]:
        """Predict multiple designs and return named outputs with units."""
        predictions = self.predict_array(inputs)

        return [
            {
                name: {
                    "value": float(row[i]),
                    "unit": TARGET_UNITS[name],
                }
                for i, name in enumerate(TARGET_COLUMNS)
            }
            for row in predictions
        ]


def load_surrogate(
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> SurrogateInference:
    """Load the currently selected surrogate."""
    return SurrogateInference(artifact_dir)
