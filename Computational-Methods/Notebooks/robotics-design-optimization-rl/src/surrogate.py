
"""
PyTorch neural-network surrogate for the 3-DOF rotational torsional system.

Inputs:
    J1, J2, J3, k_theta1, k_theta2, k_theta3

Outputs:
    omega1, omega2, omega3,
    theta2_max, theta3_max, alpha_max, theta_rms

The model stores its input and output normalization parameters as buffers.
Training uses normalized targets; ordinary inference returns predictions
in the original physical units.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from numpy.typing import ArrayLike


INPUT_DIM = 6
OUTPUT_DIM = 7
DEFAULT_HIDDEN_DIMS = (64, 64)


class SurrogateMLP(nn.Module):
    """Fully connected neural network for mechanical response prediction."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")

        if output_dim < 1:
            raise ValueError("output_dim must be positive.")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        if any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in hidden_dims
        ):
            raise ValueError("Every hidden-layer width must be a positive integer.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for width in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, width),
                    nn.ReLU(),
                ]
            )
            previous_dim = width

        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # These buffers are included in state_dict and move with the model.
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_scale", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))
        self.register_buffer("y_scale", torch.ones(output_dim))

    def set_normalization(
        self,
        x_mean: ArrayLike,
        x_scale: ArrayLike,
        y_mean: ArrayLike,
        y_scale: ArrayLike,
    ) -> None:
        """
        Set normalization statistics calculated from training data only.

        Scale values must be strictly positive.
        """
        values = {
            "x_mean": (x_mean, self.input_dim),
            "x_scale": (x_scale, self.input_dim),
            "y_mean": (y_mean, self.output_dim),
            "y_scale": (y_scale, self.output_dim),
        }

        converted: dict[str, torch.Tensor] = {}

        for name, (value, expected_size) in values.items():
            array = np.asarray(value, dtype=np.float64)

            if array.shape != (expected_size,):
                raise ValueError(
                    f"{name} must contain exactly {expected_size} values."
                )

            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values.")

            if name.endswith("_scale") and np.any(array <= 0.0):
                raise ValueError(f"{name} values must be strictly positive.")

            converted[name] = torch.as_tensor(
                array,
                dtype=self.x_mean.dtype,
                device=self.x_mean.device,
            )

        with torch.no_grad():
            self.x_mean.copy_(converted["x_mean"])
            self.x_scale.copy_(converted["x_scale"])
            self.y_mean.copy_(converted["y_mean"])
            self.y_scale.copy_(converted["y_scale"])

    def forward_scaled(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return predictions in normalized target space.

        Use this method when calculating the training loss.
        Inputs are supplied in their original physical units.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape (n_samples, {self.input_dim})."
            )

        normalized_inputs = (inputs - self.x_mean) / self.x_scale
        return self.network(normalized_inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return predictions in original physical units.

        Output columns correspond to the seven documented target variables.
        """
        normalized_predictions = self.forward_scaled(inputs)
        return normalized_predictions * self.y_scale + self.y_mean
