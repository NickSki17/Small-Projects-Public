
import numpy as np
import pytest
import torch

from src.surrogate import (
    DEFAULT_HIDDEN_DIMS,
    INPUT_DIM,
    OUTPUT_DIM,
    SurrogateMLP,
)


@pytest.fixture
def model():
    return SurrogateMLP()


def test_default_architecture_dimensions(model):
    assert model.input_dim == 6
    assert model.output_dim == 7
    assert model.hidden_dims == DEFAULT_HIDDEN_DIMS


def test_model_output_shape_for_batch(model):
    inputs = torch.randn(32, INPUT_DIM)

    predictions = model(inputs)

    assert predictions.shape == (32, OUTPUT_DIM)


def test_model_accepts_single_sample_batch(model):
    inputs = torch.randn(1, INPUT_DIM)

    assert model(inputs).shape == (1, OUTPUT_DIM)


def test_model_rejects_wrong_input_shape(model):
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(10, 5))


def test_scaled_forward_returns_correct_shape(model):
    inputs = torch.randn(8, INPUT_DIM)

    predictions = model.forward_scaled(inputs)

    assert predictions.shape == (8, OUTPUT_DIM)


def test_normalization_changes_predictions(model):
    inputs = torch.ones(3, INPUT_DIM)

    original = model(inputs).detach().clone()

    model.set_normalization(
        x_mean=np.ones(INPUT_DIM),
        x_scale=np.full(INPUT_DIM, 2.0),
        y_mean=np.full(OUTPUT_DIM, 10.0),
        y_scale=np.full(OUTPUT_DIM, 3.0),
    )

    updated = model(inputs).detach()

    assert not torch.allclose(original, updated)


def test_zero_normalized_output_maps_to_target_mean(model):
    y_mean = np.arange(OUTPUT_DIM, dtype=float) + 5.0
    y_scale = np.full(OUTPUT_DIM, 2.0)

    model.set_normalization(
        x_mean=np.zeros(INPUT_DIM),
        x_scale=np.ones(INPUT_DIM),
        y_mean=y_mean,
        y_scale=y_scale,
    )

    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()

    predictions = model(torch.zeros(2, INPUT_DIM))

    expected = torch.tensor(y_mean, dtype=torch.float32).repeat(2, 1)
    torch.testing.assert_close(predictions, expected)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("x_mean", np.zeros(5)),
        ("x_scale", np.zeros(6)),
        ("y_mean", np.zeros(6)),
        ("y_scale", np.full(7, -1.0)),
        ("x_mean", np.full(6, np.nan)),
        ("y_mean", np.full(7, np.inf)),
    ],
)
def test_invalid_normalization_is_rejected(model, name, value):
    defaults = {
        "x_mean": np.zeros(INPUT_DIM),
        "x_scale": np.ones(INPUT_DIM),
        "y_mean": np.zeros(OUTPUT_DIM),
        "y_scale": np.ones(OUTPUT_DIM),
    }

    defaults[name] = value

    with pytest.raises(ValueError):
        model.set_normalization(**defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0},
        {"output_dim": 0},
        {"hidden_dims": ()},
        {"hidden_dims": (64, 0)},
        {"hidden_dims": (64, -1)},
    ],
)
def test_invalid_architecture_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SurrogateMLP(**kwargs)


def test_model_outputs_are_finite(model):
    predictions = model(torch.randn(10, INPUT_DIM))

    assert torch.isfinite(predictions).all()


def test_normalization_buffers_are_in_state_dict(model):
    state = model.state_dict()

    assert "x_mean" in state
    assert "x_scale" in state
    assert "y_mean" in state
    assert "y_scale" in state
