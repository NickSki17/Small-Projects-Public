import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.train_surrogate import TARGET_UNITS, split_dataset
from src.optimization import TARGET_COLUMNS, load_validation_allowances


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration(tmp_path):
    calibration = tmp_path / "calibration"
    model_dir = tmp_path / "model"
    calibration.mkdir(); model_dir.mkdir()
    dataset_path = tmp_path / "dataset.csv"
    frame = pd.DataFrame({"design_id": range(1, 41),
                          "J1": np.linspace(.01, .05, 40),
                          "J2": np.linspace(.05, .01, 40),
                          "J3": np.linspace(.01, .04, 40)})
    frame.to_csv(dataset_path, index=False)
    _, validation, _ = split_dataset(frame, 42)
    cutoff = float(validation[["J1", "J2", "J3"]].sum(axis=1).quantile(.1))
    low_count = int((validation[["J1", "J2", "J3"]].sum(axis=1) <= cutoff).sum())

    checkpoint = model_dir / "selected_model_state.pt"
    checkpoint.write_bytes(b"fixture weights")
    model = {"model": "MLP_very_wide_deep", "seed": 42,
             "model_file": checkpoint.name,
             "input_columns": ["J1", "J2", "J3", "k_theta1", "k_theta2", "k_theta3"],
             "target_columns": list(TARGET_COLUMNS), "target_units": TARGET_UNITS,
             "dataset_sha256": _sha(dataset_path), "checkpoint_sha256": _sha(checkpoint)}
    model_path = model_dir / "selected_model_metadata.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    rows = []
    for region, count, value in (("all_validation", len(validation), .2),
                                 ("lowest_inertia_validation_10pct", low_count, .1)):
        for target in TARGET_COLUMNS:
            rows.append({"region": region, "target": target,
                         "units": TARGET_UNITS[target], "sample_count": count,
                         "max_overprediction": value,
                         "max_underprediction": -value})
    summary_path = calibration / "validation_error_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    direction_rows = [{"region": region, "target": target,
                       "dangerous_error_direction": "frequency_overprediction" if i < 3 else "maximum_response_underprediction"}
                      for region in ("all_validation", "lowest_inertia_validation_10pct")
                      for i, target in enumerate(TARGET_COLUMNS)]
    direction_path = calibration / "validation_constraint_direction.csv"
    pd.DataFrame(direction_rows).to_csv(direction_path, index=False)
    ids_hash = hashlib.sha256(",".join(map(str, validation.design_id.tolist())).encode()).hexdigest()
    metadata = {"test_partition_evaluated": False,
                "error_sign_convention": "prediction - actual",
                "target_columns": list(TARGET_COLUMNS), "inertia_definition": "J1 + J2 + J3",
                "model": model["model"], "seed": 42,
                "dataset_sha256": _sha(dataset_path),
                "checkpoint_sha256": _sha(checkpoint),
                "model_metadata_sha256": _sha(model_path),
                "error_summary_sha256": _sha(summary_path),
                "constraint_direction_sha256": _sha(direction_path),
                "validation_design_ids_sha256": ids_hash,
                "validation_rows": len(validation), "low_inertia_validation_rows": low_count,
                "low_inertia_fraction": .1, "low_inertia_cutoff": cutoff}
    (calibration / "diagnostic_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return calibration, model_dir, dataset_path


def test_valid_calibration_produces_directional_physical_unit_allowances(tmp_path):
    calibration, model, dataset = _calibration(tmp_path)
    global_allowance, low_allowance, cutoff = load_validation_allowances(calibration, model, dataset)
    np.testing.assert_allclose(global_allowance, .2)
    np.testing.assert_allclose(low_allowance, .1)
    assert cutoff > 0


@pytest.mark.parametrize("corruption, message", [
    ("summary_hash", "error_summary_sha256"),
    ("dataset_identity", "dataset fingerprint"),
    ("target_order", "target ordering"),
    ("split_identity", "validation split fingerprint"),
    ("model_identity", "model identity/seed"),
    ("unit", "units are incompatible"),
])
def test_calibration_corruption_and_mismatch_are_rejected(tmp_path, corruption, message):
    calibration, model, dataset = _calibration(tmp_path)
    if corruption == "summary_hash":
        with (calibration / "validation_error_summary.csv").open("a") as stream:
            stream.write("tampered\n")
    elif corruption == "dataset_identity":
        frame = pd.read_csv(dataset); frame.loc[0, "J1"] += .001; frame.to_csv(dataset, index=False)
    else:
        path = calibration / "diagnostic_metadata.json"
        metadata = json.loads(path.read_text())
        if corruption == "target_order":
            metadata["target_columns"] = list(reversed(TARGET_COLUMNS))
        elif corruption == "split_identity":
            metadata["validation_design_ids_sha256"] = "incorrect"
        elif corruption == "model_identity":
            metadata["model"] = "other_model"
        elif corruption == "unit":
            summary_path = calibration / "validation_error_summary.csv"
            summary = pd.read_csv(summary_path); summary.loc[0, "units"] = "wrong"
            summary.to_csv(summary_path, index=False)
            metadata["error_summary_sha256"] = _sha(summary_path)
        path.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        load_validation_allowances(calibration, model, dataset)


def test_missing_direction_report_fails_clearly(tmp_path):
    calibration, model, dataset = _calibration(tmp_path)
    (calibration / "validation_constraint_direction.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_validation_allowances(calibration, model, dataset)
