"""Recompute and audit a saved canonical optimization run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_config import LOWER_BOUNDS, UPPER_BOUNDS
from src.optimization import constraint_margins, load_validation_allowances
from src.optimization_adapter import INPUT_COLUMNS, TARGET_COLUMNS, physics_outputs
from src.surrogate_inference import load_surrogate

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_NAMES = (
    "omega1_minimum", "omega2_minimum", "omega3_minimum",
    "theta2_maximum", "theta3_maximum", "alpha_maximum", "theta_rms_maximum",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def verify_run(run_dir: Path) -> dict:
    run_dir = project_path(run_dir)
    summary_path = run_dir / "optimization_summary.json"
    manifest_path = run_dir / "run_manifest.json"
    comparison_path = run_dir / "design_comparison.csv"
    if not all(path.is_file() for path in (summary_path, manifest_path, comparison_path,
                                            run_dir / "physics_verification.csv")):
        raise FileNotFoundError(f"Optimization run is incomplete: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_revision") is None:
        raise ValueError("Run manifest does not identify a source revision.")
    for filename, expected in manifest["outputs_sha256"].items():
        path = run_dir / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Run output hash mismatch: {filename}")

    inputs = manifest["inputs"]
    data_path = project_path(inputs["dataset_path"])
    model_dir = project_path(inputs["model_directory"])
    calibration_dir = project_path(inputs["calibration_directory"])
    exact_inputs = {
        "dataset_sha256": data_path,
        "model_metadata_sha256": model_dir / "selected_model_metadata.json",
        "calibration_metadata_sha256": calibration_dir / "diagnostic_metadata.json",
        "calibration_summary_sha256": calibration_dir / "validation_error_summary.csv",
        "calibration_directions_sha256": calibration_dir / "validation_constraint_direction.csv",
    }
    for field, path in exact_inputs.items():
        if not path.is_file() or sha256(path) != manifest.get(field):
            raise ValueError(f"Run input hash mismatch or missing input: {field}")
    model_metadata = json.loads((model_dir / "selected_model_metadata.json").read_text(encoding="utf-8"))
    checkpoint = model_dir / model_metadata["model_file"]
    if not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint hash mismatch or missing checkpoint.")

    baseline_design = np.asarray([summary["baseline"]["design"][key] for key in INPUT_COLUMNS])
    selected_design = np.asarray([summary["selected_design"]["design"][key] for key in INPUT_COLUMNS])
    lower, upper = np.asarray(LOWER_BOUNDS), np.asarray(UPPER_BOUNDS)
    if np.any(selected_design < lower) or np.any(selected_design > upper):
        raise ValueError("Selected design is outside configured physical bounds.")
    baseline_physics = physics_outputs(baseline_design)
    selected_physics = physics_outputs(selected_design)
    np.testing.assert_allclose(baseline_physics,
                               [summary["baseline"]["physics_outputs"][k] for k in TARGET_COLUMNS],
                               rtol=1e-10, atol=1e-12)
    expected_margins = constraint_margins(selected_physics, baseline_physics)
    tolerance = float(summary["configuration"]["physics_feasibility_tolerance"])
    if np.any(expected_margins < -tolerance):
        raise ValueError("Selected design fails independently recomputed physics constraints.")
    np.testing.assert_allclose(selected_physics,
                               [summary["selected_design"]["physics_outputs"][k] for k in TARGET_COLUMNS],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(expected_margins,
                               [summary["selected_design"]["physics_margins"][k] for k in CONSTRAINT_NAMES],
                               rtol=1e-9, atol=1e-11)
    actual_inertia = float(np.sum(selected_design[:3]))
    if not np.isclose(actual_inertia, summary["selection"]["total_inertia"], rtol=0, atol=1e-12):
        raise ValueError("Selected design and reported objective disagree.")

    selected_name = summary["selection"]["selected_solver"]
    accepted = [candidate for method in summary["solver_runs"].values()
                for candidate in method.get("candidate_records", [])
                if candidate.get("accepted_for_selection")]
    if selected_name == "BASELINE":
        if accepted:
            raise ValueError("Baseline fallback was selected despite an accepted improving candidate.")
    else:
        selected_record = next((candidate for candidate in accepted
                                if candidate.get("candidate") == selected_name), None)
        if selected_record is None:
            raise ValueError("Selected candidate is not classified as accepted in the run summary.")
        np.testing.assert_allclose(selected_design,
                                   [selected_record["design"][key] for key in INPUT_COLUMNS],
                                   rtol=0, atol=1e-12)
        global_allowance, low_allowance, cutoff = load_validation_allowances(
            calibration_dir, model_dir, data_path)
        allowance = low_allowance if actual_inertia <= cutoff else global_allowance
        inference = load_surrogate(model_dir)
        prediction = inference.predict_array(selected_design.reshape(1, -1))[0]
        surrogate_margins = constraint_margins(prediction, baseline_physics, allowance)
        if np.any(surrogate_margins < -float(summary["configuration"]["surrogate_feasibility_tolerance"])):
            raise ValueError("Selected design fails the recomputed conservative surrogate screen.")
        np.testing.assert_allclose(surrogate_margins,
                                   [selected_record["surrogate_margins"][key] for key in CONSTRAINT_NAMES],
                                   rtol=1e-8, atol=1e-10)

    manifest_design = np.asarray(manifest["selected"]["design"], dtype=float)
    np.testing.assert_allclose(manifest_design, selected_design, rtol=0, atol=1e-12)
    if manifest["selected"]["name"] != selected_name:
        raise ValueError("Run manifest selected candidate differs from summary JSON.")

    comparison = pd.read_csv(comparison_path)
    chosen = comparison[comparison["selected"]]
    if len(chosen) != 1 or chosen.iloc[0]["candidate"] != summary["selection"]["selected_solver"]:
        raise ValueError("Comparison CSV does not identify the same selected candidate as JSON.")
    csv_design = chosen.iloc[0][list(INPUT_COLUMNS)].to_numpy(dtype=float)
    np.testing.assert_allclose(csv_design, selected_design, rtol=0, atol=1e-12)
    if not np.isclose(float(chosen.iloc[0]["total_inertia"]), actual_inertia, rtol=0, atol=1e-12):
        raise ValueError("Comparison CSV objective does not match the selected design.")
    return {"run_directory": str(run_dir), "selected_candidate": summary["selection"]["selected_solver"],
            "total_inertia_kg_m2": actual_inertia,
            "physical_margins": dict(zip(CONSTRAINT_NAMES, expected_margins.tolist())),
            "in_bounds": True, "physics_feasible": True, "hashes_consistent": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently recompute a canonical optimization run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
