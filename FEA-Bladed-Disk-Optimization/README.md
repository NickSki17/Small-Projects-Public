# FEA Bladed-Disk Optimization

Rotating bladed-disk analysis and optimization study using Ansys Workbench. The private source project contains structural, modal, and topology-optimization systems and includes a model label for a final geometry at 900 RPM.

## Engineering problem

The project addresses the analysis and geometric optimization of a bladed disk/rotor. The available project metadata shows a parameterized blade model and multiple analysis/optimization systems. The exact formal objective function and constraint set are not independently recoverable from the public-sized files, so this README does not infer whether the optimization prioritized mass, stiffness, stress, or another measure.

## Model and method

- Software: Ansys Workbench.
- The private Workbench source shows structural, modal, thermal, and structural topology-optimization systems, plus optimization design-point inputs and outputs.
- The model contains a label, `Final Geometry(900RPM)`, indicating the documented operating/model case represented by that geometry.
- The private project metadata includes `Solid Mass` and topology-optimization result containers, but numerical result values and exported optimized geometry are not available in this public copy.

## Results and evidence

The supported quantitative evidence in this public copy is limited to:

- the `900 RPM` model label in the private Workbench project.

No physical test results are included. No final stress, deformation, modal-frequency, mass-reduction, mesh-convergence, or optimization-performance value is claimed here because those values could not be reliably extracted from the available report/project files without opening the original Ansys environment and packaged project.

## Files

- [Design study report](Results/fea_bladed_disk_optimization_report.pdf) — available report evidence.

## Limitations and reproducibility

This is an analysis study, not an experimentally validated design. The original `.wbpj` was inspected but is not included because it embeds absolute local paths into a user Downloads directory and may depend on files contained in the original Ansys package or generated working directory. The original `.wbpz` package is approximately 1.49 GB and is intentionally excluded from this public repository. No cache, solver-output, temporary, or instructor/reference files were copied.

The report is image-heavy and the public-sized evidence does not establish the exact loads, boundary conditions, mesh settings, optimization objective, constraints, or final numerical results. Manual review in Ansys and review of the report are still required before treating the project as fully reproducible or publication-ready.
