# FEA Topology Optimization

Structural and topology-optimization study using Ansys Workbench. The curated copy includes the report, exported initial/optimized geometry, and an optimized geometry result file.

## Evidence

- `Results/fea_topology_optimization_report.pdf` contains the available report and result figures.
- `Results/initial_topology.stl` and `Results/optimized_topology.stl` are exported geometry files.
- `Results/optimized_geometry.pmdb` is an exported Ansys geometry/result file.

The original Workbench project and `.wbpz` package are excluded. The Workbench file contained absolute local paths and was not portable as a standalone public source file. Generated solver/cache data are also excluded because of size. Exact boundary conditions, mesh settings, and numerical results should be taken from the report rather than inferred from the exported geometry alone.
