# CNC Bracket Design

Comparative bracket design study covering SolidWorks modeling, static FEA, drawing release, fixturing documentation, and Mastercam programming.

## Engineering work

- Bracket P1 and P2 were modeled in SolidWorks from provided blueprints.
- The report documents a 6061-T6 aluminum static study with a 500 lbf load and fixed 0.875 in and 0.250 in through-holes.
- Reported maximum stresses were 482 MPa for P1 and 152 MPa for P2; reported minimum factors of safety were 0.571 and 1.81, respectively. These are reported SolidWorks Simulation results, not physical test measurements.
- P1 and P2 include native SolidWorks parts and drawings, exported drawing PDFs, and native Mastercam programming files.
- The report describes a 3.75 × 3.00 × 1.00 in stock setup, soft-jaw machining, and Color Loop toolpath verification. These are documented workflow details, not independent manufacturing validation.

## Evidence

- [Design report](Documentation/cnc_bracket_design_report.pdf)
- Native CAD: `CAD/Midterm P1-Skiba.SLDPRT`, `CAD/Midterm P2-Skiba.SLDPRT`, and their `.SLDDRW` files
- [P1 drawing PDF](Documentation/bracket_p1_drawing.pdf) and [P2 drawing PDF](Documentation/bracket_p2_drawing.pdf)
- Native Mastercam files: `Manufacturing/Midterm P1-Skiba.emcam` and `Manufacturing/Midterm P2-Skiba.emcam`

The public copy excludes provided instructions, templates, reference fixturing, soft-jaw/vice assemblies, solver-generated files, STEP duplicates, and intermediate exports.
