# Arbor Press Design

Iterative arbor-press design study covering static FEA, engineering drawings, assembly/mold work, and Mastercam programming.

## Engineering objective

The project evaluated how geometry, material selection, fillets, wall thickness, and manufacturing method affect the structural performance and manufacturability of an arbor-press frame. Four revisions were documented: P1, P2, P3, and P4.

## Design ownership and method

- P1 was a provided completed SolidWorks part and is not included as original design work.
- P2 was modeled from the provided drawing as a cast-style revision with thicker lower sections, larger fillets, and smoother transitions.
- P3 was revised from the baseline as a machined design with geometry changes intended to improve load distribution while maintaining machinability.
- P4 was designed without a provided drawing using the earlier FEA results to reinforce high-stress regions and develop a cast-oriented frame.
- Parts were modeled in SolidWorks and documented with native drawings and exported drawing PDFs.
- A P2 mold assembly was created using the cavity workflow; its native assembly and mold components are included.

## FEA results

The report documents SolidWorks Simulation static studies using a 3 ton load, a fixed entire flat bottom surface, default solid meshing, and the reported materials: 6061-T6 aluminum for P1/P3 and gray cast iron for P2/P4. These are calculated/simulated results, not physical measurements.

| Revision | Maximum stress | Reported volume |
| --- | ---: | ---: |
| P1 | 301.1 MPa | 27.57 in³ |
| P2 | 346 MPa | 25.22 in³ |
| P3 | 311.4 MPa | 23.86 in³ |
| P4 | 188.7 MPa | 29.50 in³ |

The report identifies P4 as the strongest of the four documented cases by maximum stress. No independent physical load test or experimental validation is claimed.

## Manufacturing evidence

Mastercam source files are included for the documented Arbor Press P1 machining operation and the P2 bottom-mold operation. The report describes stock setup, roughing, finishing, and Color Loop toolpath verification using a provided tool library. The included files are manufacturing-programming evidence; they do not establish that the parts were physically machined.

## Files

- [Design report](Documentation/arbor_press_design_report.pdf)
- `CAD/Parts/` — student-modeled/revised P2, P3, and P4 frame parts
- `CAD/Drawings/` — native SolidWorks drawings and exported PDFs for P2–P4
- `CAD/P2-Mold-Assembly/` — native P2 mold assembly, mold parts, drawing, and PDF
- `Manufacturing/` — native Mastercam source files for the documented P1 and P2-mold operations

## Limitations and public-release scope

The project should be interpreted with the report’s assumptions and setup details. The public copy excludes the provided P1 baseline part, provided drawing/programming templates, reference fixtures, tool libraries, solver-generated FEA files, STEP duplicates, and additional mold files outside the documented scope. The P2 mold assembly may still require the original SolidWorks reference context; its native references should be checked manually before treating it as a standalone reproduction.
