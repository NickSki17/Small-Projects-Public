# Computational Methods

A collection of computational mechanics, numerical optimization, vibration analysis, scientific machine learning, and reinforcement-learning projects.

## Repository Structure

```text
Computational-Methods/
├── README.md
└── Notebooks/
    ├── optimization_and_vibration_analysis(original).ipynb
    ├── optimization-project/
    │   ├── optimization-project-code/
    │   ├── figures/
    │   └── results/
    └── robotics-design-optimization-rl/
        ├── README.md
        ├── requirements.txt
        ├── .gitignore
        ├── src/
        ├── scripts/
        ├── tests/
        ├── figures/
        ├── data/
        └── artifacts/
```

The tree shows the intended high-level organization. Generated files and individual module contents may vary as development progresses.

## Projects

### Project 1: Numerical Optimization and Vibration Analysis

**Location:** `Notebooks/optimization-project/`

A completed computational engineering project covering numerical optimization and mechanical vibration analysis.

The cleaned Python implementation is maintained in `optimization-project-code/`. Generated figures and numerical results are stored alongside the implementation. The original exploratory Jupyter notebook is preserved in `Notebooks/` for reference.

The project covers:

- Gradient Descent, Newton's method, and BFGS optimization.
- Convergence behavior across multiple test functions.
- Mechanical vibration and modal analysis.
- Frequency-domain analysis using FFT.
- Reproducible numerical results and visualizations.

### Project 2: Robotics Design Optimization and Reinforcement Learning

**Location:** `Notebooks/robotics-design-optimization-rl/`

An in-progress computational engineering project combining a three-degree-of-freedom rotational-torsional model, numerical simulation, engineering response metrics, neural-network surrogate modeling, constrained design optimization, and planned reinforcement-learning control.

The intended workflow is:

1. Define the rotational mechanical system and its physical parameters.
2. Compute natural frequencies and simulate time-domain responses.
3. Extract engineering response metrics and generate a reproducible dataset.
4. Train and evaluate a PyTorch surrogate.
5. Perform constrained design optimization and validate candidate designs using the original physics model.
6. Develop a compatible Gymnasium control environment and compare PPO against PID.
7. Evaluate prediction error, computational cost, control performance, and robustness.

#### Current development status

The rotational-torsional dataset generator, PyTorch surrogate definition, surrogate tests, and training script are implemented.

The dataset generator is configured to produce 5,000 designs with a fixed random seed. A 20-design smoke test completed successfully. The full dataset must be confirmed complete and validated before baseline training results can be reported.

The current neural-network architecture has two hidden layers of 64 units each. Deeper architectures may be compared experimentally, with architecture selection based on validation performance and final reporting based on a held-out test set.

Constrained optimization and reinforcement-learning control are downstream stages. They should not be described as completed until the implementations, validation procedures, and numerical results are available.

See the [Project 2 README](Notebooks/robotics-design-optimization-rl/README.md) for the workflow, model inputs and outputs, testing commands, implementation status, and limitations.

## Engineering Approach

Across these projects, the emphasis is on reproducible numerical methods, explicit model assumptions, software testing, quantitative evaluation, and validation against an appropriate reference.

Predictions from a numerical model or neural network are not automatically evidence of physical accuracy. Results should be accompanied by meaningful error metrics, limitations, and enough implementation detail to reproduce the analysis.