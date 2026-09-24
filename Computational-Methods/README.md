# Computational Methods

A collection of computational mechanics, numerical optimization, vibration
analysis, scientific machine learning, and reinforcement-learning projects.

## Repository structure

```text
Computational-Methods/
├── README.md
└── Notebooks/
    ├── optimization_and_vibration_analysis(original).ipynb
    ├── optimization-project/
    │   ├── optimization-project-code/
    │   ├── figures/
    │   └── results/
    └── physics-surrogate-rl/
        ├── README.md
        ├── requirements.txt
        ├── .gitignore
        ├── src/
        ├── scripts/
        ├── tests/
        ├── figures/
        └── data/
```

## Projects

### Project 1: Numerical Optimization and Vibration Analysis

`Notebooks/optimization-project/`

Completed and preserved as the repository's cleaned numerical optimization and
vibration-analysis project. Its cleaned Python modules are in
`Notebooks/optimization-project/optimization-project-code/`, with generated
figures and results alongside them. The original exploratory notebook is
retained in `Notebooks/` for reference and provenance.

### Project 2: Physics Surrogate RL

`Notebooks/physics-surrogate-rl/`

Scaffold-only project for physics-based design optimization, surrogate
modeling, and reinforcement-learning control. No physics, machine-learning,
optimization, or reinforcement-learning implementation has been added yet.

The planned workflow is:

1. Design parameters and physical modeling.
2. Numerical simulation and engineering metrics.
3. Synthetic dataset generation.
4. PyTorch surrogate modeling.
5. Constrained design optimization and true-physics validation.
6. Gymnasium control environment with PPO and PID comparison.

See the [Project 2 README](Notebooks/physics-surrogate-rl/README.md) for its
planned module responsibilities and dependency direction.
