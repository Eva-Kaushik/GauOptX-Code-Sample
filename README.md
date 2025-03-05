# GauOptX: Gaussian Process Optimization Framework
This repository is a collaborative effort of Eva Kaushik(kaushikeva0026@gmail.com) and Rohit Kaushik(kaushikrohit004@gmail.com) 

## Introduction

**GauOptX** is a high-performance, research-driven optimization framework that leverages **Bayesian optimization with Gaussian processes (GPs)** for efficient global optimization. Unlike traditional optimization techniques, which rely on gradient-based methods or brute-force search, GauOptX intelligently balances exploration and exploitation to find optima with minimal function evaluations.

Bayesian optimization is particularly useful for expensive-to-evaluate functions, such as:
- Hyperparameter tuning in deep learning and machine learning models.
- Experimental design in physical sciences and engineering.
- Automated decision-making in reinforcement learning.

GauOptX provides a modular, extensible, and scalable implementation that supports **batch optimization, multi-fidelity optimization, and large-scale data handling.**

--------------------------
## Why Bayesian Optimization? The Science Behind It

Optimization problems often involve **black-box functions**—functions where we lack an explicit mathematical formula, and evaluation is costly (e.g., computationally expensive simulations, real-world experiments). Traditional methods struggle in these scenarios due to:

- **No Gradient Information**: Many real-world functions are non-differentiable or noisy, making gradient-based methods ineffective.
- **Computational Cost**: Grid search and brute-force methods are infeasible in high-dimensional spaces.
- **Local Minima Traps**: Many heuristic methods (e.g., hill climbing, simulated annealing) may converge to suboptimal solutions.

**Bayesian optimization (BO)** mitigates these issues by using a probabilistic model (typically a **Gaussian process**) to approximate the function and guide sampling based on uncertainty.

---------------------------
### Gaussian Processes (GPs) as a Surrogate Model

A **Gaussian process** is a non-parametric probabilistic model that provides a distribution over possible functions that fit observed data. Given a set of observations, a GP can predict the mean and variance at unobserved points, allowing us to make **informed decisions** about where to evaluate next.

Mathematically, a Gaussian process is defined as:
\[
 f(x) \sim \mathcal{GP} (m(x), k(x, x'))
\]
where:
- \(m(x)\) is the mean function (often set to zero for simplicity).
- \(k(x, x')\) is the covariance/kernel function (e.g., **RBF kernel**, Matern kernel), which determines the smoothness and complexity of the function approximation.

Using the GP model, we choose the next evaluation point using an **acquisition function**, which balances exploration (sampling in uncertain regions) and exploitation (sampling near promising points).

### Acquisition Functions

GauOptX supports multiple acquisition functions, each with different trade-offs:
- **Expected Improvement (EI)**: Prioritizes points that are likely to improve upon the current best solution.
- **Upper Confidence Bound (UCB)**: Encourages exploration based on confidence intervals.
- **Probability of Improvement (PI)**: Focuses on sampling points with a high probability of being better than the current optimum.
- **Thompson Sampling**: Uses random samples from the posterior to guide search.

--------------------------

## Key Features

- **Global Optimization**: Finds the global minimum of black-box functions efficiently.
- **Batch Optimization**: Supports parallel evaluations for faster convergence.
- **Multi-fidelity Optimization**: Integrates noisy and multi-resolution function evaluations.
- **Machine Learning Applications**: Optimize hyperparameters with fewer training iterations.
- **Custom Kernel Support**: Use built-in kernels or define your own covariance functions.
- **Visualization Tools**: Built-in plotting for convergence analysis.

## Installation

### Using `pip`
To install GauOptX, simply use:
```bash
pip install gauoptx
```

### Installing from Source
1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/GauOptX.git
    cd GauOptX
    ```
2. Install in development mode:
    ```bash
    python setup.py develop
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
----------------------------

## Dependencies

GauOptX requires the following libraries:
- `GPy` (for Gaussian process modeling)
- `numpy` and `scipy` (for numerical computations)
- `matplotlib` (for visualization)

Optional dependencies:
- `pyDOE` (for experimental design strategies)
- `sobol_seq` (for quasi-random sequence sampling)
- `cma` (for evolutionary strategies)
- `DIRECT` (for derivative-free optimization)

Install all required dependencies using:
```bash
pip install -r requirements.txt
```
------------------------------

## Usage Example

```python
from gauoptx import GauOptX
from gauoptx.benchmarks import synthetic_function

# Define the objective function
def objective_function(x):
    return synthetic_function(x)

# Initialize optimizer
optimizer = GauOptX(domain=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}])

# Run optimization
results = optimizer.optimize(objective_function)

# Display results
print("Optimal value found: ", results.x)
print("Objective value at optimum: ", results.fx)
optimizer.plot_convergence()
```
-------------------------------

## Real-World Applications

1. **Hyperparameter Optimization**: Bayesian optimization has been widely used in optimizing deep learning models (e.g., tuning CNN/LSTM architectures).
2. **Drug Discovery**: BO efficiently searches large chemical spaces for optimal molecular configurations.
3. **Robotics**: GauOptX can optimize control policies in reinforcement learning applications.
4. **Engineering Design**: Used for optimizing expensive simulation-based problems in aerospace, automotive, and energy sectors.

--------------------------------
## Contributing

We welcome contributions! Whether you’re fixing a bug, adding new functionality, or improving documentation, your help is greatly appreciated.

### How to Contribute:
1. Fork the repository.
2. Create a branch for your feature or bug fix:
    ```bash
    git checkout -b feature-or-bug
    ```
3. Make your changes and commit them.
4. Push to your branch:
    ```bash
    git push origin feature-or-bug
    ```
5. Submit a pull request.

---------------------------------

## Citation

If you use **GauOptX** in your research, please cite:
```
@article{Kaushik2025,
  author = {Eva Kaushik and Rohit Kaushik},
  title = {GauOptX: A Gaussian Process Optimization Framework},
  journal = {Journal of Machine Learning Optimization},
  year = {2025},
}
```
----------------------------------

## Contact
For support, contact the maintainer at `kaushikeva0026@gmail.com' or 'Kaushikrohit004@gmail.com' .

