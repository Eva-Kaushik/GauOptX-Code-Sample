from GauOptX.fmodels.experiments2d import branin as branin_creator
import numpy as np

# Initialize the Branin function
branin_function = branin_creator()

def branin(x: float, y: float) -> float:
    """Evaluates the Branin function.

    Args:
        x (float): First input variable.
        y (float): Second input variable.

    Returns:
        float: The function value at (x, y).
    """
    return branin_function.f(np.array([x, y]))
