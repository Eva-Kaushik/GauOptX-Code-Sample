# Copyright (c) 2024-2025, Eva Kaushik

import numpy as np
from GauOptX.objective_examples.experiments2d import branin as branin_creator 

# Create a Branin function instance
branin_instance = branin_creator()

def branin(x, y):
    """Evaluates the Branin function at given x and y coordinates."""
    return branin_instance.f(np.array([x, y]))  # Ensures input is always a 2D NumPy array
