# Copyright (c) 2024-2025, GauOptX. All rights reserved.

import math

def camel(x: float, y: float) -> float:
    """Computes the Six-Hump Camel Function value for given (x, y) coordinates."""
    x2 = x ** 2
    x4 = x ** 4
    y2 = y ** 2

    return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x * y + (-4.0 + 4.0 * y2) * y2

def main(job_id: int, params: dict) -> float:
    """Evaluates the camel function for given job ID and parameter dictionary."""
    x = params['x'][0]  # Lowercase keys for consistency
    y = params['y'][0]
    result = camel(x, y)

    print("The Six-Hump Camel Function:")
    print(f"\tf({x:.4f}, {y:.4f}) = {result:.6f}")

    return result

if __name__ == "__main__":
    main(23, {'x': [0.0898], 'y': [-0.7126]})  # Dictionary keys
