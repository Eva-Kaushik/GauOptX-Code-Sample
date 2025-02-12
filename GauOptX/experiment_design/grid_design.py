import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import RandomDesign

class GridDesign(ExperimentDesign):
    """
    Grid-based experimental design approach.
    Uses random sampling for discrete variables and employs a square grid for continuous ones.
    """

    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Grid-based designs do not support sampling with constraints.')
        super(GridDesign, self).__init__(space)

    def _adjust_init_points_count(self, init_points_count):
        # Note: The total number of generated points is the smallest integer n^d closest to the requested points.
        print('Important: For grid-based designs, the total number of points is rounded to the nearest integer of n^d matching the selected point count.')
        continuous_dims = len(self.space.get_continuous_dims())
        self.data_per_dimension = iroot(continuous_dims, init_points_count)
        return self.data_per_dimension**continuous_dims

    def get_samples(self, init_points_count):
        """
        This function may generate fewer points than requested.
        The total number of points generated corresponds to the closest integer of n^d to the requested point count.
        """

        init_points_count = self._adjust_init_points_count(init_points_count)
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random sampling for filling the non-continuous variables
        random_design = RandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            X_design = multigrid(self.space.get_continuous_bounds(), self.data_per_dimension)
            samples[:, self.space.get_continuous_dims()] = X_design

        return samples

# Computes integer root
# The largest integer whose k-th power is less than or equal to n
# This is the maximum x such that x^k <= n
def iroot(k, n):
    # Uses Newton's method to compute the integer root
    # This approach iteratively finds the integer root by refining an initial guess
    u, s = n, n+1
    while u < s:
        s = u
        t = (k-1) * s + n // pow(s, k-1)
        u = t // k
    return s

def multigrid(bounds, points_count):
    """
    Generates a multi-dimensional grid (lattice).
    :param bounds: A set of box constraints for each dimension.
    :param points_count: Number of points to generate along each dimension.
    """
    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], points_count).reshape(points_count, 1)
    x_grid_rows = np.meshgrid(*[np.linspace(b[0], b[1], points_count) for b in bounds])
    x_grid_columns = np.vstack([x.flatten(order='F') for x in x_grid_rows]).T
    return x_grid_columns
