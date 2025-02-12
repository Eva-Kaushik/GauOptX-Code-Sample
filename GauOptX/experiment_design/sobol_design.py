import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import GauOptXRandomDesign

class GauOptXSobolDesign(ExperimentDesign):
    """
    Sobol sequence experiment design.
    Uses random design for non-continuous variables and Sobol sequence for continuous variables.
    """
    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Constraint-based sampling is not allowed in Sobol design')
        super(GauOptXSobolDesign, self).__init__(space)

    def get_samples(self, init_points_count):
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random design for filling non-continuous variables
        random_design = GauOptXRandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            bounds = self.space.get_continuous_bounds()
            lower_bound = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
            upper_bound = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
            diff = upper_bound - lower_bound

            from sobol_seq import i4_sobol_generate
            X_design = np.dot(i4_sobol_generate(len(self.space.get_continuous_bounds()), init_points_count), np.diag(diff.flatten()))[None, :] + lower_bound
            samples[:, self.space.get_continuous_dims()] = X_design

        return samples
