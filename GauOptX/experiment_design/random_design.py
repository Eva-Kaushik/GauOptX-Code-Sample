import numpy as np

from .base import ExperimentDesign
from ..core.task.variables import BanditVariable, DiscreteVariable, CategoricalVariable


class GauOptXRandomDesign(ExperimentDesign):
    """
    Random sampling experiment design.
    Generates random values for all variables within their respective bounds.
    """
    def __init__(self, space):
        super(GauOptXRandomDesign, self).__init__(space)

    def get_samples(self, init_points_count):
        if self.space.has_constraints():
            return self._generate_samples_with_constraints(init_points_count)
        else:
            return self._generate_samples_without_constraints(init_points_count)

    def _generate_samples_with_constraints(self, init_points_count):
        """
        Draw random samples and retain only those that fulfill the constraints.
        The process continues until the required number of valid samples is obtained.
        """
        samples = np.empty((0, self.space.dimensionality))

        while samples.shape[0] < init_points_count:
            domain_samples = self._generate_samples_without_constraints(init_points_count)
            valid_indices = (self.space.indicator_constraints(domain_samples) == 1).flatten()
            if sum(valid_indices) > 0:
                valid_samples = domain_samples[valid_indices, :]
                samples = np.vstack((samples, valid_samples))

        return samples[0:init_points_count, :]

    def fill_noncontinuous_variables(self, samples):
        """
        Assign values to non-continuous variables in the sample set.
        """
        init_points_count = samples.shape[0]

        for (idx, var) in enumerate(self.space.space_expanded):
            if isinstance(var, DiscreteVariable) or isinstance(var, CategoricalVariable):
                sample_var = np.atleast_2d(np.random.choice(var.domain, init_points_count))
                samples[:, idx] = sample_var.flatten()

            # Sampling for bandit variables
            elif isinstance(var, BanditVariable):
                # Bandit variables are represented by several adjacent columns in the samples array
                idx_samples = np.random.randint(var.domain.shape[0], size=init_points_count)
                bandit_idx = np.arange(idx, idx + var.domain.shape[1])
                samples[:, bandit_idx] = var.domain[idx_samples, :]

    def _generate_samples_without_constraints(self, init_points_count):
        samples = np.empty((init_points_count, self.space.dimensionality))

        self.fill_noncontinuous_variables(samples)

        if self.space.has_continuous():
            X_design = generate_uniform_samples(self.space.get_continuous_bounds(), init_points_count)
            samples[:, self.space.get_continuous_dims()] = X_design

        return samples


def generate_uniform_samples(bounds, points_count):
    """
    Generates uniformly distributed multidimensional samples within the given bounds.
    :param bounds: A tuple defining the box constraints.
    :param points_count: The number of sample points to generate.
    """
    dim = len(bounds)
    Z_rand = np.zeros(shape=(points_count, dim))
    for k in range(0, dim):
        Z_rand[:, k] = np.random.uniform(low=bounds[k][0], high=bounds[k][1], size=points_count)
    return Z_rand
