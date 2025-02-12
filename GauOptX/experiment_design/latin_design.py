import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import RandomDesign

class GauOptXLatinDesign(ExperimentDesign):
    """
    Latin Hypercube Experiment Design.
    Employs random sampling for discrete variables and Latin hypercube sampling for continuous variables.
    """
    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Latin design does not support sampling with constraints.')
        super(GauOptXLatinDesign, self).__init__(space)

    def get_samples(self, init_points_count, criterion='center'):
        """
        Generate the specified number of sample points.

        :param init_points_count: Number of samples to generate.
        :param criterion: Specifies the type of Latin Hypercube sampling. 
                          Default is 'center'. Refer to pyDOE.lhs documentation for more details on this parameter.
        :returns: The generated sample points.
        """
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random sampling to populate non-continuous variables
        random_design = RandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            bounds = self.space.get_continuous_bounds()
            lower_bound = np.asarray(bounds)[:, 0].reshape(1, len(bounds))
            upper_bound = np.asarray(bounds)[:, 1].reshape(1, len(bounds))
            diff = upper_bound - lower_bound

            from pyDOE import lhs
            X_design_aux = lhs(len(self.space.get_continuous_bounds()), init_points_count, criterion=criterion)
            I = np.ones((X_design_aux.shape[0], 1))
            X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)

            samples[:, self.space.get_continuous_dims()] = X_design

        return samples
