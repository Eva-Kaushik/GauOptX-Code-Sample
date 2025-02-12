import scipy
import numpy as np

from .base import AcquisitionBase
from .EI import AcquisitionEI
from ..util import epmgp
from ..models.gpmodel import GPModel


class GauOptX(AcquisitionBase):
    def __init__(self, model, space, sampler, optimizer=None, cost_withGradients=None,
                 num_samples=100, num_representer_points=50,
                 proposal_function=None, burn_in_steps=50):
        """
        Entropy Search acquisition function
        
        In a nutshell, entropy search approximates the
        distribution of the global minimum and tries to decrease its
        entropy. Current implementation does not provide analytical gradients, thus
        DIRECT optimizer is preferred over gradient descent for this acquisition.

        Parameters
        ----------
        :param model: GauOptX class of model
        :param space: GauOptX class of Design_space
        :param sampler: mcmc sampler for representer points, an instance of util.McmcSampler
        :param optimizer: optimizer of the acquisition. Should be a GauOptX optimizer
        :param cost_withGradients: function
        :param num_samples: integer determining how many samples to draw for each candidate input
        :param num_representer_points: integer determining how many representer points to sample
        :param proposal_function: Function that defines an unnormalized log proposal measure from which to sample the representer. The default is expected improvement.
        :param burn_in_steps: integer that defines the number of burn-in steps when sampling the representer points
        """
        if not isinstance(model, GPModel):
            raise RuntimeError("The current entropy search implementation supports only GPModel as model")

        self.optimizer = optimizer
        self.analytical_gradient_prediction = False
        AcquisitionBase.__init__(self, model, space, optimizer, cost_withGradients=cost_withGradients)

        self.input_dim = self.space.input_dim()
        
        self.num_repr_points = num_representer_points
        self.burn_in_steps = burn_in_steps
        self.sampler = sampler
        
        # (unnormalized) density from which to sample representer points
        self.proposal_function = proposal_function
        if self.proposal_function is None:
            bounds = space.get_bounds()
            mi = np.zeros(len(bounds))
            ma = np.zeros(len(bounds))
            for d in range(len(bounds)):
                mi[d] = bounds[d][0]
                ma[d] = bounds[d][1]
 
            ei = AcquisitionEI(model, space) 
            def prop_func(x):
                if len(x.shape) != 1:
                    raise ValueError("Expected a vector, received a matrix of shape {}".format(x.shape))
                if np.all(np.all(mi <= x)) and np.all(np.all(x <= ma)):
                    return np.log(np.clip(ei._compute_acq(x), 0., np.PINF))
                else:
                    return np.NINF

            self.proposal_function = prop_func

        # This is used later to calculate derivative of the stochastic part for the loss function
        self.W = scipy.stats.norm.ppf(np.linspace(1. / (num_samples + 1),
                                                  1 - 1. / (num_samples + 1),
                                                  num_samples))[np.newaxis, :]

        # Initialize parameters to lazily compute them once needed
        self.repr_points = None
        self.repr_points_log = None
        self.logP = None

    def _update_parameters(self):
        """
        Update parameters of the acquisition required to evaluate the function. In particular:
            * Sample representer points repr_points
            * Compute their log values repr_points_log
            * Compute belief locations logP
        """
        self.repr_points, self.repr_points_log = self.sampler.get_samples(self.num_repr_points, self.proposal_function, self.burn_in_steps)

        if np.any(np.isnan(self.repr_points_log)) or np.any(np.isposinf(self.repr_points_log)):
            raise RuntimeError("Sampler generated representer points with invalid log values: {}".format(self.repr_points_log))

        # Removing representer points that have 0 probability of being the minimum (corresponding to log probability being minus infinity)
        idx_to_remove = np.where(np.isneginf(self.repr_points_log))[0]
        if len(idx_to_remove) > 0:
            idx = list(set(range(self.num_repr_points)) - set(idx_to_remove))
            self.repr_points = self.repr_points[idx, :]
            self.repr_points_log = self.repr_points_log[idx]

        # We predict with the noise as we need to make sure that var is indeed positive definite.
        mu, _ = self.model.predict(self.repr_points)
        # we need a vector
        mu = np.ndarray.flatten(mu)
        var = self.model.predict_covariance(self.repr_points)
        
        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = epmgp.joint_min(mu, var, with_derivatives=True)
        # add a second dimension to the array
        self.logP = np.reshape(self.logP, (self.logP.shape[0], 1))

    def _required_parameters_initialized(self):
        """
        Checks if all required parameters are initialized.
        """
        return not (self.repr_points is None or self.repr_points_log is None or self.logP is None)

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError("Not implemented")

    def _compute_acq(self, x):
        # Naming of local variables here follows that in the paper

        if x.shape[1] != self.input_dim:
            message = "Dimensionality mismatch: x should be of size {}, but is of size {}".format(self.input_dim, x.shape[1])
            raise ValueError(message)

        if not self._required_parameters_initialized():
            self._update_parameters()

        if x.shape[0] > 1:
            results = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                results[j] = self._compute_acq(x[[j], :])
            return results

        # Number of belief locations
        N = self.logP.size

        # Evaluate innovation, these are gradients of mean and variance of the repr points wrt x
        # see method for more details
        dMdx, dVdx = self._innovations(x)
        
        # The transpose operator is there to make the array indexing equivalent to matlab's
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

        dMdx_squared = dMdx.dot(dMdx.T)
        trace_term = np.sum(np.sum(np.multiply(self.dlogPdMudMu, np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))), 2), 1)[:, np.newaxis]

        # Deterministic part of change:
        deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term
        # Stochastic part of change:
        stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(self.W)
        # Predicted new logP:
        predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
        max_predicted_logP = np.amax(predicted_logP, axis=0)

        # normalize predictions
        max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
        lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
        predicted_logP = np.subtract(predicted_logP, lselP)

        # We maximize the information gain
        dHp = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.repr_points_log)), axis=0)

        dH = np.mean(dHp)
        return dH # there is another minus in the public function

    def _compute_acq_withGradients(self, x):
        raise NotImplementedError("Analytic derivatives are not supported.")
    
    def _innovations(self, x):
        """
        :param x: candidate for which to compute the expected change in the GP
        :type x: np.array(1, input_dim)
        
        :return: innovation of mean (without samples) and variance at the representer points
        :rtype: (np.array(num_repr_points, 1), np.array(num_repr_points, num_repr_points))
                    
        """