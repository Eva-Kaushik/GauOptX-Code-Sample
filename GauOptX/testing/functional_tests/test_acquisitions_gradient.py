import numpy as np
import unittest
from unittest.mock import Mock

import GauOptX
from GauOptX.util.general import samples_multidimensional_uniform
from GauOptX.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB
from GauOptX.models.gradient_checker import GradientChecker

class AcquisitionForTest:
    '''
    Class to run the unit test for the gradients of the acquisitions
    '''
    def __init__(self, gauoptx_acq):
        self.gauoptx_acq = gauoptx_acq

    def acquisition_function(self, x):
        # Get the acquisition value
        return self.gauoptx_acq.acquisition_function_withGradients(x)[0]

    def d_acquisition_function(self, x):
        # Get the gradient of the acquisition function
        return self.gauoptx_acq.acquisition_function_withGradients(x)[1]


class TestAcquisitionsGradients(unittest.TestCase):
    '''
    Unittest for the gradients of the available acquisition functions
    '''

    def setUp(self):
        np.random.seed(10)
        self.tolerance = 0.05  # Tolerance for difference between true and approximated gradients
        
        # Objective function from GauOptX examples (Forrester)
        objective = GauOptX.objective_examples.experiments1d.forrester()
        
        # Define the feasible region for the design space
        self.feasible_region = GauOptX.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': objective.bounds[0]}])
        
        # Generate initial design
        n_initial_design = 10
        X = samples_multidimensional_uniform(objective.bounds, n_initial_design)
        Y = objective.f(X)
        self.X_test = samples_multidimensional_uniform(objective.bounds, n_initial_design)

        # Mock the model used for prediction
        self.model = Mock()
        self.model.get_fmin.return_value = 0.0
        self.model.predict_withGradients.return_value = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(X.shape), np.zeros(X.shape)

    def test_check_grads_EI(self):
        # Test for Expected Improvement (EI) gradient check
        acquisition_ei = AcquisitionForTest(AcquisitionEI(self.model, self.feasible_region))
        grad_ei = GradientChecker(acquisition_ei.acquisition_function, acquisition_ei.d_acquisition_function, self.X_test)
        self.assertTrue(grad_ei.checkgrad(tolerance=self.tolerance))

    def test_check_grads_MPI(self):
        # Test for Maximum Probability of Improvement (MPI) gradient check
        acquisition_mpi = AcquisitionForTest(AcquisitionMPI(self.model, self.feasible_region))
        grad_mpi = GradientChecker(acquisition_mpi.acquisition_function, acquisition_mpi.d_acquisition_function, self.X_test)
        self.assertTrue(grad_mpi.checkgrad(tolerance=self.tolerance))

    def test_check_grads_LCB(self):
        # Test for Lower Confidence Bound (LCB) gradient check
        acquisition_lcb = AcquisitionForTest(AcquisitionLCB(self.model, self.feasible_region))
        grad_lcb = GradientChecker(acquisition_lcb.acquisition_function, acquisition_lcb.d_acquisition_function, self.X_test)
        self.assertTrue(grad_lcb.checkgrad(tolerance=self.tolerance))


if __name__ == '__main__':
    unittest.main()
