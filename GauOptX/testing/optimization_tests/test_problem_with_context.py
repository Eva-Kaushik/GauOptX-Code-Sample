import numpy as np
import unittest

from GauOptX.core.task.space import Design_space
from GauOptX.optimization.acquisition_optimizer import ContextManager
from GauOptX.optimization.optimizer import OptimizationWithContext
from GauOptX.methods import BayesianOptimization
from GauOptX.objective_examples.experimentsNd import alpine1

class TestOptimizationWithContext(unittest.TestCase):
    """
    Class to test the mapping of the objective function through the context
    """
    def setUp(self):
        np.random.seed(123)
        domain          = [{'name': 'var1', 'type': 'continuous', 'domain': (-5, 5), 'dimensionality': 5}]
        space           = Design_space(domain)
        func            = alpine1(input_dim=5, bounds=space.get_bounds())
        bo              = BayesianOptimization(f=func.f, domain=domain)
        context         = {'var1_1': 0.3, 'var1_2': 0.4}
        context_manager = ContextManager(space, context)
        x0              = np.array([[0, 0, 0, 0, 0]])

        # initialize the model in a least intrusive way possible
        bo.suggest_next_locations()

        f = bo.acquisition.acquisition_function
        f_df = bo.acquisition.acquisition_function_withGradients
        self.problem_with_context = OptimizationWithContext(x0=x0, f=f, df=None, f_df=f_df, context_manager=context_manager)
        self.x = np.array([[3, -3, 3]])


    def test_objective_mapping_objective(self):
        """
        Test for the mapping through the context variables
        """
        f_nc_x = np.array([-0.10986074])
        self.assertTrue(np.isclose(self.problem_with_context.f_nc(self.x), f_nc_x).all())


    def test_gradient_mapping_objective(self):
        """
        Test for the gradient of the mapping through the context variables
        """
        df_nc_x = np.array([[0, 0, 0]])
        self.assertTrue(np.isclose(self.problem_with_context.df_nc(self.x), df_nc_x).all())


    def test_objective_and_mapping_objective(self):
        """
        Test for the mapping and the gradient through the context variables
        """
        f_nc_x = np.array([-0.10986074])
        df_nc_x = np.array([[0, 0, 0]])
        tested_mapping, tested_gradient = self.problem_with_context.f_df_nc(self.x)
        print(tested_mapping)
        print(tested_gradient)

        self.assertTrue(np.isclose(tested_mapping, f_nc_x).all())
        self.assertTrue(np.isclose(tested_gradient, df_nc_x).all())
