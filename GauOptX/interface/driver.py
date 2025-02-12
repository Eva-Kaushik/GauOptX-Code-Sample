import numpy as np
import time
from ..methods import BayesianOptimization

class GauOptXOptimizerDriver(object):
    """
    This class handles the driving of the Bayesian optimization process based on the given configuration.
    """

    def __init__(self, config=None, objective_function=None, output_engine=None):
        """
        Initializes the optimization driver with configuration, objective function, and output engine.
        """
        
        if config is None:
            from .config_parser import default_config
            import copy
            self.config = copy.deepcopy(default_config)
        else:
            self.config = config
        self.objective_function = objective_function
        self.output_engine = output_engine

    def _get_objective(self, space):
        """
        Retrieves the objective function and prepares it for optimization.
        """
        obj_func = self.objective_function
        
        from ..core.task import SingleObjective
        return SingleObjective(obj_func, self.config['resources']['cores'], space=space, unfold_args=True)
        
    def _get_search_space(self):
        """
        Retrieves the search space and constraints based on the configuration.
        """
        assert 'space' in self.config, 'The search space is not defined in the configuration!'
        
        space_config = self.config['space']
        constraint_config = self.config['constraints']
        from ..core.task.space import DesignSpace
        return DesignSpace.fromConfig(space_config, constraint_config)
    
    def _get_prediction_model(self):
        """
        Retrieves the model for making predictions.
        """

        from copy import deepcopy
        model_args = deepcopy(self.config['model'])
        del model_args['type']
        from ..models import select_model
        
        return select_model(self.config['model']['type']).fromConfig(model_args)
        
    def _get_acquisition_function(self, model, space):
        """
        Retrieves the acquisition function used in the optimization process.
        """

        from copy import deepcopy        
        acquisition_config = deepcopy(self.config['acquisition']['optimizer'])
        acq_opt_name = acquisition_config['name']
        del acquisition_config['name']
        
        from ..optimization import AcquisitionOptimizer
        acquisition_optimizer = AcquisitionOptimizer(space, acq_opt_name, **acquisition_config)
        from ..acquisitions import select_acquisition
        return select_acquisition(self.config['acquisition']['type']).fromConfig(model, space, acquisition_optimizer, None, self.config['acquisition'])
    
    def _get_acquisition_evaluator(self, acquisition):
        """
        Retrieves the evaluator for the acquisition function.
        """

        from ..core.evaluators import select_evaluator
        from copy import deepcopy
        evaluator_args = deepcopy(self.config['acquisition']['evaluator'])
        del evaluator_args['type']
        return select_evaluator(self.config['acquisition']['evaluator']['type'])(acquisition, **evaluator_args)
    
    def _check_stop_condition(self, iterations, elapsed_time, converged):
        """
        Defines the stopping condition for the optimization process.
        """

        resources_config = self.config['resources']
        
        stop = False
        if converged == 0:
            stop = True
        if resources_config['maximum-iterations'] != 'NA' and iterations >= resources_config['maximum-iterations']:
            stop = True
        if resources_config['max-run-time'] != 'NA' and elapsed_time / 60.0 >= resources_config['max-run-time']:
            stop = True
        return stop
            
    def run_optimization(self):
        """
        Executes the Bayesian optimization process using the previously loaded configurations and models.
        """

        space = self._get_search_space()
        objective_func = self._get_objective(space)
        prediction_model = self._get_prediction_model()
        acquisition_function = self._get_acquisition_function(prediction_model, space)
        acquisition_evaluator = self._get_acquisition_evaluator(acquisition_function)
                
        from ..experiment_design import initial_design
        initial_samples = initial_design(self.config['initialization']['type'], space, self.config['initialization']['num-eval'])

        from ..methods import ModularBayesianOptimization
        optimizer = ModularBayesianOptimization(prediction_model, space, objective_func, acquisition_function, acquisition_evaluator, initial_samples)
                
        optimizer.run_optimization(max_iter=self.config['resources']['maximum-iterations'],
                                   max_time=self.config['resources']['max-run-time'] if self.config['resources']['max-run-time'] != "NA" else np.inf,
                                   eps=self.config['resources']['tolerance'],
                                   verbosity=True)        
        return optimizer
