from ..acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP, AcquisitionEntropySearch
from ..core.bo import BO
from ..core.errors import InvalidConfigError
from ..core.task.space import Design_space, bounds_to_space
from ..core.task.objective import SingleObjective
from ..core.task.cost import CostModel
from ..experiment_design import initial_design
from ..util.arguments_manager import ArgumentsManager
from ..core.evaluators import Sequential, RandomBatch, LocalPenalization, ThompsonBatch
from ..models.gpmodel import GPModel, GPModel_MCMC
from ..models.rfmodel import RFModel
from ..models.warpedgpmodel import WarpedGPModel
from ..models.input_warped_gpmodel import InputWarpedGPModel
from ..optimization.acquisition_optimizer import AcquisitionOptimizer
import GauOptX

import warnings
warnings.filterwarnings("ignore")

class BayesianOptimization(BO):
    """
    Core class to initialize a Bayesian Optimization method.
    :param f: function to optimize. It should accept 2D numpy arrays as input and return 2D outputs (one evaluation per row).
    :param domain: list of dictionaries describing input variables (see GauOptX.core.task.space.Design_space class for more information).
    :param constraints: list of dictionaries defining problem constraints (see GauOptX.core.task.space.Design_space class).
    :cost_withGradients: cost function associated with the objective. The input can be:
        - A function that returns both cost and derivatives for a set of points in the domain.
        - 'evaluation_time': A Gaussian process (mean) used to manage the evaluation cost.
    :model_type: model type used as surrogate:
        - 'GP', standard Gaussian process.
        - 'GP_MCMC', Gaussian process with prior in the hyper-parameters.
        - 'sparseGP', sparse Gaussian process.
        - 'warpedGP', warped Gaussian process.
        - 'InputWarpedGP', input warped Gaussian process.
        - 'RF', random forest (using scikit-learn).
    :param X: 2D numpy array containing initial inputs (one per row).
    :param Y: 2D numpy array containing initial outputs (one per row).
    :initial_design_numdata: number of initial data points to collect before beginning optimization.
    :initial_design_type: type of initial design:
        - 'random', random sampling.
        - 'latin', Latin hypercube sampling (discrete variables are sampled randomly).
    :acquisition_type: type of acquisition function used.
        - 'EI', expected improvement.
        - 'EI_MCMC', integrated expected improvement (requires GP_MCMC model).
        - 'MPI', maximum probability of improvement.
        - 'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model).
        - 'LCB', GP-Lower confidence bound.
        - 'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model).
    :param normalize_Y: whether to normalize the outputs before performing optimization (default is True).
    :exact_feval: whether the outputs are exact (default is False).
    :acquisition_optimizer_type: type of acquisition function optimizer:
        - 'lbfgs', L-BFGS.
        - 'DIRECT', Dividing Rectangles.
        - 'CMA', covariance matrix adaptation.
    :param model_update_interval: interval of collected observations to update the model (default is 1).
    :param evaluator_type: type of evaluator for the objective (batch size = 1 is equivalent for all methods).
        - 'sequential', sequential evaluations.
        - 'random', batch evaluation with random selections.
        - 'local_penalization', batch method based on local penalization (Gonzalez et al. 2016).
        - 'thompson_sampling', batch method using Thompson sampling.
    :param batch_size: number of samples for evaluating the objective (default is 1).
    :param num_cores: number of cores used for evaluation (default is 1).
    :param verbosity: whether to print details of models and options during optimization (default is False).
    :param maximize: whether to maximize the objective function (default is False).
    :param **kwargs: additional parameters to tune the optimization setup or use deprecated options.
    """

    def __init__(self, f, domain=None, constraints=None, cost_withGradients=None, model_type='GP', X=None, Y=None,
                 initial_design_numdata=5, initial_design_type='random', acquisition_type='EI', normalize_Y=True,
                 exact_feval=False, acquisition_optimizer_type='lbfgs', model_update_interval=1, evaluator_type='sequential',
                 batch_size=1, num_cores=1, verbosity=False, verbosity_model=False, maximize=False, de_duplication=False, **kwargs):
        
        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kwargs
        self.problem_config = ArgumentsManager(kwargs)

        # --- Select design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)

        # --- Select objective function
        self.maximize = maximize
        if 'objective_name' in kwargs:
            self.objective_name = kwargs['objective_name']
        else:
            self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = SingleObjective(self.f, self.batch_size, self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- Select the cost model
        self.cost = CostModel(cost_withGradients)

        # --- Select initial design
        self.X = X
        self.Y = Y
        self.initial_design_type = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- Select model type. If a user-defined model instance is passed, it will be used.
        self.model_type = model_type
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y

        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GauOptX.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User-defined model in use.'
                print('Using user-defined model.')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- Select acquisition optimizer
        kwargs.update({'model': self.model})
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type, **kwargs)

        # --- Select acquisition function. If a user-defined acquisition is passed, it will be used.
        self.acquisition_type = acquisition_type
        if 'acquisition' in self.kwargs:
            if isinstance(kwargs['acquisition'], GauOptX.acquisitions.AcquisitionBase):
                self.acquisition = kwargs['acquisition']
                self.acquisition_type = 'User-defined acquisition in use.'
                print('Using user-defined acquisition.')
            else:
                self.acquisition = self._acquisition_chooser()
        else:
            self.acquisition = self._acquisition_chooser()

        # --- Select evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(BayesianOptimization, self).__init__(model=self.model,
                                                  space=self.space,
                                                  objective=self.objective,
                                                  acquisition=self.acquisition,
                                                  evaluator=self.evaluator,
                                                  X_init=self.X,
                                                  Y_init=self.Y,
                                                  cost=self.cost,
                                                  normalize_Y=self.normalize_Y,
                                                  model_update_interval=self.model_update_interval,
                                                  de_duplication=self.de_duplication)

    def _model_chooser(self):
        return self.problem_config.model_creator(self.model_type, self.exact_feval, self.space)

    def _acquisition_chooser(self):
        return self.problem_config.acquisition_creator(self.acquisition_type, self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients)

    def _evaluator_chooser(self):
        return self.problem_config.evaluator_creator(self.evaluator_type, self.acquisition, self.batch_size, self.model_type, self.model, self.space, self.acquisition_optimizer)

    def _init_design_chooser(self):
        """
        Initializes the selection of X and Y based on the chosen initial design and the number of selected points.
        """

        # If objective function was not provided, we require some initial sample data
        if self.f is None and (self.X is None or self.Y is None):
            raise InvalidConfigError("Initial data for both X and Y is required when objective function is not provided")

        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)
        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)

    def _sign(self, f):
        if self.maximize:
            f_copy = f
            def f(x): return -f_copy(x)
        return f
