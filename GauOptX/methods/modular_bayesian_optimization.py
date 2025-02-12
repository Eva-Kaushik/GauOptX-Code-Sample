from ..core.bo import BO

class ModularBayesianOptimization(BO):
    """
    Modular Bayesian optimization. This class wraps the optimization loop around different handlers.

    :param model: GauOptX model class.
    :param space: GauOptX space class.
    :param objective: GauOptX objective class.
    :param acquisition: GauOptX acquisition class.
    :param evaluator: GauOptX evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GauOptX cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: instantiated de_duplication GauOptX class.
    """

    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost=None, normalize_Y=True, model_update_interval=1, de_duplication=False):
        
        self.initial_iter = True
        self.modular_optimization = True

        # --- Create optimization space
        super(ModularBayesianOptimization, self).__init__(model=model,
                                                         space=space,
                                                         objective=objective,
                                                         acquisition=acquisition,
                                                         evaluator=evaluator,
                                                         X_init=X_init,
                                                         Y_init=Y_init,
                                                         cost=cost,
                                                         normalize_Y=normalize_Y,
                                                         model_update_interval=model_update_interval,
                                                         de_duplication=de_duplication)
