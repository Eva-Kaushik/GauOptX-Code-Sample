import abc
from six import with_metaclass

class BOModel(with_metaclass(abc.ABCMeta, object)):
    """
    The abstract Model for Bayesian Optimization
    """
    
    MCMC_sampler = False
    analytical_gradient_prediction = False
    
    @abc.abstractmethod
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        "Augment the dataset of the model"
        return
    
    @abc.abstractmethod
    def predict(self, X):
        "Get the predicted mean and std at X."
        return

    def predict_withGradients(self, X):
        "Get the gradients of the predicted mean and variance at X."
        return

    @abc.abstractmethod
    def get_fmin(self):
        "Get the minimum of the current model."
        return
