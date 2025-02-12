import numpy as np
from scipy.special import erfc
import time
from ..core.errors import InvalidConfigError

def compute_integrated_acquisition(acquisition, x):
    """
    Computes the acquisition function when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: Acquisition function with GauOptX model type GP_MCMC.
    :param x: Location where the acquisition is evaluated.
    :return: Averaged acquisition function value.
    """
    acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i, :]
        acqu_x += acquisition.acquisition_function(x)

    return acqu_x / acquisition.model.num_hmc_samples

def compute_integrated_acquisition_with_gradients(acquisition, x):
    """
    Computes the acquisition function with gradients when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: Acquisition function with GauOptX model type GP_MCMC.
    :param x: Location where the acquisition is evaluated.
    :return: Tuple of acquisition function value and its gradient.
    """
    acqu_x = 0
    d_acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i, :]
        acqu_x_sample, d_acqu_x_sample = acquisition.acquisition_function_withGradients(x)
        acqu_x += acqu_x_sample
        d_acqu_x += d_acqu_x_sample

    acqu_x /= acquisition.model.num_hmc_samples
    d_acqu_x /= acquisition.model.num_hmc_samples

    return acqu_x, d_acqu_x

def best_guess(f, X):
    """
    Computes the best current guess from a vector of evaluations.

    :param f: Function to evaluate.
    :param X: Locations to evaluate the function.
    :return: Best guesses at each step.
    """
    n = X.shape[0]
    x_best = np.zeros(n)
    for i in range(n):
        ff = f(X[:i + 1])
        x_best[i] = ff[np.argmin(ff)]
    return x_best

def samples_multidimensional_uniform(bounds, num_data):
    """
    Generates a multidimensional grid uniformly distributed.

    :param bounds: Tuple defining the box constraints.
    :param num_data: Number of data points to generate.
    :return: Uniformly distributed samples.
    """
    dim = len(bounds)
    Z_rand = np.zeros((num_data, dim))
    for k in range(dim):
        Z_rand[:, k] = np.random.uniform(low=bounds[k][0], high=bounds[k][1], size=num_data)
    return Z_rand

def reshape(x, input_dim):
    """
    Reshapes x into a matrix with input_dim columns.

    :param x: Input array.
    :param input_dim: Number of columns.
    :return: Reshaped array.
    """
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x

def get_moments(model, x):
    """
    Computes the moments (mean and standard deviation) of a GP model at x.

    :param model: GPy model.
    :param x: Input location(s).
    :return: Mean, standard deviation, and current minimum.
    """
    input_dim = model.X.shape[1]
    x = reshape(x, input_dim)
    fmin = min(model.predict(model.X)[0])
    m, v = model.predict(x)
    s = np.sqrt(np.clip(v, 0, np.inf))
    return m, s, fmin

def get_d_moments(model, x):
    """
    Computes gradients with respect to x of the moments (mean and standard deviation) of the GP.

    :param model: GPy model.
    :param x: Location where gradients are evaluated.
    :return: Gradients of mean and standard deviation.
    """
    input_dim = model.input_dim
    x = reshape(x, input_dim)
    _, v = model.predict(x)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:, :, 0]
    dsdx = dvdx / (2 * np.sqrt(v))
    return dmdx, dsdx

def get_quantiles(acquisition_par, fmin, m, s):
    """
    Computes quantiles of the Gaussian distribution useful for acquisition functions.

    :param acquisition_par: Acquisition function parameter.
    :param fmin: Current minimum.
    :param m: Vector of means.
    :param s: Vector of standard deviations.
    :return: Tuple of phi, Phi, and u values.
    """
    if isinstance(s, np.ndarray):
        s[s < 1e-10] = 1e-10
    elif s < 1e-10:
        s = 1e-10

    u = (fmin - m - acquisition_par) / s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return phi, Phi, u

def best_value(Y, sign=1):
    """
    Computes a vector where each component is the minimum or maximum of Y[:i].

    :param Y: Input vector.
    :param sign: 1 for minimum, -1 for maximum.
    :return: Best values vector.
    """
    n = Y.shape[0]
    Y_best = np.ones(n)
    for i in range(n):
        Y_best[i] = Y[:i + 1].min() if sign == 1 else Y[:i + 1].max()
    return Y_best

def spawn(f):
    """
    Function for parallel evaluation of the acquisition function.

    :param f: Function to evaluate.
    :return: Wrapped function.
    """
    def fun(pipe, x):
        pipe.send(f(x))
        pipe.close()
    return fun

def evaluate_function(f, X):
    """
    Evaluates a function and measures evaluation time.

    :param f: Function to evaluate.
    :param X: Input locations.
    :return: Function evaluations and evaluation times.
    """
    num_data, dim_data = X.shape
    Y_eval = np.zeros((num_data, dim_data))
    Y_time = np.zeros((num_data, 1))

    for i in range(num_data):
        time_zero = time.time()
        Y_eval[i, :] = f(X[i, :])
        Y_time[i, :] = time.time() - time_zero

    return Y_eval, Y_time

def values_to_array(input_values):
    """
    Transforms int, float, or tuple values into a column vector numpy array.

    :param input_values: Input values.
    :return: Transformed numpy array.
    """
    if isinstance(input_values, tuple):
        values = np.array(input_values).reshape(-1, 1)
    elif isinstance(input_values, np.ndarray):
        values = np.atleast_2d(input_values)
    elif isinstance(input_values, (int, float, np.int64)):
        values = np.atleast_2d(np.array(input_values))
    else:
        raise TypeError('Type to transform not recognized')
    return values

def merge_values(values1, values2):
    """
    Merges two numpy arrays by calculating all possible row combinations.

    :param values1: First array.
    :param values2: Second array.
    :return: Merged array.
    """
    array1 = values_to_array(values1)
    array2 = values_to_array(values2)

    if array1.size == 0:
        return array2
    if array2.size == 0:
        return array1

    merged_array = [
        np.hstack((row_array1, row_array2))
        for row_array1 in array1
        for row_array2 in array2
    ]
    return np.atleast_2d(merged_array)

def normalize(Y, normalization_type='stats'):
    """
    Normalizes a vector using statistics or its range.

    :param Y: Vector to normalize.
    :param normalization_type: 'stats' for mean and std or 'maxmin' for range.
    :return: Normalized vector.
    """
    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 1:
        raise ValueError('Only 1-dimensional arrays are supported.')

    if normalization_type == 'stats':
        Y_norm = Y - Y.mean()
        std = Y.std()
        if std > 0:
            Y_norm /= std
    elif normalization_type == 'maxmin':
        Y_norm = Y - Y.min()
        y_range = np.ptp(Y)
        if y_range > 0:
            Y_norm = 2 * ((Y_norm / y_range) - 0.5)
    else:
        raise ValueError(f'Unknown normalization type: {normalization_type}')

    return Y_norm
