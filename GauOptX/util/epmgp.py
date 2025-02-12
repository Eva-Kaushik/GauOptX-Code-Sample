#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GauOptX: A Python package for Gaussian Process optimization using 
Entropy Propagation for Gaussian Processes (EPMGP) to compute probabilities.

License: Redistribution and use in source and binary forms, with or without
modification, are permitted under the conditions below:
1. Include the copyright notice and conditions in redistributions.
2. Do not use the name GauOptX or contributors' names for endorsements without permission.
3. No warranties provided. Use at your own risk.
"""

import numpy as np
from scipy import special

# Constants for numerical stability and optimization
SQRT_2 = np.sqrt(2)  # Square root of 2
EPS = np.finfo(np.float32).eps  # Machine epsilon for float32
LOG_2PI = np.log(2) + np.log(np.pi)  # Logarithm of 2*pi

def joint_min(mu, var, with_derivatives=False):
    """
    Calculate the probability of each point being the minimum using EPMGP.

    Parameters:
        mu (np.ndarray): Mean values of the N points (shape: (N,)).
        var (np.ndarray): Covariance matrix (shape: (N, N)).
        with_derivatives (bool, optional): Compute gradients if True.

    Returns:
        np.ndarray: Probability distribution of being the minimum (pmin).
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Gradients if with_derivatives=True.
    """
    log_probs = np.zeros(mu.shape)
    num_points = mu.shape[0]

    if with_derivatives:
        dlog_probs_mu = np.zeros((num_points, num_points))
        dlog_probs_sigma = np.zeros((num_points, int(0.5 * num_points * (num_points + 1))))

    for i in range(num_points):
        generator = min_factor(mu, var, i)
        log_probs[i] = next(generator)

        if with_derivatives:
            dlog_probs_mu[i, :] = next(generator).T
            dlog_probs_sigma[i, :] = next(generator).T

    log_probs = normalize_log_probs(log_probs)

    if not with_derivatives:
        return log_probs

    return log_probs, dlog_probs_mu, dlog_probs_sigma

def normalize_log_probs(log_probs):
    """
    Normalize log probabilities for numerical stability.

    Parameters:
        log_probs (np.ndarray): Logarithmic probabilities.

    Returns:
        np.ndarray: Normalized log probabilities.
    """
    max_log_prob = np.max(log_probs)
    log_probs -= max_log_prob
    log_probs -= np.log(np.sum(np.exp(log_probs)))
    return log_probs

def min_factor(mu, sigma, k):
    """
    Compute the log probability and its gradients for index k using iterative updates.

    Parameters:
        mu (np.ndarray): Mean values of the points.
        sigma (np.ndarray): Covariance matrix.
        k (int): Index of the point being evaluated.

    Yields:
        float: Log probability of the k-th point.
        np.ndarray: Gradients with respect to mu and sigma.
    """
    d = mu.shape[0]
    for _ in range(50):  # Iterative approximation limit
        diff = 0
        for i in range(d - 1):
            l = i if i < k else i + 1
            mu, sigma, delta = update_factors(mu, sigma, k, l)
            diff += np.abs(delta)

        if diff < 1e-3:  # Convergence criterion
            break

    log_prob = compute_log_prob(mu, sigma, k)
    yield log_prob

def update_factors(mu, sigma, k, l):
    """
    Update mean and covariance factors for iterative approximation.

    Parameters:
        mu (np.ndarray): Mean values of the points.
        sigma (np.ndarray): Covariance matrix.
        k (int): Current index of interest.
        l (int): Loop index.

    Returns:
        tuple: Updated mean (mu), covariance (sigma), and delta value.
    """
    delta = (sigma[l, l] - 2 * sigma[k, l] + sigma[k, k]) / 2
    mu[l] -= delta  # Update mean based on delta
    return mu, sigma, delta

def compute_log_prob(mu, sigma, k):
    """
    Calculate the log probability for the k-th point.

    Parameters:
        mu (np.ndarray): Mean values of the points.
        sigma (np.ndarray): Covariance matrix.
        k (int): Index of the point being evaluated.

    Returns:
        float: Log probability of the k-th point.
    """
    return -0.5 * np.log(2 * np.pi * sigma[k, k]) - (mu[k] ** 2) / (2 * sigma[k, k])

if __name__ == "__main__":
    """
    Example usage to demonstrate how to call the `joint_min` function.
    """
    mu = np.array([0.5, 1.0, -0.5])  # Example mean values
    sigma = np.array([[1.0, 0.2, 0.1],  # Example covariance matrix
                      [0.2, 1.0, 0.3],
                      [0.1, 0.3, 1.0]])
    pmin = joint_min(mu, sigma)  # Compute probabilities of being minimum
    print("Probability of minimum:", np.exp(pmin))  # Convert log probabilities to probabilities
