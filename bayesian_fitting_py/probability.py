"""
Probability distribution functions for Bayesian inference.

This module contains functions for calculating probability distributions
and combining them in a Bayesian framework.

Translated from MATLAB VBR vbr/fitting/*.m
"""

import numpy as np
from typing import List, Union, Optional

ArrayLike = Union[np.ndarray, float]


def probability_normal(x: ArrayLike, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the probability of having an observed value x given
    a normal distribution with mean mu and standard deviation sigma.

    Parameters
    ----------
    x : array-like
        Observed value(s)
    mu : float
        Mean value of distribution
    sigma : float
        Standard deviation of the distribution

    Returns
    -------
    np.ndarray
        Probability for each of the values in x
    """
    x = np.asarray(x)
    normal_pdf = (2 * np.pi * sigma**2) ** (-0.5) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )
    return normal_pdf


def probability_uniform(
    x: ArrayLike, min_val: float, max_val: float
) -> np.ndarray:
    """
    Calculate the probability of having an observed value x given
    a uniform distribution between min_val and max_val.

    Parameters
    ----------
    x : array-like
        Observed value(s)
    min_val : float
        Minimum for uniform distribution
    max_val : float
        Maximum for uniform distribution

    Returns
    -------
    np.ndarray
        Probability for each of the values in x
    """
    x = np.asarray(x)
    uniform_pdf = np.ones(x.shape) / (max_val - min_val)
    return uniform_pdf


def probability_lognormal(x: ArrayLike, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the probability of having an observed value x given
    a log normal distribution with mean mu and standard deviation sigma.

    Parameters
    ----------
    x : array-like
        Observed value(s), must be dimensionless and > 0
    mu : float
        Mean value of distribution in log-space
    sigma : float
        Standard deviation of the distribution in log-space

    Returns
    -------
    np.ndarray
        Probability for each of the values in x
    """
    x = np.asarray(x)
    denom = x * sigma * np.sqrt(2 * np.pi)
    lognormal_pdf = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / denom
    return lognormal_pdf


def likelihood_from_residuals(
    obs_val: float, obs_std: float, predicted_vals: np.ndarray
) -> np.ndarray:
    """
    Calculate the likelihood (pdf) of the observed value at each of the given
    combination of state variables by comparing the observed value to the
    calculated value, scaled by the observed standard deviation.

    The likelihood p(D|A), e.g., P(Vs | T, phi, gs), is calculated using
    the residual:
        p(D|A) = 1 / sqrt(2 * pi * sigma^2) * exp(-0.5 * chi-square)
    where chi-square = sum((x_obs - x_preds)^2 / sigma^2)

    Parameters
    ----------
    obs_val : float
        Observed (seismic) property
    obs_std : float
        Standard deviation on the observed value
    predicted_vals : np.ndarray
        Matrix of calculated values of the observed property at each of
        the different parameter sweep combinations

    Returns
    -------
    np.ndarray
        Likelihood for each of the proposed parameter combinations
    """
    chi_squared = (predicted_vals - obs_val) ** 2 / (obs_std**2)
    likelihood = (2 * np.pi * obs_std**2) ** (-0.5) * np.exp(-0.5 * chi_squared)
    return likelihood


def joint_independent_probability(marginals: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the joint probability (assuming independent) of two or more
    marginal probabilities, {p(A), p(B), ...}.

    As we are assuming all of these are independent, this is a simple product:
        p(A, B, ..., N) = p(A) * p(B) * ... * p(N)

    Parameters
    ----------
    marginals : list of np.ndarray
        List of marginal probabilities. All must be the same size.

    Returns
    -------
    np.ndarray
        Joint independent probability, p(A, B, ...)
    """
    joint_pdf = np.ones(marginals[0].shape)
    for marginal in marginals:
        joint_pdf = joint_pdf * marginal
    return joint_pdf


def conditional_bayes(
    p_b_given_a: ArrayLike, p_a: ArrayLike, p_b: ArrayLike
) -> np.ndarray:
    """
    Calculate the conditional probability using Bayes' Theorem:
        p(A | B) = p(B | A) * p(A) / p(B)

    Parameters
    ----------
    p_b_given_a : array-like
        Probability of B given A (likelihood)
    p_a : array-like
        Prior probability of A
    p_b : array-like
        Prior probability of B

    Returns
    -------
    np.ndarray
        Posterior probability of A given B
    """
    p_b_given_a = np.asarray(p_b_given_a)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    
    p_a_given_b = p_b_given_a * p_a / p_b
    return p_a_given_b


def conditionally_independent_c_given_ab(
    p_a_given_c: ArrayLike,
    p_b_given_c: ArrayLike,
    p_c: ArrayLike,
    p_a_and_b: ArrayLike,
) -> np.ndarray:
    """
    Calculate the conditional probability of C given A and B, assuming that
    A and B are dependent but conditionally independent given C.

        P(C | A, B) = P(A | C) P(B | C) P(C) / P(A, B)

    Parameters
    ----------
    p_a_given_c : array-like
        Conditional probability of A given C
    p_b_given_c : array-like
        Conditional probability of B given C
    p_c : array-like
        Prior probability of C
    p_a_and_b : array-like
        Joint probability of A and B

    Returns
    -------
    np.ndarray
        Conditional probability of C given both A and B
    """
    p_a_given_c = np.asarray(p_a_given_c)
    p_b_given_c = np.asarray(p_b_given_c)
    p_c = np.asarray(p_c)
    p_a_and_b = np.asarray(p_a_and_b)
    
    p_c_given_ab = p_a_given_c * p_b_given_c * p_c / p_a_and_b
    return p_c_given_ab


def probability_distributions(distribution_flag: str, *args) -> np.ndarray:
    """
    Main dispatcher function for probability calculations.

    Parameters
    ----------
    distribution_flag : str
        Type of calculation to perform. Options:
        - 'normal': Normal distribution PDF
        - 'uniform': Uniform distribution PDF
        - 'lognormal': Log-normal distribution PDF
        - 'likelihood from residuals': Calculate likelihood from obs/pred
        - 'joint independent': Joint probability of independent variables
        - 'A|B': Conditional probability via Bayes' theorem
        - 'C|A,B conditionally independent': Conditional probability with
          conditionally independent A and B given C
    *args : variable
        Arguments depend on distribution_flag

    Returns
    -------
    np.ndarray
        Calculated probability distribution
    """
    if distribution_flag == "uniform":
        x, min_val, max_val = args
        return probability_uniform(x, min_val, max_val)
    
    elif distribution_flag == "normal":
        x, mu, sigma = args
        return probability_normal(x, mu, sigma)
    
    elif distribution_flag == "lognormal":
        x, mu, sigma = args
        return probability_lognormal(x, mu, sigma)
    
    elif distribution_flag == "likelihood from residuals":
        obs_val, obs_std, predicted = args
        return likelihood_from_residuals(obs_val, obs_std, predicted)
    
    elif distribution_flag == "joint independent":
        marginals = args[0]
        return joint_independent_probability(marginals)
    
    elif distribution_flag == "A|B":
        likelihood_b_given_a, prior_a, prior_b = args
        return conditional_bayes(likelihood_b_given_a, prior_a, prior_b)
    
    elif distribution_flag == "C|A,B conditionally independent":
        p_a_given_c, p_b_given_c, p_c, p_a_and_b = args
        return conditionally_independent_c_given_ab(
            p_a_given_c, p_b_given_c, p_c, p_a_and_b
        )
    
    else:
        raise ValueError(f"Invalid probability distribution: {distribution_flag}")
