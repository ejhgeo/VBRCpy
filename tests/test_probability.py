"""
Known-answer tests for the probability module.

Tests use analytical formulas to verify correctness — no golden files needed.
"""

import numpy as np
import pytest

from vbrcpy.probability import (
    probability_normal,
    probability_lognormal,
    probability_uniform,
    likelihood_from_residuals,
    conditional_bayes,
    conditionally_independent_c_given_ab,
    joint_independent_probability,
)

pytestmark = pytest.mark.fast


class TestProbabilityNormal:
    def test_value_at_mean(self):
        """PDF at x=mu should equal 1/(sigma*sqrt(2*pi))."""
        sigma = 2.0
        mu = 5.0
        expected = 1.0 / (sigma * np.sqrt(2 * np.pi))
        result = probability_normal(np.array([mu]), mu, sigma)
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_symmetry(self):
        """PDF should be symmetric about the mean."""
        mu, sigma = 3.0, 1.5
        x = np.array([mu - 1.0, mu + 1.0])
        result = probability_normal(x, mu, sigma)
        np.testing.assert_allclose(result[0], result[1], rtol=1e-12)

    def test_integral_approx_one(self):
        """Numerical integral over +/-5 sigma should be ~1."""
        mu, sigma = 0.0, 1.0
        x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 10000)
        pdf = probability_normal(x, mu, sigma)
        integral = np.trapezoid(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-6)

    def test_wider_sigma_lower_peak(self):
        """Wider sigma should produce lower peak."""
        mu = 0.0
        peak_narrow = probability_normal(np.array([mu]), mu, 1.0)[0]
        peak_wide = probability_normal(np.array([mu]), mu, 2.0)[0]
        assert peak_narrow > peak_wide


class TestProbabilityLognormal:
    def test_mode(self):
        """Mode of lognormal is at exp(mu - sigma^2)."""
        mu, sigma = 1.0, 0.5
        mode = np.exp(mu - sigma**2)
        x = np.linspace(0.1, 20.0, 10000)
        pdf = probability_lognormal(x, mu, sigma)
        peak_x = x[np.argmax(pdf)]
        np.testing.assert_allclose(peak_x, mode, atol=0.01)

    def test_integral_approx_one(self):
        """Numerical integral should be ~1."""
        mu, sigma = 0.0, 1.0
        x = np.linspace(0.001, 30.0, 50000)
        pdf = probability_lognormal(x, mu, sigma)
        integral = np.trapezoid(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-3)

    def test_positive_only(self):
        """All PDF values should be non-negative for x > 0."""
        mu, sigma = 0.5, 0.3
        x = np.linspace(0.01, 10.0, 1000)
        pdf = probability_lognormal(x, mu, sigma)
        assert np.all(pdf >= 0)


class TestProbabilityUniform:
    def test_constant_value(self):
        """Uniform PDF should be constant at 1/(max-min)."""
        min_val, max_val = 2.0, 5.0
        x = np.linspace(min_val, max_val, 100)
        pdf = probability_uniform(x, min_val, max_val)
        expected = 1.0 / (max_val - min_val)
        np.testing.assert_allclose(pdf, expected, rtol=1e-12)

    def test_integral_is_one(self):
        """Integral over [min, max] should be 1."""
        min_val, max_val = -3.0, 7.0
        x = np.linspace(min_val, max_val, 1000)
        pdf = probability_uniform(x, min_val, max_val)
        integral = np.trapezoid(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-6)

    def test_invalid_range_raises(self):
        """max_val <= min_val should raise ValueError."""
        with pytest.raises(ValueError):
            probability_uniform(np.array([1.0]), 5.0, 3.0)


class TestLikelihoodFromResiduals:
    def test_peak_at_observed(self):
        """Likelihood should peak when predicted == observed."""
        obs_val = 4.3
        obs_std = 0.05
        predicted = np.linspace(3.5, 5.0, 1000)
        likelihood = likelihood_from_residuals(obs_val, obs_std, predicted)
        peak_idx = np.argmax(likelihood)
        np.testing.assert_allclose(predicted[peak_idx], obs_val, atol=0.002)

    def test_gaussian_shape(self):
        """Likelihood should equal a normal PDF centered at obs_val."""
        obs_val = 80.0
        obs_std = 10.0
        predicted = np.array([70.0, 80.0, 90.0])
        likelihood = likelihood_from_residuals(obs_val, obs_std, predicted)
        expected = probability_normal(predicted, obs_val, obs_std)
        np.testing.assert_allclose(likelihood, expected, rtol=1e-12)

    def test_multidimensional(self):
        """Should work with multi-dimensional predicted arrays."""
        obs_val = 4.5
        obs_std = 0.1
        predicted = np.random.uniform(4.0, 5.0, (3, 4, 5))
        likelihood = likelihood_from_residuals(obs_val, obs_std, predicted)
        assert likelihood.shape == (3, 4, 5)
        assert np.all(likelihood >= 0)


class TestConditionalBayes:
    def test_basic_formula(self):
        """p(A|B) = p(B|A) * p(A) / p(B) with known values."""
        p_b_given_a = np.array([0.8])
        p_a = np.array([0.3])
        p_b = np.array([0.5])
        result = conditional_bayes(p_b_given_a, p_a, p_b)
        expected = 0.8 * 0.3 / 0.5
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_array_operation(self):
        """Should work element-wise on arrays."""
        p_b_given_a = np.array([0.9, 0.1, 0.5])
        p_a = np.array([0.2, 0.5, 0.3])
        p_b = np.array([0.4, 0.4, 0.4])
        result = conditional_bayes(p_b_given_a, p_a, p_b)
        expected = p_b_given_a * p_a / p_b
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestConditionallyIndependent:
    def test_formula(self):
        """P(C|A,B) = P(A|C) * P(B|C) * P(C) / P(A,B)."""
        shape = (5,)
        p_a_given_c = np.random.uniform(0.1, 0.9, shape)
        p_b_given_c = np.random.uniform(0.1, 0.9, shape)
        p_c = np.random.uniform(0.1, 0.5, shape)
        p_a_and_b = np.random.uniform(0.1, 0.9, shape)

        result = conditionally_independent_c_given_ab(
            p_a_given_c, p_b_given_c, p_c, p_a_and_b
        )
        expected = p_a_given_c * p_b_given_c * p_c / p_a_and_b
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestJointIndependent:
    def test_product_of_marginals(self):
        """Joint = product of independent marginals."""
        m1 = np.array([0.1, 0.2, 0.3])
        m2 = np.array([0.5, 0.5, 0.5])
        m3 = np.array([1.0, 2.0, 3.0])
        result = joint_independent_probability([m1, m2, m3])
        expected = m1 * m2 * m3
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_single_marginal(self):
        """Single marginal should be returned as-is."""
        m1 = np.array([0.3, 0.7])
        result = joint_independent_probability([m1])
        np.testing.assert_allclose(result, m1, rtol=1e-12)
