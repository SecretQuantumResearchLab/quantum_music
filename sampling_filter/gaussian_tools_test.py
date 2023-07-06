import numpy as np

from sampling_filter import gaussian_tools


def test_scaled_sampler():
    sigmas = np.random.uniform(1, 5, 10)
    mus = np.random.uniform(2, 10, 10)
    n_samples = 100000
    scaled_samples = gaussian_tools.sample_scaled_shifted_gaussians(
        n_samples, sigmas, mus, lambda x: np.random.normal(size=x))
    for i, (sigma, mu) in enumerate(zip(sigmas, mus)):
        this_samples = scaled_samples[:, i]
        a = np.mean(this_samples)
        b = np.std(this_samples)
        np.testing.assert_allclose(a, mu, rtol=0.05)
        np.testing.assert_allclose(b, sigma, rtol=0.05)


