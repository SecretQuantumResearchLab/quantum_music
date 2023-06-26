import numpy as np


def sample_scaled_shifted_gaussians(n_samples_per, sigmas, mus, standard_gauss_sampler):
    n_sigmas = len(sigmas)
    n_mus = len(mus)
    samples = standard_gauss_sampler(n_sigmas * n_mus * n_samples_per).reshape(
        (-1, n_sigmas, n_mus))
    return np.einsum("bda, d -> bda", samples + mus, sigmas)
