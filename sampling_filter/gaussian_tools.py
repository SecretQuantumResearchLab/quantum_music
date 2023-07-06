import numpy as np


def sample_scaled_shifted_gaussians(n_samples_per, sigmas, mus, standard_gauss_sampler):
    n_sigmas = len(sigmas)
    samples = standard_gauss_sampler(n_sigmas * n_samples_per).reshape(
        (-1, n_sigmas))
    return np.einsum("bd, d -> bd", samples, sigmas) + mus
