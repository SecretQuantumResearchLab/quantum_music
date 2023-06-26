import numpy as np


def sample_scaled_shifted_gaussians(sigmas, mus, standard_gauss_sampler):
    n_samples = len(sigmas) * len(mus)
    samples = standard_gauss_sampler(n_samples).reshape((len(sigmas), len(mus), -1))
    mus = mus.
