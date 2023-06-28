import numpy as np


def sample_scaled_shifted_gaussians(n_samples_per, sigmas, mus, standard_gauss_sampler):
    n_sigmas = len(sigmas)
    samples = standard_gauss_sampler(n_sigmas * n_samples_per).reshape(
        (-1, n_sigmas))
    return np.einsum("bd, d -> bd", samples, sigmas) + mus


def sampled_partial_real_idft(coeffs, k_vals, output_grid, n_samples, sigma_vs_k, standard_gauss_sampler):
    abs_coeffs = np.abs(coeffs)
    angle_coeffs = np.angle(coeffs)
    mk_grid = np.einsum("m, k -> mk", output_grid, k_vals)
    cos_term = np.cos(mk_grid + angle_coeffs)
    means = np.einsum("mk, k -> mk", cos_term, abs_coeffs)
    sigmas = sigma_vs_k(k_vals)
    sigmas = np.broadcast_to(sigmas, means.shape)
    samples = sample_scaled_shifted_gaussians(n_samples, sigmas.flatten(), means.flatten(), standard_gauss_sampler).reshape((-1, *means.shape))
    return 2 * np.sum(np.mean(samples, axis=0), axis=-1)
