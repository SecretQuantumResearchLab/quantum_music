import numpy as np


def sample_scaled_shifted_gaussians(n_samples_per, sigmas, mus, standard_gauss_sampler):
    n_sigmas = len(sigmas)
    samples = standard_gauss_sampler(n_sigmas * n_samples_per).reshape(
        (-1, n_sigmas))
    return np.einsum("bd, d -> bd", samples, sigmas) + mus


def partial_sampled_ifft(signal_positive_coeffs, n_largest_modes_to_sample, n_samples,
                         sigma, standard_gauss_sampler):
    N = 2 * (len(signal_positive_coeffs) - 1)
    k_sample = np.argsort(np.abs(signal_positive_coeffs))[::-1][:n_largest_modes_to_sample]
    coeffs_sample = signal_positive_coeffs[k_sample]
    mask = np.ones(signal_positive_coeffs.shape, dtype=np.bool)
    mask[k_sample] = 0
    fft_part = np.fft.irfft(mask * signal_positive_coeffs, norm="forward")
    m_vals = np.arange(start=0, stop=N, step=1)
    phases = np.angle(coeffs_sample)
    meds = np.einsum("mk, k -> mk", np.cos((2 * (np.pi/N) * np.einsum("k, m -> mk", k_sample, m_vals)) - phases), np.abs(coeffs_sample))
    meds = meds.flatten()
    sampled = np.mean(sample_scaled_shifted_gaussians(n_samples, sigma * np.ones(len(meds)), meds, standard_gauss_sampler).reshape((n_samples, N, n_largest_modes_to_sample)), axis=0)

    return fft_part + 2 * np.sum(sampled, axis=-1)

