import numpy as np
import scipy

from sampling_filter import sampled_fft, fourier_core, fourier_core_test


def test_scaled_sampler():
    sigmas = np.random.uniform(1, 5, 10)
    mus = np.random.uniform(2, 10, 10)
    n_samples = 100000
    scaled_samples = sampled_fft.sample_scaled_shifted_gaussians(
        n_samples, sigmas, mus, lambda x: np.random.normal(size=x))
    for i, (sigma, mu) in enumerate(zip(sigmas, mus)):
        this_samples = scaled_samples[:, i]
        a = np.mean(this_samples)
        b = np.std(this_samples)
        np.testing.assert_allclose(a, mu, rtol=0.05)
        np.testing.assert_allclose(b, sigma, rtol=0.05)


def test_sampled_idft_no_randomness():
    grid, random_real_signal = fourier_core_test._get_random_real_signal(5)
    random_real_signal_coeffs = fourier_core.fourier_series_coeffs(random_real_signal)
    custom_k_vals = np.random.choice(np.arange(start=1, stop=16, step=1), 10, replace=False)

    idft_fn = lambda coeffs, k_vals, output_grid: sampled_fft.sampled_partial_real_idft(
        coeffs, k_vals, output_grid, 1, lambda x: 0, lambda x: np.random.normal(size=x))

    custom_inv = fourier_core.partial_custom_real_idft(random_real_signal_coeffs, custom_k_vals,
                                                       idft_fn)
    np.testing.assert_allclose(custom_inv, random_real_signal, rtol=1e-8)


def test_sampled_convergence():
    grid, random_real_signal = fourier_core_test._get_random_real_signal(5)
    random_real_signal_coeffs = fourier_core.fourier_series_coeffs(random_real_signal)
    custom_k_vals = np.random.choice(np.arange(start=1, stop=16, step=1), 5, replace=False)

    idft_fn = lambda coeffs, k_vals, output_grid: sampled_fft.sampled_partial_real_idft(
        coeffs, k_vals, output_grid, 1000000, lambda x: 0.1, lambda x: np.random.normal(size=x))

    custom_inv = fourier_core.partial_custom_real_idft(random_real_signal_coeffs, custom_k_vals,
                                                       idft_fn)
    error = np.abs(custom_inv - random_real_signal)/np.max(np.abs(random_real_signal))
    assert np.max(error) < 0.005
