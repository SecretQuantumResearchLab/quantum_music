import numpy as np
import scipy

from sampling_filter import fourier_core


def _complex_quadrature(func, a, b, **kwargs):

    real_integral = scipy.integrate.quad(lambda x: np.real(func(x)), a, b, **kwargs)
    imag_integral = scipy.integrate.quad(lambda x: np.imag(func(x)), a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]


def test_fourier_series_coeffs_1d():
    test_f = lambda t: np.exp(np.cos(10 * t) ** 4 - np.sin(2 * t))
    fourier_integrand = lambda t, k, f: 1/(2 * np.pi) * np.exp(-1j * k * t) * f(t)
    t_vals = fourier_core.fourier_grid(10)
    f_vals = test_f(t_vals)
    fft_coeffs = fourier_core.fourier_series_coeffs(f_vals)
    all_k_vals = fourier_core.get_k_vals(2**10)
    k_vals = all_k_vals[:50]
    exact_coeffs = []
    for k in k_vals:
        exact_coeff = _complex_quadrature(lambda t: fourier_integrand(t, k, test_f), 0, 2 * np.pi)
        exact_coeffs.append(exact_coeff)
    np.testing.assert_allclose(fft_coeffs[:len(k_vals)], exact_coeffs, atol=1e-11)


def _get_random_real_signal(log_length):
    grid = fourier_core.fourier_grid(log_length)
    random_real_signal = np.random.uniform(-1, 1, len(grid))
    return grid, random_real_signal


def test_orginization():
    grid, random_real_signal = _get_random_real_signal(5)
    random_real_signal = random_real_signal - np.mean(random_real_signal) + 1
    random_real_signal_coeffs = fourier_core.fourier_series_coeffs(random_real_signal)
    zero_coeff, positive_coeffs, negative_coeff = fourier_core.organize_coeffs(random_real_signal_coeffs)
    np.testing.assert_allclose(zero_coeff, 1, rtol=1e-8)
    np.testing.assert_allclose(positive_coeffs, np.conjugate(negative_coeff), rtol=1e-8)


