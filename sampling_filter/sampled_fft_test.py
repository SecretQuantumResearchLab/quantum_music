import numpy as np
import scipy

from sampling_filter import sampled_fft


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


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = scipy.integrate.quad(real_func, a, b, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def test_series_coeffs():
    N = 2**10
    test_fn = lambda x: np.cos(x) * np.exp(x)
    grid = np.arange(start=0, stop=N, step=1) * 2 * np.pi/N
    fn_on_grid = test_fn(grid)
    coeffs_fast = np.fft.rfft(fn_on_grid, norm="forward")
    for k in range(int(N/2+1)):
        integrand = lambda t: np.exp(-1j * k * t) * test_fn(t)
        real_coeff = 1/(2 * np.pi) * complex_quadrature(integrand, 0, 2 * np.pi)[0]


def test_cosine_trns():
    N = 128
    #signal_t = np.random.uniform(-1, 1, N)
    #signal_t = signal_t - np.mean(signal_t)
    signal_k = np.random.uniform(-1, 1, int(N/2 + 1)) + 1j * np.random.uniform(-1, 1, int(N/2 + 1))
    signal_k[0] = 0
    #signal_k[0] = 0
    #signal_k = np.zeros(int(N/2 + 1)).astype(np.complex128)
    #signal_k[5] = np.exp(-1j * np.pi/7)
    #signal_k[31] = 0.21 * np.exp(-1j * np.pi / 9)
    #signal_k[-5] = 10 * np.exp(-1j * np.pi / 3)
    signal_t = np.fft.irfft(signal_k, norm="forward")
    vals_m = np.arange(start=0, stop=N, step=1)
    vals_k = np.arange(start=1, stop=N/2+1, step=1)
    nonzero_coeffs = signal_k[1:]
    abs_nz = np.abs(nonzero_coeffs)
    angle_nz = np.angle(nonzero_coeffs)
    mk_grid = np.einsum("m, k ->mk", vals_m, vals_k)
    inv_trns = signal_k[0] + 2 * np.einsum("k, mk -> m", abs_nz, np.cos((2 * (np.pi/N) * mk_grid) + angle_nz))
    a = inv_trns/signal_t
    print("")


def test_sampled_fft_no_var():
    N = 256
    signal_t = np.random.uniform(-1, 1, N)
    signal_t = signal_t - np.mean(signal_t)
    signal_k = np.fft.rfft(signal_t, norm="forward")
    #signal_k[3] = np.exp(-1j * np.pi/4)
    #signal_k[4] = 0.5 * np.exp(-1j * np.pi / 8)
    #signal_k[20] = 0.22 * np.exp(-1j * 0.76)
    #signal_k[-2] = 0.123123 * np.exp(-1j * 2.2)
    signal_t = np.fft.irfft(signal_k, norm="forward")
    samp_ver = sampled_fft.partial_sampled_ifft(signal_k, 128, 1, 0, lambda x: np.random.normal(size=x))
    a = samp_ver/signal_t
    print("")

test_cosine_trns()
