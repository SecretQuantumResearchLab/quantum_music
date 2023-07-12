import numpy as np


def fourier_grid(k):
    grid = np.linspace(start=0, stop=2 * np.pi, num=2 ** k + 1)
    return grid[:-1]


def get_k_vals(n):
    return np.fft.fftfreq(n, 1/n).astype(int)


def fourier_series_coeffs(x, axes_from=None):
    axes = None
    if axes_from is not None:
        axes = list(range(len(x.shape)))[axes_from:]
    return np.fft.fftn(x, norm="forward", axes=axes)


def unpack_coeffs(fourier_coeffs):
    break_point = int(fourier_coeffs.shape[-1]/2)
    zero_coeff = fourier_coeffs[..., 0]
    positive_coeffs = fourier_coeffs[..., 1:break_point]
    nyquist_term = fourier_coeffs[..., break_point]
    negative_coeffs = fourier_coeffs[..., break_point + 1:][..., ::-1]
    return zero_coeff, positive_coeffs, nyquist_term, negative_coeffs


def pack_coeffs(zero_coeff, positive_coeffs, nyquist_term, negative_coeffs):
    return np.concatenate([np.atleast_1d(zero_coeff), positive_coeffs,
                           np.atleast_1d(nyquist_term), negative_coeffs[..., ::-1]], axis=-1)


