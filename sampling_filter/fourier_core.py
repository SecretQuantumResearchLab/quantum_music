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
    break_point = int(len(fourier_coeffs)/2)
    zero_coeff = fourier_coeffs[..., 0]
    positive_coeffs = fourier_coeffs[..., 1:break_point]
    nyquist_term = fourier_coeffs[..., break_point]
    negative_coeffs = fourier_coeffs[..., break_point + 1:][..., ::-1]
    return zero_coeff, positive_coeffs, nyquist_term, negative_coeffs


def pack_coeffs(zero_coeff, positive_coeffs, nyquist_term, negative_coeffs):
    return np.concatenate([np.atleast_1d(zero_coeff), positive_coeffs,
                           np.atleast_1d(nyquist_term), negative_coeffs[..., ::-1]], axis=-1)


def partial_custom_real_idft(all_fft_coeffs, custom_k_vals, custom_idft_fn):
    reindex_custom_k = custom_k_vals - 1
    zero_term, positive_terms, nyquist_term, negative_terms = unpack_coeffs(all_fft_coeffs)
    this_fourier_grid = fourier_grid(int(np.log2(all_fft_coeffs.shape[-1])))
    custom_part = custom_idft_fn(positive_terms[..., reindex_custom_k], custom_k_vals, this_fourier_grid)
    zeroed_positive = np.copy(positive_terms)
    zeroed_positive[..., reindex_custom_k] = 0
    fft_part = np.fft.ifft(pack_coeffs(zero_term, zeroed_positive, nyquist_term, np.conjugate(zeroed_positive)), norm="forward")
    return np.real_if_close(fft_part + custom_part)
