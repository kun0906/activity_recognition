"""
https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python

"""
import numpy as np

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# x1 = x.reshape((-1, 2, 3))
# print(x1)

from ar.features.feature import _get_fft

x = np.array([-1, -5.2354542542, -1.034, 0, 1, 2, 1, 0, 2, 3, 4])
n = 5
w1 = _get_fft(x, fft_bin=n)  # Direct current: sum(x[:n])
print(w1)
print('\n')
w = np.fft.fft(x, n)
print(np.abs(w), np.sum(x[:n]), np.abs(w) - w1)
freqs = np.fft.fftfreq(len(x))

print(w, freqs)
for coef, freq in zip(w, freqs):
    if coef:
        print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef, f=freq))

# (8+0j) * exp(2 pi i t * 0.0)
#    -4j * exp(2 pi i t * 0.25)
#     4j * exp(2 pi i t * -0.25)
