import numpy as np
import matplotlib.pyplot as plt

from sliced import SlicedInverseRegression
from sliced import datasets

X, y = datasets.make_cubic(random_state=123)
print(X.shape, y.shape, len(np.unique(y)))

sir = SlicedInverseRegression(n_directions='auto', n_slices=len(np.unique(y)))
# X_sir = sir.fit_transform(X, y)
sir.fit(X, y)
print()
X_sir = sir.transform(X)
print(X_sir.shape, y.shape)
# estimate of the first dimension reducing directions
beta1_hat = sir.directions_[0, :]

# plot data projected onto the first direction
plt.scatter(X_sir[:, 0], y, c=y, cmap='viridis', linewidth=0.5, edgecolor='k')
plt.xlabel("$X\hat{\\beta_1}$")
plt.ylabel("y")

# annotation showing the direction found
beta_text = "$\\beta_1$ = " + "{0}".format([0.707, 0.707])
plt.annotate(beta_text, xy=(-2, 6.5))
beta1_hat_text = "$\hat{\\beta_1}$ = " + "{0}".format(
	np.round(beta1_hat, 3).tolist()[:2])
plt.annotate(beta1_hat_text, xy=(-2, 7.5))

plt.show()
