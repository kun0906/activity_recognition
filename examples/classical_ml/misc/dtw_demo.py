from sklearn.model_selection import train_test_split

X = [1 for i in range(50)]
Y = [1 * 2 for i in range(50)]
Z = [1 * 3 for i in range(50)]
data = train_test_split((X, Y, Z))

import numpy as np

## A noisy sine wave as query
idx = np.linspace(0, 6.28, num=100)
query = np.sin(idx) + np.random.uniform(size=100) / 10.0

## A cosine is for template; sin and cos are offset by 25 samples
idx = np.linspace(0, 6.28, num=50)
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *

alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")
#
# ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
# dtw(query, template, keep_internals=True,
#     step_pattern=rabinerJuangStepPattern(6, "c"))\
#     .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6, "c"))
rabinerJuangStepPattern(6, "c").plot()
