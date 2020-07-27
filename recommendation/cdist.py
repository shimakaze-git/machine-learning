import numpy as np
from scipy.spatial.distance import cdist

x_list = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1]
]
X = np.array(x_list)

XA = X[:1]
XB = X

print(X)
print(XA)
print(XB)
print(1 - cdist(XA, XB, 'cosine'))

# X = np.random.randint(-5, 6, size=(5, 2))
