import numpy as np
from MPD import MPD

def eigen(A: np.matrix):
	(n, m) = np.shape(A)
	eigen_values = []
	eigen_vectors = []

	x0 = np.zeros(n)
	x0[0] = 1
	x0 = np.transpose(np.matrix(x0))

	for i in range(0, n):
		lmb, x = MPD(A, x0, 1e-10, 100)
		eigen_values.append(lmb)
		eigen_vectors.append(x)

		A = A - lmb * x * np.transpose(x)

	return eigen_values, eigen_vectors
