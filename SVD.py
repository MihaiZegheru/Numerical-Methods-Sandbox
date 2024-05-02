import numpy as np
from eigen import eigen

def SVD_right_ext(A: np.matrix, n: int, m: int, r: int):
	eigen_values, eigen_vectors = eigen(np.transpose(A) * A, r)
	S = np.sqrt(eigen_values[0:r])
	Vh = np.concatenate(eigen_vectors[0:r], axis=1)

	U = A * Vh
	for i in range(0, r):
		U[0:m, i] /= S[i]
	Vh = np.transpose(np.concatenate(eigen_vectors[0:r], axis=1))

	return (U, S, Vh)

def SVD_left_ext(A: np.matrix, n: int, m: int, r: int):
	eigen_values, eigen_vectors = eigen(np.transpose(A) * A, r)
	S = np.sqrt(eigen_values[0:r])
	Vh = np.concatenate(eigen_vectors[0:r], axis=1)

	U = A * Vh
	for i in range(0, r):
		U[0:n, i] /= S[i]
	Vh = np.transpose(np.concatenate(eigen_vectors[0:r], axis=1))

	return (U, S, Vh)

def SVD(A: np.matrix, r=0):
	(n, m) = np.shape(A)

	if (r == 0):
		r = min(n, m)

	if n < m:
		return SVD_right_ext(A, n, m, r)
	else:
		return SVD_left_ext(A, n, m, r)
