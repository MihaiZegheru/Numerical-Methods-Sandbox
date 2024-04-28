import numpy as np
from eigen import eigen

def SVD_right_ext(A: np.matrix, n: int, m: int):
	eigen_values, eigen_vectors = eigen(np.transpose(A) * A)
	S = np.sqrt(eigen_values[0:n])
	Vh = np.concatenate(eigen_vectors[0:n], axis=1)

	U = A * Vh
	for i in range(0, n):
		U[0:m, i] /= S[i]
	Vh = np.transpose(np.concatenate(eigen_vectors[0:n], axis=1))

	return (U, S, Vh)

def SVD_left_ext(A: np.matrix, n: int, m: int):
	eigen_values, eigen_vectors = eigen(np.transpose(A) * A)
	S = np.sqrt(eigen_values[0:m])
	Vh = np.concatenate(eigen_vectors[0:m], axis=1)

	U = A * Vh
	for i in range(0, m):
		U[0:n, i] /= S[i]
	Vh = np.transpose(np.concatenate(eigen_vectors[0:m], axis=1))

	return (U, S, Vh)

def SVD(A: np.matrix, r: int):
	(n, m) = np.shape(A)
	if n < m:
		return SVD_right_ext(A, n, m)
	else:
		return SVD_left_ext(A, n, m)
