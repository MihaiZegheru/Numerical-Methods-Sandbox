import numpy as np
from eigen import eigen

def SVD_right_ext(A: np.matrix, n: int, m: int):

	eigen_values, eigen_vectors = eigen(np.transpose(A) * A)

	eigen_values = np.transpose(np.matrix(eigen_values))
	singular_values = np.sqrt(eigen_values[0:n])


	S = np.zeros((n, n))

	np.fill_diagonal(S, singular_values)

	Vh = np.concatenate(eigen_vectors[0:n], axis=1)


	U = A * Vh
	for i in range(0, n):
		U[0:m, i] /= singular_values[i]


	Vh = np.transpose(np.concatenate(eigen_vectors[0:n], axis=1))

	return 0

def SVD_left_ext(A: np.matrix, n: int, m: int):

	eigen_values, eigen_vectors = eigen(np.transpose(A) * A)

	eigen_values = np.transpose(np.matrix(eigen_values))
	singular_values = np.sqrt(eigen_values[0:m])
	print(singular_values)

	S = np.zeros((m, m))

	np.fill_diagonal(S, singular_values)

	Vh = np.concatenate(eigen_vectors[0:m], axis=1)
	print(Vh)

	U = A * Vh
	for i in range(0, m):
		U[0:n, i] /= singular_values[i]
	print(U)
	print()

	Vh = np.transpose(np.concatenate(eigen_vectors[0:m], axis=1))

	return 0

def SVD():
	(n, m) = np.shape(A)

	if n < m:
		return SVD_right_ext(A, n, m)
	else:
		return SVD_left_ext(A, n, m)

A = np.matrix('1 2 4; 2 3 1')
SVD()