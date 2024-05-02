import numpy as np

def MPD(A: np.matrix, x0: np.matrix, tol: float, max_steps: int):
	x = x0
	lmb = 0
	steps = 0

	while np.linalg.norm(A * x - lmb * x) > tol and steps < max_steps:
		steps += 1

		x = A * x
		x /= np.linalg.norm(x)
		lmb = (np.transpose(x) * A * x)[0, 0]

	return lmb, x
