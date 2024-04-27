import numpy as np

def MPD(A: np.matrix, x0: np.matrix, tol: float, max_steps: int):
	x = x0 / np.linalg.norm(x0)
	lmb = float(np.transpose(x) * A * x)
	steps = 0

	while np.linalg.norm(A * x - lmb * x) > tol and steps < max_steps:
		steps += 1
		# print(lmb)
		x = A * x
		x /= np.linalg.norm(x)
		lmb = float(np.transpose(x) * A * x)


	return (lmb, x)
