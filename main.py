import numpy as np
from PIL import Image
from SVD import SVD
from eigen import eigen
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

image = Image.open('image.jpg')
A = np.asmatrix(image)
A = A[0:15, 0:10]
# A = np.matrix('1 23 3; 9 8 1')

with open('a.txt', 'w') as file:
	file.write(str(A))

# A = A.T * A
# print(A.T * A)

# print(np.linalg.cholesky(A.T * A))

print(eigen(np.transpose(A) * A))
print()
print(np.linalg.eig(np.transpose(A) * A))


(U, S, Vh) = SVD(A, 1)
# print(np.transpose(A) * A)
# print(np.linalg.eig(np.transpose(A) * A))

# (U, S, Vh) = np.linalg.svd(A)
E = np.zeros((len(S), len(S)))

np.fill_diagonal(E, S)


A_new = U[:, 0:125] * E * Vh
A_new = np.matrix(A_new)

# with open('b.txt', 'w') as file:
# 	file.write(str(A_new))

plt.imshow(A_new)
plt.show()
# new_image = Image.fromarray(U * S * Vh)
# new_image.save("new.png")
