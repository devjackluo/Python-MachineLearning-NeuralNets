import numpy as np

a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.dot(a, b))

a = [2,2,2,2,2,2]
b = [2,2,2,2,2,2]
print(np.dot(a, b))

# a = np.arange(3*4*5*6).reshape((3,4,5,6))
# b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
# print(sum(a[2,3,2,:] * b[1,2,:,2]))


a = [[[2,2], [2,2]], [[2,2], [2,2]]]
b = [[[2,2], [2,2]], [[2,2], [2,2]]]
print(np.dot(a, b))