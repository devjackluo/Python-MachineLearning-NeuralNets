import numpy as np
import sys
import random

a = [[1, -50], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.dot(a, b))

print(abs(max(a[0], key=abs)))

a = [2,2,2,2,2,2]
b = [2,2,2,2,2,2]
print(np.dot(a, b))
print(np.inner(a,b))
print("fsdsdsdsdsdsdsdsd")

# a = np.arange(3*4*5*6).reshape((3,4,5,6))
# b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
# print(sum(a[2,3,2,:] * b[1,2,:,2]))


a = [[[2,2], [2,2]], [[2,2], [2,2]]]
b = [[[2,2], [2,2]], [[2,2], [2,2]]]
print(np.dot(a, b))



print("#########")

def mulN(x,g,h):
    return (x**2)*(g+h)

listx = [1,2,3,4,5,6]
listg = [1,2,3,4,5,6]
listh = [1,2,3,4,5,6]
lamlist = list(map(lambda x,g,h: (x**2)*(g+h), listx, listg, listh))
newlist = list(map(mulN, listx, listg, listh))
print(newlist)
print(lamlist)


print(sys.maxsize)
# highest it can really go on my computer
# print(100**100000)


for i in range(100):
    print(random.randint(0, 2))

