
import numpy

x = numpy.ones((1,10,4))
print(x)
x = x.reshape([-1,4,1])
print(x)


y = [100,2,8,9,1,1,1,2,1,1,1,3,3,3,6,6,6]
max = numpy.argmax(y)
print(max)


a = [[1,2],[3,4],[5,6],[6,7]]
print(a)
print(len(a))
print(a[2])
a = numpy.array(a)
print(a)
print(len(a))
print(a[2])


# userinput = input('To Save Model? (Y/N) : ').lower()
# if userinput == 'y':
#     nameInput = input('Model Name: ').lower()
#     print(nameInput + ".model")
