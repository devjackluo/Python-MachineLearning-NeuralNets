import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.datasets import make_blobs


data_dict = {1:[[2,0],[2,2],[2,4],[2,-2]],
             2:[[7,3],[4,2],[8,4]],
             3: [[4, 12], [5, 16], [6, 18]],
             4: [[1, 12], [0, 16], [-2, 18]],
             5: [[-4, 0], [-5, 2], [-6, 4], [-7, -2]]}

X = []
y = []

for i in data_dict:
    for n in data_dict[i]:
        X.append(n)
        y.append(i)

X = np.array(X)
y = np.array(y)



# we create 40 separable points
#X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# print(X)
# print(y)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000, decision_function_shape='ovo')
clf.fit(X, y)


#print(clf.score([[2,10]],[3]))

def checkClass(x):
    print('class:', x, 'is', clf.score([[-2, 4]], [x]))

plt.scatter(-2, 4, color='k', s=100)

for i in data_dict:
    checkClass(i)



plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
#
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# print(xy)
# Z = clf.decision_function(xy) #.reshape(XX.shape)
#
# # plot decision boundary and margins
# # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
# #            linestyles=['--', '-', '--'])
# #plot support vectors
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none')

plt.show()