import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans


x = np.array([[10,10],[1,2],[1.5,1.8],[5,8],[8,8],[1, 0.6],[9,11],[12,15], [7,4]])

# print(x[:,1])
# plt.scatter(x[:,0], x[:,1], s=150)
# plt.show()

clf = KMeans(n_clusters=4)
clf.fit(x)

centroids = clf.cluster_centers_
# labels is the clf's fitted data's class with respect to their cluster #
print(clf.labels_)
labels = clf.labels_
print(set(labels))

colors = 10*['g.','r.','c.','b.','k.']

for i in range(len(x)):
    plt.plot(x[i][0], x[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()





