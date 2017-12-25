import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random
from sklearn.cluster import KMeans


x = np.array([[10,10],[1,2],[1.5,1.8],[5,8],[8,8],[1, 0.6],[9,11],[12,15],[7,4],[-2,-2],[-5,-10],[15,20],[5,-10],[3,-6],[0,9],[-5,20]])

#print(x[:,1])
# plt.scatter(x[:,0], x[:,1], s=150)
# plt.show()

colors = 10*['g','r','c','b','k']


class K_Means:

    def __init__(self, k=4, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        # empty list for centroids
        self.centroids = {}

        # create unique random K range numbers for initial centroid selection
        initCentroidSpots = random.sample(range(0,len(data)), self.k)
        #print(initCentroidSpots)

        for i in range(self.k):
            # create k number of initial centroids
            self.centroids[i] = data[initCentroidSpots[i]]

        for i in range(self.max_iter):

            # each iteration, we create the classification
            self.classifications = {}

            for i in range(self.k):

                # create k number of classes
                self.classifications[i] = []

            for featureset in data:

                # each data's distance to the K number of centroids
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # see which centroid it was closes to by taking the index of the distance (because centroids in order)
                classification = distances.index(min(distances))
                # then append the classification (0-k) with the feature set
                #print(self.classifications)
                self.classifications[classification].append(featureset)

            # creates a copy instead of linked
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # np.average calculates the average centroid of all feature sets?
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)


            # guilty until proven innocent
            optimized = True
            # for all centroids
            for c in self.centroids:
                # the original c spot centroid
                original_centroid = prev_centroids[c]
                # the new centroid after each iter
                current_centroid = self.centroids[c]
                # if the centroid moved less than self.tol percentage then it is as good as it gets
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            # if it is good, then we end iterations
            if optimized:
                break


    def predict(self, data):

        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classsification in clf.classifications:
    color = colors[classsification]
    for featureset in clf.classifications[classsification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

unknowns = np.array([[5,5],[2,4],[3,3.5],[-5,-8],[-8,-8],[-1, -0.6],[12,15],[12,-15], [7,14]])

for unknown in unknowns:
    classsifi = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classsifi], s=150, linewidths=5)

plt.show()
























