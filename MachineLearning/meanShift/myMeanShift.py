import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np



x = np.array([[12,12],[1,2],[1.5,1.8],[6,2],[8,4],[1, 0.6],[11,13],[12,15], [7,4]])

# plt.scatter(x[:,0], x[:,1], s=150)
# plt.show()

colors = 10*['r','g','b','c','k','y','m']

class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius


    def fit(self, data):


        # create centroid at every point of data
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        # until centroids don't move
        while True:

            new_centroids = []
            # for the current centroids
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]



                # get all featuresets in the current centriod's radius
                for featureset in data:

                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)


                # calculate the center of all the contained featuresets to get new centroid
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # get unique new centroids
            uniques = sorted(list(set(new_centroids)))


            # previous c
            prev_centroids = dict(centroids)

            # empty out the centroids then populate with new ones
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            # until proven to move,
            optimized = True
            # for all our new centroids created, check if it was same as previous centroids
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break

            # if none move, then it is optimized
            if optimized:
                break

        # then populate the global centroids with new centroids
        self.centroids = centroids

    def predict(self, data):
        pass

clf = Mean_Shift()
clf.fit(x)

centroids = clf.centroids
print(centroids)
plt.scatter(x[:,0], x[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()








