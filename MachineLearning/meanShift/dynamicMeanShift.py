import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random

#x = np.array([[12, 12], [1, 2], [1.5, 1.8], [6, 2], [8, 4], [1, 0.6], [11, 13], [12, 15], [7, 4]])

#x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])

centers = random.randrange(2,8)
print(centers)
x, y = make_blobs(n_samples=30, centers=3, n_features=2)

# plt.scatter(x[:,0], x[:,1], s=150)
# plt.show()

colors = 10 * ['r', 'g', 'b', 'c', 'k', 'y', 'm']


class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        # get radius of entire data and set radius to 1/100 of it
        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step


        # create centroid at every point of data
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        # create list of # of weights reversed 99 to 0?
        weights = [i for i in range(self.radius_norm_step)][::-1]

        # until centroids don't move
        while True:

            new_centroids = []
            # for the current centroids
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                # get all featureset's dynamic value towards the centroid
                for featureset in data:

                    # feature set distance to centroid
                    distance = np.linalg.norm(featureset - centroid)
                    # if it is on it, then set it to something small
                    if distance == 0:
                        distance = 0.000000001
                    # calculate where on the weights it is located
                    weight_index = int(distance / self.radius)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    # add to in band (which is everythhing in dynamic)
                    to_add = (weights[weight_index] ** 2) * [featureset]
                    in_bandwidth += to_add


                # calculate the center of all the contained featuresets to get new centroid
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # get unique new centroids
            uniques = sorted(list(set(new_centroids)))


            # remove centroids that are pretty close to each other
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius/2:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass





            # previous c
            prev_centroids = dict(centroids)
            #print(prev_centroids)

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

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = Mean_Shift()
clf.fit(x)

centroids = clf.centroids


for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)


for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()








