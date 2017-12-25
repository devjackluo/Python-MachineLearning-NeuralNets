import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

# there was a name for this!
# two dimension is fun, three is cool but four???!@!?? wtf is four dimensions... [1,2,3,4] to [5,6,7,8] .. gets worst..
# plot1 = [0,0,0]
# plot2 = [4,4,4]
# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 +(plot1[2]-plot2[2])**2)

#euclidean distance only works when you have a variable that defines/split the data into groups already
# dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
# new_features = [5,7]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))

            # euclidean_distance = sqrt(
            #     ((features[0] - predict[0]) ** 2) +
            #     ((features[1] - predict[1]) ** 2) +
            #     ((features[2] - predict[2]) ** 2) +
            #     ((features[3] - predict[3]) ** 2) +
            #     ((features[4] - predict[4]) ** 2) +
            #     ((features[5] - predict[5]) ** 2) +
            #     ((features[6] - predict[6]) ** 2) +
            #     ((features[7] - predict[7]) ** 2) +
            #     ((features[8] - predict[8]) ** 2)
            # )

            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(distances)
    # print("sorted euclid distance for all points towars the new point: ",sorted(distances))
    # print("first three votes in sorted distances: ",votes)

    # print("in votes, most common one was [('type', 'how often')] : ",Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    #print("vote result is most-common-ONE's first item in list's first item of tuple: ",vote_result)

    return vote_result, confidence


#get data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()


accuracies = []
for i in range(25):


    # shuffle data so you can get random test/train data from overall
    random.shuffle(full_data)

    # split data in data to 'train' with and data to 'test' with
    test_size = 0.2
    train_set = {2:[], 4: []}
    test_set = {2:[], 4:[]}
    # train_data = full_data[:-int(test_size*len(full_data))]  # all data from point 0 to somewhere
    # test_data = full_data[-int(test_size*len(full_data)):]   # all data from somewhere to end
    train_data = full_data[0:-300]
    test_data = full_data[-300:len(full_data)]


    for i in train_data:
        # train data's current data's class append everything except the class
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        # same for test data
        test_set[i[-1]].append(i[:-1])

    # print(len(full_data))
    # print(len(train_data))
    # print(len(test_data))




    correct = 0
    total = 0
    # loop thru each class group
    for group in test_set:
        # loop thru each class group's data
        for data in test_set[group]:
            # get the k nearest class for that data
            vote, confidence = k_nearest_neighbors(train_set, data, k=9)
            # if the k nearest class is same as the group then it is correct
            # (we previously removed the class but classified in class groups)
            if group == vote:
                correct += 1
            #else:
                #print(confidence)
            total += 1

    print('Accuracy:', float(correct/total))
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))

#result = k_nearest_neighbors(dataset,new_features,k=3)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=50, color=result)
# plt.show()