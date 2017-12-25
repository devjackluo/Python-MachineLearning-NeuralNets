import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
from sklearn.cluster import KMeans


accuracies = []

for i in range(25):

    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    x = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)

    clf.fit(x_train,y_train)

    accuracy = clf.score(x_test, y_test)
    print(accuracy)
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
    # example_measures = example_measures.reshape(2, -1)
    #
    # prediction = clf.predict(example_measures)
    # print(prediction)

    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))
