import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


# df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# df.replace('?', -99999, inplace=True)
# df.drop(['id'], 1, inplace=True)
#
# x = np.array(df.drop(['class'], 1))
# y = np.array(df['class'])
#
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)
#

data_dict = {1:np.array([[2,0,1],[2,2,2],[2,4,4],[2,-2,6]]),
             0:np.array([[7,3,8],[4,2,7],[8,4,5]])}

x = []
y = []

for i in data_dict:
    for n in data_dict[i]:
        x.append(n)
        y.append(i)


# this just creates the algorithm?module?classifier
clf = svm.SVC(kernel='poly', C=1000)

# this puts our 'train' data into it and the classifier makes appropriate adjustments to make it work.
clf.fit(x,y)



print(clf.score([[2.9,1,6]],[1]))

colors = {1:'r', -1:'b', 2:'r', -2:'b', 666:'r', -666:'b', 0:'b'}

[[plt.scatter(x[0], x[1], s=100, color=colors[i]) for x in data_dict[i]] for i in data_dict]



plt.show()

# then we score how good the classifier was by inputting our 'test' data (data that was cut off)
# accuracy = clf.score(x_test, y_test)
# print(accuracy)




# example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
# example_measures = example_measures.reshape(2, -1)
#
# prediction = clf.predict(example_measures)
# print(prediction)

