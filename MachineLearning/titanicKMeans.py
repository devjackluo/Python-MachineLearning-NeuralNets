import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation, model_selection
import pandas as pd
from sklearn.utils import shuffle


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'boat', 'sibsp', 'ticket' , 'fare', 'home.dest', 'pclass', 'parch', 'embarked', 'age', 'cabin'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = shuffle(df)
print(df.head())


def handle_non_numerical_data(df):

    # ['pclass' 'survived' 'sex' 'age' 'sibsp' 'parch' 'ticket' 'fare' 'cabin' 'embarked' 'boat' 'home.dest']
    columns = df.columns.values

    for column in columns:

        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            # print(unique_elements)
            x = 0
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x += 1

            # df[column] = list(map(lambda x: text_digits_vals[x], df[column]))
            df[column] = list(map(convert_to_int, df[column]))

    return df


df = handle_non_numerical_data(df)
#print(df)


x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])


clf = KMeans(n_clusters=2)
clf.fit(x)

print(clf.labels_)
labels = clf.labels_
print(set(labels))

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    # change array of predictions into an [] where it contains 1 array and each of those has len(predict_me)
    predict_me = predict_me.reshape(1, len(predict_me))

    prediction = clf.predict(predict_me)

    if prediction[0] == y[i]:
        correct+=1

print(correct/len(x))




















##############################
##############################
# x_train = np.split(x, 7)
#
# xT = []
# yT = []
# count = 0
# for i in range(len(x_train)-1):
#     for n in x_train[i]:
#         xT.append(n)
#         yT.append(y[count])
#         count+=1
#
#
# xT = np.array(xT)
#
# xTest = []
# yTest = []
# for n in x_train[-1]:
#     xTest.append(n)
#     yTest.append(y[count])
#     count+=1
#
#
# xTest = np.array(xTest)


# clf = KMeans(n_clusters=2)
# clf.fit(xT)
#
# correct = 0
# for i in range(len(xTest)):
#     predict_me = np.array(xTest[i].astype(float))
#     # change array of predictions into an [] where it contains 1 array and each of those has len(predict_me)
#     predict_me = predict_me.reshape(1, len(predict_me))
#     # print(predict_me)
#     # print(predict_me)
#     # print("")
#     prediction = clf.predict(predict_me)
#     #print(prediction)
#     if prediction[0] == yTest[i]:
#         correct+=1
#
# print(correct/len(x))



