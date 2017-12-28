import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation, model_selection
import pandas as pd
from sklearn.utils import shuffle



df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name', 'boat'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#df = shuffle(df)
#print(df.head())


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
print(df.head())


x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])


clf = MeanShift()
clf.fit(x)

# print(clf.labels_)
labels = clf.labels_
# print(set(labels))
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

# print(original_df.head())
for i in range(len(x)):
    original_df['cluster_group'].iloc[i] = labels[i]
# print(original_df.head())

n_clusters_ = len(np.unique(labels))
survival_rates = {}


for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    print(temp_df.head())
    print("####################")
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate


print(survival_rates)