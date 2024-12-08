import numpy as np
import pandas as pd
import tensorflow as tf

dataset_train = pd.read_csv("dataset/train.csv")
dataset_test = pd.read_csv("dataset/test.csv")

# this code takes since the second column till end except for 8 and 3
x = dataset_train.iloc[:, [2, 4, 5, 6, 7, 9, 10, 11]].values
# this one takes just the first column
y = dataset_train.iloc[:, [1]].values


from sklearn.preprocessing import LabelEncoder

# transforming the gender column into 0 and 1
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# onehotencoder with 'cabine' column
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(sparse_output=False), [6, 7])],
    remainder="passthrough",
)  # using the sparse_output false i got the whole and in this case the desired output
x = ct.fit_transform(x)


print(x)
