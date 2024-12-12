import numpy as np
import pandas as pd
import tensorflow as tf

dataset_train = pd.read_csv("dataset/train.csv")
dataset_test = pd.read_csv("dataset/test.csv")

#filling out missing data, specific the 'age' (mean, median and mode can be used just with numbers)
dataset_train['Age'] = dataset_train['Age'].fillna(dataset_train['Age'].median())
#filling out missing data with 'Unknown'
dataset_train['Cabin'].fillna('Unknown', inplace=True)

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

# splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scaling the data for standard
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# creating a variable for the ANN
ann = tf.keras.models.Sequential()

# adding the layers input, hidden layers and output layer
ann.add(tf.keras.layers.Dense(units=12, activation="relu"))
ann.add(tf.keras.layers.Dense(units=12, activation="relu"))
ann.add(tf.keras.layers.Dense(units=12, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# compiling the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# training the ANN
ann.fit(x_train, y_train, batch_size=32, epochs=100)