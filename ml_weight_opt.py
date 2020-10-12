# Initialize neural network object and fit object
# ADAPTED FORM https://mlrose.readthedocs.io/en/stable/source/tutorial3.html

import mlrose_hiive as mlrose

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
df = pd.DataFrame(X, columns=breast_cancer.feature_names)


n_cols = breast_cancer.data.shape[1]
# stratify=y so train and test sets have same proportion of labels as unsplit
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,stratify=y,random_state=1)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
y_train_minmax = min_max_scaler.fit_transform(y_train.reshape(-1, 1))
df_foo = pd.DataFrame(X_train_minmax, columns=breast_cancer.feature_names)

# std_scaler = preprocessing.StandardScaler()
# X_train_std = std_scaler.fit_transform(X_train)
# X_test_std = std_scaler.fit_transform(X_test)
# y_train_std =std_scaler.fit_transform(y_train.reshape(-1, 1))

std_scaler = preprocessing.StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.fit_transform(X_test)
y_train_std =std_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_std =std_scaler.fit_transform(y_test.reshape(-1, 1))

nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[3], activation='relu',
                                 algorithm='random_hill_climb', max_iters=30,
                                 bias=True, is_classifier=True, learning_rate=0.0001,
                                 early_stopping=True, clip_max=5, max_attempts=100,
                                 random_state=3)
nn_rhc.fit(X_train_std, y_train)

from sklearn.metrics import accuracy_score

# Predict labels for train set and assess accuracy
print("Random Hill Climb:")
y_train_pred = nn_rhc.predict(X_train_std)

y_train_accuracy = accuracy_score(y_train, y_train_pred)

print(y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_rhc.predict(X_test_std)

y_test_accuracy = accuracy_score(y_test, y_test_pred)

print(y_test_accuracy)

print("="*20)
# nn_sa = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
#                                  algorithm='random_hill_climb', max_iters=1000,
#                                  bias=True, is_classifier=True, learning_rate=0.0001,
#                                  early_stopping=True, clip_max=5, max_attempts=100,
#                                  random_state=3)
# nn_sa.fit(X_train_std, y_train_std)
#
#
# nn_ga = mlrose.NeuralNetwork(hidden_nodes=[2], activation='relu',
#                                  algorithm='random_hill_climb', max_iters=1000,
#                                  bias=True, is_classifier=True, learning_rate=0.0001,
#                                  early_stopping=True, clip_max=5, max_attempts=100,
#                                  random_state=3)
# nn_ga.fit(X_train_std, y_train_std)