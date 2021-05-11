import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

max_iter = 1000
lr = .03
sigma = .20
probability = .5

def kernel(X, v, sigma):
    sq_sum = np.square(X - v).sum(axis=1)
    denom = 2 * sigma * sigma
    return np.exp(-sq_sum/denom)

# preprocessing
data = pd.read_csv('./data/haberman.csv').dropna()
# create train, test sets
X = data[['age', 'year', 'numnodes']].to_numpy()
y = data['survival'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# train
weights = np.full(X_train.shape[0], .5)
beta = np.array([0])
for i in range(max_iter):
    for j in range(len(X_train)):
        # kernalize
        row = X_train[j]
        target = y_train[j]
        kernels = kernel(X_train, row, sigma)
        # apply weights
        z = kernels.dot(weights) + beta
        # predict
        y_hat = 1 / (1 + np.e**(-z))
        # Gradient descent
        weights = weights + kernels * (target - y_hat) / len(X_train)
        beta = beta + lr * (target - y_hat) / len(X_train)

# test
y_pred = []
for row in X_test:
    kernels = kernel(X_train, row, sigma)
    # apply weights
    z = kernels.dot(weights) + beta
    # predict
    y_hat = 1 / (1 + np.e**(-z))
    if y_hat > probability:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(y_pred)
# Performance
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy:', np.round(accuracy, decimals=4))
print('F1:', np.round(f1, decimals=4))