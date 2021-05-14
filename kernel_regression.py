import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import stats

max_iter = 10000
lr = .01
probability = .5

def kernel(X_train):
    kernel_values = np.zeros(X_train.shape[0])
    for i in range(X_train.shape[0]):
        row = X_train[i]
        value = (row[0] + row[1] + .5)**3
        kernel_values[i] = 1 / (1 + np.exp(-value))
    return kernel_values

# preprocessing
data = pd.read_csv('./data/data.csv')
# create train, test sets
X = data[['radius_mean', 'texture_mean']].to_numpy()
y = data['diagnosis'].to_numpy()
y = [1 if i == 'M' else 0 for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

np.savetxt('dataset.txt', X, delimiter=' ')
np.savetxt('target.txt', y, delimiter=' ')

# pearson's coefficients
p_vals = open("p_vals.txt", "a")
p1 = stats.pearsonr(X_train[:, 0], y_train)[0]
p2 = stats.pearsonr(X_train[:, 1], y_train)[0]
p_vals.write(str(p1) + ' ' + str(p2))

# train
weights = np.zeros(X_train.shape[1])
wfile = np.zeros((max_iter, X_train.shape[1] + 1))
k = 0
beta = 0
for iter in range(max_iter):
    kernels = kernel(X_train)
    z = X_train.dot(weights) + kernels * k + beta
    # predict
    y_hat = 1 / (1 + np.exp(-z))
    # Gradient descent
    weights = weights - lr * X_train.T.dot(y_hat - y_train) / len(X_train)
    k = k - lr * kernels.dot(y_hat - y_train) / len(X_train)
    beta = beta - lr * np.sum(y_hat - y_train) / len(X_train) * .5
    wfile[iter] = np.append(weights, k)

np.savetxt('weights.txt', wfile, delimiter=' ')

# test
kernels = kernel(X_test)
# apply weights
z = X_test.dot(weights) + kernels * k + beta
# predict
y_hats = 1 / (1 + np.exp(-z))
y_pred = [1 if i > probability else 0 for i in y_hats]

# Performance
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy:', np.round(accuracy, decimals=4))
print('F1:', np.round(f1, decimals=4))