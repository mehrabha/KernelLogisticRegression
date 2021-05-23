import numpy as np
from numpy.core.arrayprint import format_float_scientific
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats

base_dir = './generated_files/'
max_iter = 15000
lr = .05
probability = .5

class KernelRegression():
    def __init__(self, lr=.01, probability=.5, use_beta=True):
        self.lr = lr
        self.probability = .5
        self.use_beta=use_beta
    
    def fit(self, X, y, p):
        self.X = X
        self.y = y
        self.p = p
        self.beta = 0
        self.weights = np.zeros(X.shape[1])
        self.wfile = np.zeros((max_iter, X.shape[1]))
        return self

    def train(self, max_iter=10000):
        X = self.X
        y = self.y
        lr = self.lr

        for iter in range(max_iter):
            kernels = self.kernel(X)
            z = kernels.dot(self.weights) + self.beta
            # predict
            y_hat = 1 / (1 + np.exp(- z))
            # Gradient descent
            self.weights -= lr * X.T.dot(y_hat - y) / len(X)
            if self.use_beta:
                self.beta -= lr * np.sum(y_hat - y) / len(X) * .5
            self.wfile[iter] = self.weights

    def predict(self, X_test, probs=False):
        # test
        kernels = self.kernel(X_test)
        # apply weights
        z = kernels.dot(self.weights) + self.beta
        # predict
        y_hats = 1 / (1 + np.exp(-z))

        if probs:
            return y_hats
        return [1 if i > self.probability else 0 for i in y_hats]

    def kernel(self, X):
        p = self.p
        kernel_values = np.zeros(X.shape)
        for i in range(X.shape[0]):
            row = X[i]
            value = [row[0]**p[0], row[1]**p[1], row[2]**p[2]]
            kernel_values[i] = value
        return kernel_values

# preprocessing
data = pd.read_csv('./data/data.csv')
# create train, test sets
X = data[['radius_mean', 'texture_mean', 'symmetry_mean']].to_numpy()
y = data['diagnosis'].to_numpy()
y = [1 if i == 'M' else 0 for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

np.savetxt(base_dir + 'dataset.txt', X, delimiter=' ', fmt='%f')
np.savetxt(base_dir + 'target.txt', y, delimiter=' ', fmt='%f')

# pearson's coefficients
p_vals = open(base_dir + "p_vals.txt", "w")
p0 = stats.pearsonr(X_train[:, 0], y_train)[0]
p1 = stats.pearsonr(X_train[:, 1], y_train)[0]
p2 = stats.pearsonr(X_train[:, 2], y_train)[0]
p_vals.write(str(p0) + ' ' + str(p1) + ' ' + str(p2))

# Train
model = KernelRegression(lr, probability)
model.fit(X_train, y_train, [p0, p1, p2]).train(max_iter)
np.savetxt(base_dir + 'weights.txt', model.wfile, delimiter=' ', fmt='%f')

# ROC
y_probs = model.predict(X_test, probs=True)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs)
plt.xlabel("False Positives")
plt.ylabel("True Positives")
plt.plot(fpr, tpr)
plt.show()

# Performance
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy:', np.round(accuracy, decimals=4))
print('F1:', np.round(f1, decimals=4))