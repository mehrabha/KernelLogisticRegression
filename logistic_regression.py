import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# params
lr = .01
loops = 5000
probability = .5

# preprocessing
data = pd.read_csv('./data/data.csv')
# create train, test sets
X = data[['radius_mean', 'texture_mean', 'smoothness_mean']].to_numpy()
y = data['diagnosis'].to_numpy()
y = [1 if i == 'M' else 0 for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

# train
weights = np.zeros(X_train.shape[1])
for i in range(loops):
    # Predict X
    z = X_train.dot(weights)
    y_hat = 1 / (1 + np.e**(2 -z))
    # Gradient descent
    weights = weights - lr * X_train.T.dot(y_hat - y_train) / len(X_train)
    break

# test
z = X_test.dot(weights)
y_pred = z > probability

# Performance
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy:', np.round(accuracy, decimals=4))
print('F1:', np.round(f1, decimals=4))