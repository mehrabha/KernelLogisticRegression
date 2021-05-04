import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# params
lr = .1
loops = 1000
probability = .5

# preprocessing
# data source: https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
data = pd.read_csv('./data/stroke-dataset.csv').dropna().head(600)
# encode gender with 0 & 1s
data['gender'] = [0 if i == 'Female' else 1 for i in data['gender']]
# create train, test sets
X = data[['gender', 'age', 'hypertension', 'heart_disease',
            'avg_glucose_level', 'bmi']]
y = data['stroke']
unique, counts = np.unique(y, return_counts = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

# train
weights = np.random.rand(X_train.shape[1])
for i in range(loops):
    # Predict X
    z = X_train.dot(weights)
    y_hat = 1 / (1 + np.e**(-z))
    # Gradient descent
    weights = weights - lr * X_train.T.dot(y_hat - y_train) / len(X_train)

# test
z = X_test.dot(weights)
y_pred = z > probability

# Performance
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy:', np.round(accuracy, decimals=4))
print('F1:', np.round(f1, decimals=4))