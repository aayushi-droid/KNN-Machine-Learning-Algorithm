# -*- coding: utf-8 -*-
"""K-NN Algorithm.ipynb
"""

import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def eculidean_distance(point,k):
    euc_distance = np.sqrt(np.sum((X - point)**2, axis=1))
    return np.argsort(euc_distance)[0:k]

def predict(prediction_points, k):
    points_labels = []

    for point in prediction_points:
        distances = eculidean_distance(point,k)

        results =[]
        for index in distances:
            results.append(y[index])

        label = Counter(results).most_common(1)
        points_labels.append([point,label[0][0]])
    return points_labels

def get_accuracy(predictions):
    error = np.sum((predictions-y)**2)
    accuracy = 100 - (error/len(y)) *100
    return accuracy

# Implement
X, y = make_blobs(
    n_samples=100, n_features=2, centers=2,cluster_std=7,
    random_state=2020)

X = pd.DataFrame(X)
y = pd.DataFrame(y)
print(f'Independent Variable X : \n{X}')
print(f'Dependent Variable y : \n{y}')

prediction_points = [[-6,15],[-3,4],[-15,5,],[-2,5],[-9,10],[0,-10]]
prediction_points = np.array(prediction_points)
print(f'Point For Predictions :\n{prediction_points}')

results = predict(prediction_points,4)
print(f'Results :\n{results}')

"""# Check Accuracy"""

accu = []
for k in range(1,10):
    results = predict(X,k)
    predicitions = []
    for result in results:
        predicitions.append(result[1])
    accu.append([get_accuracy(predicitions), k])

print(f'Accuracy :\n{accu}')

accuracy = []
k = []
for a in accu:
    k.append(a[1])
    accuracy.append(a[0])
plt.figure(figsize=(10,5))
plt.plot(k,accuracy)
plt.xlabel('K-Value')
plt.ylabel('Accuracy')
plt.show()