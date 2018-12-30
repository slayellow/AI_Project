import matplotlib.colors as colors
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import csv

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_dataset = pd.read_csv('./fashionmnist/fashion-mnist_train.csv')
test_dataset = pd.read_csv('./fashionmnist/fashion-mnist_test.csv')

train_label = train_dataset.iloc[:, 0]
train_data = train_dataset.iloc[:, 1:]

KMeans_size = 10

kmeans = KMeans(n_clusters=KMeans_size, random_state=1, max_iter=10)
kmeans.fit(train_data)
y_predict = kmeans.predict(train_data)

counts = np.zeros(shape=(KMeans_size, 10))
for i in range(len(y_predict)):
    counts[y_predict[i]][train_label[i]] += 1
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
print(labels_map)

for i in range(len(y_predict)):
    y_predict[i] = labels_map[y_predict[i]]

print(y_predict[0:20])
print(list(train_label[0:20]))

print(accuracy_score(train_label, y_predict))

estimator = PCA(n_components=3)
X_pca = estimator.fit_transform(train_data)

kmeans = KMeans(n_clusters=KMeans_size, random_state=1, max_iter=5)
kmeans.fit(X_pca)
# y_predict = kmeans.predict(X_pca)

test_data = test_dataset.iloc[:, 1:]
test_label = test_dataset.iloc[:, 0]

est = PCA(n_components=3)
X_test_pca = est.fit_transform(test_data)
y_predict = kmeans.predict(X_test_pca)
print(est.explained_variance_ratio_)
counts = np.zeros(shape=(KMeans_size, 10))
for i in range(len(y_predict)):
    counts[y_predict[i]][test_label[i]] += 1
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
print(labels_map)

for i in range(len(y_predict)):
    y_predict[i] = labels_map[y_predict[i]]

print(y_predict[0:20])
print(list(test_label[0:20]))

print(accuracy_score(test_label, y_predict))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
for i in range(10):
    px = X_test_pca[:, 0][test_label == i]
    py = X_test_pca[:, 1][test_label == i]
    pz = X_test_pca[:, 2][test_label == i]
    ax.scatter(px, py, pz, c=colors[i], label=labels[i])
#    ax.scatter(px,py, c=colors[i],label=labels[i])
plt.legend()
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
plt.show()
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
for i in range(10):
    px = X_test_pca[:, 0][y_predict == i]
    py = X_test_pca[:, 1][y_predict == i]
    pz = X_test_pca[:, 2][y_predict == i]
    ax.scatter(px, py, pz, c=colors[i], label=labels[i])
#    ax.scatter(px,py, c=colors[i],label=labels[i])
plt.legend()
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
plt.show()
plt.close()
