import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

# X_train
X_train = pd.read_csv("dnn8_X_train.csv", sep=";",
                        header=None,
                        nrows=2000
)
print( "X_train", X_train.shape )

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
print( kmeans )

y_kmeans = kmeans.predict(X_train)

X_train_pca = PCA(n_components=10).fit_transform(X_train)
#plt.scatter(X_train_pca[:,0], X_train_pca[:,1])
centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='b', zorder=10)

# ---

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_train)

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
ax0.scatter( tsne_results[:,0], tsne_results[:,1] )
#plt.show(block=True)
ax0.set_title( "X_train" )

tsne_pca = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne_pca.fit_transform(X_train_pca)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
ax1.scatter( tsne_pca_results[:,0], tsne_pca_results[:,1] )
ax1.set_title( "X_train_pca") 
plt.show(block=True)



