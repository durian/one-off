#!/usr/bin/env python3
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import sys
import argparse

'''
# Try three clusters
est = KMeans(n_clusters=3)
est.fit(X)
est.predict(X)
labels = est.labels_
print(labels)
print(est.cluster_centers_)
print(est.inertia_)
silhouette_avg = silhouette_score(X, labels)
print( silhouette_avg )
'''

parser = argparse.ArgumentParser()
parser.add_argument( '-f', "--csv_file", type=str, default="cam100.csv", help='CSV file' )
parser.add_argument( '-t', "--type", type=str, default="kmeans", help='Cluster type' )
args = parser.parse_args()

# Data:
# framenum;class;conf;xmin;ymin;xmax;ymax;file;raw;epoch;area;cx;cy;colour
# 9;1.0;0.91793203;127.15648;88.385376;221.08005;184.96828;tl_20190115T103704Z-1547576230154.ts_bboxcords.txt;1547576270;2019-01-15 18:17:50;9071.41114464728;174.118265;136.676828;#aec7e8
#
X_train = pd.read_csv(args.csv_file, sep=";")
print( "X_train", X_train.shape )
X_data = X_train[["cx", "cy"]]
X_data = X_data[(X_data.T != 0).any()]
print( X_data[ (X_data["cx"] < 1.0) | (X_data["cy"] < 1.0) ] )
print( "X_data", X_data.values )#.head() )

for cl in [3, 4, 5]:
    if args.type == "kmeans":
        estimator = KMeans(n_clusters=cl)
    elif args.type == "dbscan":
        estimator = DBSCAN(eps=cl, min_samples=10)
    elif args.type == "average":
         estimator = AgglomerativeClustering(n_clusters=cl, linkage='average')
    elif args.type == "complete":
         estimator = AgglomerativeClustering(n_clusters=cl, linkage='complete')
    elif args.type == "ward":
         estimator = AgglomerativeClustering(n_clusters=cl, linkage='ward')
    estimator.fit(X_data.values)
    print( estimator )
    #print( estimator.transform(X_data.values) )
    #y_estimator = estimator.predict(X_data.values)
        
    print( estimator.labels_ )
    ##print( estimator.cluster_centers_ )
        
    fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax0.scatter( X_data["cx"], X_data["cy"], c=estimator.labels_ )
    ax0.set_title( "X_data "+str(cl)+" clusters "+args.type )

    if args.type == "kmeans":
        ax0.scatter( estimator.cluster_centers_[:,0], estimator.cluster_centers_[:,1], c="#ff0000" )
    
plt.show(block=True)

sys.exit(1)

#X_train_pca = PCA(n_components=2).fit_transform(X_data)
#plt.scatter(X_train_pca[:,0], X_train_pca[:,1])
#centroids = estimator.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='b', zorder=10)

# ---

#tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#tsne_results = tsne.fit_transform(X_data)

fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
ax0.scatter( tsne_results[:,0], tsne_results[:,1] )
#plt.show(block=True)
ax0.set_title( "X_data" )

tsne_pca = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne_pca.fit_transform(X_train_pca)

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
ax1.scatter( tsne_pca_results[:,0], tsne_pca_results[:,1] )
ax1.set_title( "X_train_pca") 
plt.show(block=True)



