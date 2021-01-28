# Kmeans Clustering make_blobs
#%%
## Importing Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

## The dataset
data = make_blobs(n_samples = 200,
centers = 4, n_features=2, cluster_std=1.6, random_state=111)
#print(data)

## Import Kmeans from sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
plt.scatter(data[0][:,0], data[0][:,1])
