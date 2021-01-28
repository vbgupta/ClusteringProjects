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
kmeans = KMeans(n_clusters=4, random_state=222)
kmeans.fit(data[0])
plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = "viridis")

# %%
features = data[0]

## plot the clusters with their entroids
def plot_centroid_clusters():
    clusters = kmeans.cluster_centers_
    y_kmeans = kmeans.fit_predict(features)
    # Array of Kmeans to DataFrame
    df_means = pd.DataFrame(y_kmeans, columns = ['Kmeans'])
    print(len(df_means))
    # DataFrame of the features and the Kmeans
    df = pd.DataFrame(features, df_means.Kmeans)
    print(df)
    #print(y_kmeans)
    plt.scatter(features[y_kmeans == 0,0], features[y_kmeans == 0,1], s = 20, color = "blue")
    plt.scatter(features[y_kmeans == 1,0], features[y_kmeans == 1,1], s = 20, color = "green")
    plt.scatter(features[y_kmeans == 2,0], features[y_kmeans == 2,1], s = 20, color = "yellow")
    plt.scatter(features[y_kmeans == 3,0], features[y_kmeans == 3,1], s = 20, color = "purple")
    plt.scatter(clusters[0][0], clusters[0][1], marker = '*', s = 250, color = "black")
    plt.scatter(clusters[1][0], clusters[1][1], marker = '*', s = 250, color = "black")
    plt.scatter(clusters[2][0], clusters[2][1], marker = '*', s = 250, color = "black")
    plt.scatter(clusters[3][0], clusters[3][1], marker = '*', s = 250, color = "black")
    plt.show()
plot_centroid_clusters()
