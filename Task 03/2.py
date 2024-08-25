import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs

# Step 1: Generate Synthetic Data for Testing

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: Perform Hierarchical Clustering

# Compute the distance matrix
dists = sch.distance.pdist(X)

# Perform hierarchical clustering using the 'ward' linkage method
Z = sch.linkage(dists, method='ward')

# Step 3: Visualize the Dendrogram

plt.figure(figsize=(10, 7))
sch.dendrogram(Z, truncate_mode='lastp', p=12)
plt.title('Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

# Step 4: Decide the Number of Clusters

# To decide the number of clusters, use the 'fcluster' method
from scipy.cluster.hierarchy import fcluster

# Define the maximum number of clusters
max_clusters = 4

# Form flat clusters
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# Step 5: Visualize the Clusters

plt.figure(figsize=(8, 6))

# Plot data points with cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=clusters, s=50, cmap='viridis', marker='o', edgecolor='k')

plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
