import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Step 1: Generate a Noisy Dataset

# Create a dataset with noise
X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)

# Step 2: Apply k-means Clustering

# Define the number of clusters for k-means
n_clusters = 2

# Initialize and fit k-means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_labels = kmeans.fit_predict(X)

# Step 3: Apply DBSCAN Clustering

# Define parameters for DBSCAN
eps = 0.3
min_samples = 5

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X)

# Step 4: Evaluate and Compare Clustering Performance

# Calculate silhouette scores for both clustering algorithms
kmeans_silhouette = silhouette_score(X, kmeans_labels)
dbscan_silhouette = silhouette_score(X, dbscan_labels[dbscan_labels != -1]) if len(set(dbscan_labels)) > 1 else -1

print(f"K-means Silhouette Score: {kmeans_silhouette:.2f}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")

# Step 5: Visualize the Clusters

# Plot k-means clustering results
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('K-means Clustering')

# Plot DBSCAN clustering results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('DBSCAN Clustering')

plt.show()
