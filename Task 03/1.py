import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Step 1: Implement K-means Clustering from Scratch

def kmeans(X, k, max_iters=100, tol=1e-4):
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Compute new centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Step 2: Generate Synthetic Data for Testing

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 3: Apply K-means Algorithm

k = 4  # Number of clusters
centroids, labels = kmeans(X, k)

# Step 4: Visualize the Clusters

plt.figure(figsize=(8, 6))

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', marker='o', edgecolor='k')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
