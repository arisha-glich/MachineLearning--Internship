import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

# Step 1: Generate a Noisy Dataset

# Create a dataset with noise
X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)

# Step 2: Apply Gaussian Mixture Models (GMM) Clustering

# Define the number of clusters for GMM
n_components = 2

# Initialize and fit GMM
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
gmm_labels = gmm.fit_predict(X)
gmm_probs = gmm.predict_proba(X)

# Step 3: Visualize the Probability Distributions of the Clusters

# Create a grid to evaluate the Gaussian Mixture Models
x = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
y = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.c_[X_grid.ravel(), Y_grid.ravel()]

# Compute the probabilities for each cluster
probs = np.exp(gmm.score_samples(XX))
probs = probs.reshape(X_grid.shape)

# Plot GMM clustering results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('GMM Clustering')

# Plot the probability distributions of the clusters
plt.subplot(1, 2, 2)
plt.contourf(X_grid, Y_grid, probs, levels=20, cmap='viridis', norm=LogNorm())
plt.colorbar(label='Probability Density')
plt.title('Cluster Probability Distribution')

plt.tight_layout()
plt.show()
