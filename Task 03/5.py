import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.colors import ListedColormap

# Step 1: Generate a Dataset with Outliers

# Create a dataset with normal points and outliers
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=0)
rng = np.random.RandomState(42)
X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(50, 2))])  # Adding outliers

# Step 2: Apply Isolation Forest

# Initialize and fit Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=0)
iso_labels = iso_forest.fit_predict(X)

# Step 3: Apply Local Outlier Factor (LOF)

# Initialize and fit LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
lof_labels = lof.fit_predict(X)

# Step 4: Visualize the Results

# Create a mesh grid for plotting decision boundaries
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]

# Predict with Isolation Forest
iso_preds = iso_forest.predict(X_grid)
iso_preds = iso_preds.reshape(xx.shape)

# Predict with LOF
lof_preds = lof.fit_predict(X_grid)
lof_preds = lof_preds.reshape(xx.shape)

# Plot Isolation Forest results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=iso_labels, cmap=ListedColormap(['#ff0000', '#00ff00']), edgecolor='k')
plt.title('Isolation Forest')

# Plot LOF results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=lof_labels, cmap=ListedColormap(['#ff0000', '#00ff00']), edgecolor='k')
plt.title('Local Outlier Factor (LOF)')

plt.tight_layout()
plt.show()
