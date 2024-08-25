# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Preprocess the Data
# Replace 'data.csv' with your dataset file
df = pd.read_csv('e:\\MachineLearning- Internship\\Task 01\\data4.csv')

# Dropping non-numeric columns if any
df_numeric = df.select_dtypes(include=[np.number])

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# Step 2: Apply PCA for Dimensionality Reduction
# Initialize PCA for 2 components
pca = PCA(n_components=2)  # For 2D visualization
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# If you have a target variable, include it for color-coding in visualization
if 'target' in df.columns:
    pca_df['target'] = df['target']

# Step 3: Visualize the Results
# 2D Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, palette='viridis')
plt.title('PCA 2D Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target')
plt.show()

# For 3D visualization, if you have more than 2 principal components:
# Apply PCA for 3 components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
if 'target' in df.columns:
    pca_df['target'] = df['target']

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['target'], cmap='viridis')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('PCA 3D Plot')
plt.colorbar(scatter, label='Target')
plt.show()
