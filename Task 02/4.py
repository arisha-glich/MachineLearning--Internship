import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Step 1: Load a complex dataset (e.g., the moons dataset)
X, y = datasets.make_moons(n_samples=100, noise=0.3, random_state=42)

# Step 2: Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the SVM classifier
svm = SVC(kernel='rbf', C=1, gamma=0.5, random_state=42)
svm.fit(X_train, y_train)

# Step 5: Visualize the decision boundary
def plot_decision_boundary(model, X, y):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict classifications for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(('red', 'blue')))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# Step 6: Call the plotting function
plot_decision_boundary(svm, X_test, y_test)

# Optional: Evaluate the model on the test data
y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy:.2f}")
