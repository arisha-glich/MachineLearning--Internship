import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 3, 5])

# Initialize parameters
m, b = 0, 0
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    D_m = (-2/len(X)) * sum(X * (y - y_pred))
    D_b = (-2/len(X)) * sum(y - y_pred)
    m = m - learning_rate * D_m
    b = b - learning_rate * D_b

# Plotting the results
plt.scatter(X, y)
plt.plot(X, m * X + b, color='red')
plt.show()
