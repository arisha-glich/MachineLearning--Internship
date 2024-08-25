import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Load and prepare the dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Evaluate the model
print(f"Initial Model Accuracy: {accuracy_score(y_test, y_pred)}")
print("Initial Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Initial Classification Report:")
print(classification_report(y_test, y_pred))

# Step 3: Hyperparameter Tuning Using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best Parameters found by GridSearchCV:", grid_search.best_params_)
print("Best Cross-Validation Score achieved during GridSearchCV:", grid_search.best_score_)

# Step 4: Evaluate the Tuned Model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred_best)}")
print("Tuned Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("Tuned Classification Report:")
print(classification_report(y_test, y_pred_best))

# Step 5: Cross-Validation on the Tuned Model
cv_scores = cross_val_score(best_rf, X, y, cv=5)

print(f"Cross-Validation Scores for Tuned Model: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")
