import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate XGBoost classifier (without use_label_encoder)
xgb_clf = xgb.XGBClassifier()

# Define hyperparameters to tune
params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
}

# Use GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=params, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_  # Fixed typo here
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy}")
