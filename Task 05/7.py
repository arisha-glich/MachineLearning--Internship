import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 
                 'relationship', 'race', 'sex', 'hours_per_week', 'native_country', 'income']
data = pd.read_csv(url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)

# Preprocessing
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)  # Convert to binary classification

# Define categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train and evaluate traditional ML models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy}")
    print(report)
    
    plt.figure()
    plt.title(f'{model_name} ROC Curve')
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Train and evaluate Deep Learning model
def build_dnn_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

dnn_model = build_dnn_model(X_train_preprocessed.shape[1])
history = dnn_model.fit(X_train_preprocessed, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate Deep Learning model
y_pred_dnn = (dnn_model.predict(X_test_preprocessed) > 0.5).astype("int32")
accuracy_dnn = accuracy_score(y_test, y_pred_dnn)
report_dnn = classification_report(y_test, y_pred_dnn)
cm_dnn = confusion_matrix(y_test, y_pred_dnn)
y_proba_dnn = dnn_model.predict(X_test_preprocessed)
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_proba_dnn)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)

print("Deep Learning Model Performance:")
print(f"Accuracy: {accuracy_dnn}")
print(report_dnn)

plt.figure()
plt.title('Deep Learning ROC Curve')
plt.plot(fpr_dnn, tpr_dnn, label=f'ROC curve (area = {roc_auc_dnn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()


#it takes long to run 

#THANKS FOR WATCHING, ITS THE LAST TIME OF COSMICODE INTERNSHIP :)