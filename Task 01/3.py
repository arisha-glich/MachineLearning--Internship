import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
df = pd.read_csv('e:\MachineLearning- Internship\Task 01\data2.csv')

# Handle missing values
df_filled = df.fillna(df.mean())

# Calculate the correlation matrix
correlation_matrix = df_filled.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Assume the target variable is 'target' in the DataFrame
# You can replace 'target' with the actual name of your target column
X = df_filled.drop(columns=['target'])
y = df_filled['target']

# Calculate mutual information
mutual_info = mutual_info_classif(X, y)

# Create a DataFrame for the mutual information
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})

# Sort the DataFrame by mutual information values
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Plot mutual information
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='Mutual Information', data=mi_df)
plt.xticks(rotation=90)
plt.title('Mutual Information of Features')
plt.show()
