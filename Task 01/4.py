import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('e:\\MachineLearning- Internship\\Task 01\\data3.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Get a summary of the dataset
print("\nSummary statistics of the dataset:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Check the data types
print("\nData types in the dataset:")
print(df.dtypes)

# Univariate Analysis

# Histograms for numeric features
df.hist(figsize=(10, 8), bins=20)
plt.tight_layout()
plt.show()

# Box plot for numeric features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title('Box Plot of Numeric Features')
plt.show()

# Bar plot for categorical features
plt.figure(figsize=(10, 6))
sns.countplot(x='category', data=df)
plt.title('Count of Each Category')
plt.show()

# Bivariate Analysis

# Scatter plot between feature1 and feature2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='feature1', y='feature2', hue='target', data=df)
plt.title('Scatter Plot of Feature1 vs Feature2')
plt.show()

# Pair plot
sns.pairplot(df, hue='target')
plt.show()

# Heatmap of Correlation Matrix
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Additional Insights

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable')
plt.show()

# Box plot for feature1 by category
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='feature1', data=df)
plt.title('Box Plot of Feature1 by Category')
plt.show()
