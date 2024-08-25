import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('e:\MachineLearning- Internship\Task 01\data.csv')

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Handle missing values
# Fill missing values with the column mean
df_filled = df.fillna(df.mean())

# Display DataFrame after filling missing values
print("\nDataFrame after handling missing values:")
print(df_filled)

# Step 3: Detect and remove outliers
# Calculate Z-scores
z_scores = (df_filled - df_filled.mean()) / df_filled.std()
# Remove rows where Z-scores are greater than 3 or less than -3
df_no_outliers = df_filled[(z_scores > -3).all(axis=1) & (z_scores < 3).all(axis=1)]

# Display DataFrame after removing outliers
print("\nDataFrame after removing outliers:")
print(df_no_outliers)

# Step 4: Normalize/Standardize
# Normalize/Standardize the data
df_normalized = (df_no_outliers - df_no_outliers.mean()) / df_no_outliers.std()

# Display DataFrame after normalization/standardization
print("\nNormalized/Standardized DataFrame:")
print(df_normalized)

# Step 5: Save the cleaned data
df_normalized.to_csv('e:\MachineLearning- Internship\Task 01\cleaned_dataset.csv', index=False)

print("\nCleaned data saved to 'cleaned_dataset.csv'")
