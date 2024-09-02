# Install required libraries
# pip install numpy pandas scikit-surprise

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset from the 'surprise' library
# This will automatically download and load the data for you
data = Dataset.load_builtin('ml-100k')

# Define a Reader to parse the MovieLens data
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('ml-100k/u.data', reader=reader)

# Split data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Build and train the SVD model
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Calculate and print accuracy metrics
print("RMSE (Root Mean Squared Error):", accuracy.rmse(predictions))
print("MAE (Mean Absolute Error):", accuracy.mae(predictions))

# Function to get recommendations for a user
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    # Sort and get the top-n recommendations
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get top-n recommendations
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Print recommendations for a specific user
user_id = '196'  # Example user ID from the MovieLens dataset
print(f"Top recommendations for user {user_id}:")
for iid, rating in top_n_recommendations.get(user_id, []):
    print(f"Item ID: {iid}, Estimated Rating: {rating:.2f}")

# Optional: For visualization or further analysis, you might want to create a DataFrame
recommendations_df = pd.DataFrame.from_dict(top_n_recommendations, orient='index').apply(pd.Series)
recommendations_df.to_csv('top_n_recommendations.csv', index=False)
print("Recommendations saved to 'top_n_recommendations.csv'")

