import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load the dataset (MovieLens)
data = fetch_movielens()
interaction_matrix = data['train']

# Create and train the model
model = LightFM(loss='warp')  # WARP is a ranking loss function
model.fit(interaction_matrix, epochs=30, num_threads=2)

# Evaluate the model
def evaluate_model(model, data):
    test_interactions = data['test']
    precision = precision_at_k(model, test_interactions, k=5).mean()
    return precision

precision = evaluate_model(model, data)
print(f"Precision at K=5: {precision:.2f}")

# Get recommendations for a user
def get_recommendations(model, data, user_id, num_recommendations=5):
    scores = model.predict(user_id, np.arange(data['train'].shape[1]))
    top_items = data['item_labels'][np.argsort(-scores)]
    return top_items[:num_recommendations]

user_id = 0  # Example user ID
recommendations = get_recommendations(model, data, user_id)
print(f"Top recommendations for user {user_id}: {recommendations}")
