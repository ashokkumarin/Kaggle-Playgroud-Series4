import numpy as np
import pandas as pd

def euclidean_distance(x, y):
  # Convert lists to NumPy arrays before subtraction
  x = np.array(x)
  y = np.array(y)
  return np.linalg.norm(x - y)

def cosine_similarity(x, y):
  return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def jaccard_similarity(x, y):
  intersection = len(list(set(x).intersection(set(y))))
  union = len(list(set(x).union(set(y))))
  return intersection / union

# Example usage:
vector1 = [1, 2, 3]
vector2 = [2, 3, 4]

train_data = pd.read_csv('data\\train.csv')

vector1 = train_data['Depression']
vector2 = train_data['Financial Stress']

print(f"Euclidean distance: {euclidean_distance(vector1, vector2)}")
print(f"cosine_similarity: {cosine_similarity(vector1, vector2)}")
print(f"jaccard_similarity: {jaccard_similarity(set(vector1), set(vector2))}")