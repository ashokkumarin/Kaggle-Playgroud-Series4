# Import required libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

sys.path.append ("mylib")

from visualization import *
from utility import *

# Load the dataset
train_data = pd.read_csv('data\\train.csv')
test_data = pd.read_csv('data\\test.csv')

pred_df = pd.DataFrame()
pred_df['id'] = test_data['id']

# Display the first few rows
print("Dataset Overview:")
print(train_data.head())

# Check for missing values
print("\nMissing Values:")
print(train_data.isnull().sum())

# Drop unwanted columns
unwanted_col = {"id", "Name"}
for col in unwanted_col:
    train_data = train_data.drop([col], axis=1)
    test_data = test_data.drop([col], axis=1)

# Fill missing values
fill_dummy_values(train_data)
fill_dummy_values(test_data)

# Encode categorical columns
categorical_columns = train_data.select_dtypes(exclude=[np.number])
train_data = encode_categorical(train_data, categorical_columns)
test_data = encode_categorical(test_data, categorical_columns)

# Display updated data structure
# print("\nDataset After Encoding:")
# print(train_data.head())

# Correlation heatmap to understand relationships
# plt.figure(figsize=(12, 8))
# Select numerical columns
# numerical_data = train_data.select_dtypes(include=[np.number])

# Calculate correlation matrix
# corr_matrix = numerical_data.corr()

# Use corr_matrix for heatmap or further analysis
# sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
# #sns.heatmap(data.corr(), cmap="coolwarm", annot=False, fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

# Define the target variable (e.g., 'Depression') and features
X = train_data.drop(['Depression'], axis=1)
y = train_data['Depression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Build a machine learning model
# model = RandomForestClassifier(random_state=42)
# model = DecisionTreeClassifier(random_state=42, max_depth=6)
# model = LogisticRegression()
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model = SVC(kernel='linear')  # MODEL FIT IS TAKING TOO LONG - NOT USED
# model = KNeighborsClassifier(n_neighbors=15) 
# model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
# feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Predict with actual test data
X_pred = test_data[X_train.columns]
y_pred = model.predict(X_pred)

pred_df['Depression'] = y_pred
pred_df.to_csv("submission.csv", index=False)
print("Prediction file has been created")

# Plot feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
# plt.title("Feature Importance")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()
