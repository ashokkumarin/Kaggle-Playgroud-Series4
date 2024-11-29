# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('data\\train.csv')
test = pd.read_csv('data\\test.csv')

# Display the first few rows
print("Dataset Overview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Replace missing values (example: fill numeric NaN with 0 and categorical NaN with "Unknown")
data.fillna({'Academic Pressure': 0, 'Work Pressure': 0, 'CGPA': 0}, inplace=True)

# Feature engineering: Encode categorical columns
categorical_columns = ['Gender', 'City', 'Working Professional or Student', 'Profession', 
                       'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Display updated data structure
print("\nDataset After Encoding:")
print(data.head())

# Correlation heatmap to understand relationships
plt.figure(figsize=(12, 8))
# Select numerical columns
numerical_data = data.select_dtypes(include=[np.number])

# Calculate correlation matrix
corr_matrix = numerical_data.corr()

# Use corr_matrix for heatmap or further analysis
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
#sns.heatmap(data.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Define the target variable (e.g., 'Depression') and features
X = data.drop(['Depression', 'id', 'Name'], axis=1)
X.head()
y = data['Depression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a machine learning model
model = RandomForestClassifier(random_state=42)
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
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
