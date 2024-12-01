# Kaggle-Playgroud-Series4
Kaggle Competition - Playground Series - Season 4 - Exploring Mental Health Data

---

# Machine Learning Model for Depression Prediction

This project aims to predict whether an individual is experiencing depression based on various features in the dataset. The solution uses machine learning models and includes steps for data preprocessing, feature engineering, and model evaluation.

---

## **Installation and Requirements**
To run the project, you need the following dependencies installed:
- **Python 3.8+**
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Directory Structure**
- **`data/`**: Contains the `train.csv` and `test.csv` datasets.
- **`mylib/`**: Contains custom utility scripts `visualization.py` and `utility.py` for data processing and visualization.

---

## **Code Overview**
The code consists of the following main steps:

### **1. Import Libraries**
Imports standard libraries and custom utility scripts for data processing and visualization.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from utility import *
from visualization import *
```

---

### **2. Load Data**
Loads the training and testing datasets from the `data/` directory and initializes the output dataframe.
```python
train_data = pd.read_csv('data\\train.csv')
test_data = pd.read_csv('data\\test.csv')

pred_df = pd.DataFrame()
pred_df['id'] = test_data['id']
```

---

### **3. Data Preprocessing**
- Drops unwanted columns (`id` and `Name`).
- Fills missing values using the custom utility function `fill_dummy_values`.
- Encodes categorical features using one-hot encoding via the custom function `encode_categorical`.

```python
# Drop unwanted columns
unwanted_col = {"id", "Name"}
for col in unwanted_col:
    train_data = train_data.drop([col], axis=1)
    test_data = test_data.drop([col], axis=1)

# Fill missing values and encode categorical columns
fill_dummy_values(train_data)
fill_dummy_values(test_data)
categorical_columns = train_data.select_dtypes(exclude=[np.number])
train_data = encode_categorical(train_data, categorical_columns)
test_data = encode_categorical(test_data, categorical_columns)
```

---

### **4. Feature Selection**
Defines the target variable (`Depression`) and selects the features (`X`).
```python
X = train_data.drop(['Depression'], axis=1)
y = train_data['Depression']
```

---

### **5. Train-Test Split**
Splits the dataset into training and testing sets.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

---

### **6. Model Training**
Uses a machine learning model to train on the training data. The following models are supported (uncomment the desired model to use it):
- **Gradient Boosting Classifier** (default model)
- Decision Tree
- Logistic Regression
- Random Forest
- Support Vector Machines (SVC)
- K-Nearest Neighbors
- Gaussian Naive Bayes

```python
# Uncomment the desired model
# model = RandomForestClassifier(random_state=42)
# model = DecisionTreeClassifier(random_state=42, max_depth=6)
# model = LogisticRegression()
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model = SVC(kernel='linear')
# model = KNeighborsClassifier(n_neighbors=15)
# model = GaussianNB()

model.fit(X_train, y_train)
```

---

### **7. Model Evaluation**
Evaluates the model on the test set and prints key performance metrics:
- **Accuracy**
- **Classification Report**
- **Confusion Matrix**
```python
y_pred = model.predict(X_test)

print("Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

### **8. Predictions on Test Data**
Generates predictions for the actual test dataset and saves them to a CSV file named `submission.csv`.
```python
X_pred = test_data[X_train.columns]
y_pred = model.predict(X_pred)

pred_df['Depression'] = y_pred
pred_df.to_csv("submission.csv", index=False)
print("Prediction file has been created")
```

---

## **Features**
The dataset contains the following features:
- **Categorical Columns**: Example - `Gender`, `City`, `Profession`
- **Numerical Columns**: Example - `CGPA`, `Sleep Duration`, `Study Satisfaction`

---

## **Output**
- **Model Performance**: Accuracy, Classification Report, and Confusion Matrix.
- **Predicted File**: A CSV file (`submission.csv`) containing predictions for the test dataset.

---

## **How to Use**
1. Place the training (`train.csv`) and testing (`test.csv`) datasets in the `data/` directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. The predictions will be saved in `submission.csv`.

---

## **Custom Functions**
The following utility functions are used:
- **`fill_dummy_values()`**: Fills missing values in the dataset.
- **`encode_categorical()`**: Encodes categorical features using one-hot encoding.

---

## **Future Enhancements**
- Add hyperparameter tuning for model optimization.
- Support additional models like deep learning (e.g., neural networks).
- Automate feature importance visualization.

---

Let me know if you need further modifications or additional sections for the README! 😊
