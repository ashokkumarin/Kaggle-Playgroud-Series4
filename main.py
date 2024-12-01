import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append ("mylib")

from visualization import *
from utility import *

type = 'classification'

"""
1. Read train and test dataset
2. Initial Cleanup
    - Remove unwanted columns from dataset
    - Check for nan in dataset and clean it up
"""
train_data = pd.read_csv('data\\train.csv')
test_data = pd.read_csv('data\\test.csv')

# train_data = train_data[:1000]
# test_data = test_data[:1000]

unwanted_col = {"id", "Name", "CGPA"}
for col in unwanted_col:
    train_data = train_data.drop([col], axis=1)
    test_data = test_data.drop([col], axis=1)

fill_dummy_values(train_data)
fill_dummy_values(test_data)


"""
Process categorical data set
--------------------------
1. Encode selected categorical variables
"""
categorical = train_data.select_dtypes(exclude=[np.number])
categorical = encode_categorical(categorical, categorical.columns)
categorical['Depression'] = train_data['Depression']
corr_matrix = categorical.corr()
# DrawHeatMap(corr_matrix)
print(corr_matrix['Depression'].sort_values(ascending=False)[:6])

numerical = train_data.select_dtypes(include=[np.number])

print(numerical.columns)
print(categorical.columns)

"""
Process numerical data set
--------------------------
1. Get correlation matrix
2. Get top 5 highly correlated predictors columns against 'Depression'
3. Remove outliers from training data
"""
corr_matrix = numerical.corr()
# DrawHeatMap(corr_matrix)  # Visualize correlation matrix
highly_corr_predictors = corr_matrix['Depression'].sort_values(ascending=False)[:6]
# print(f"Highly correlated predictors index: {highly_corr_predictors.index}"")
# DrawHeatMap(numerical[highly_corr_predictors.index].corr())

#skewed_feats = numerical.apply(lambda x: pd.DataFrame.skew(x.dropna())).sort_values(ascending=False)
skewed_feats = numerical.apply(lambda x: x.skew()).sort_values(ascending=False)
print(f"Skewed features: {skewed_feats}")


"""
Eliminate outliers from dataset
"""
# DrawScatterPlot(train_data['Work Pressure'],train_data['Depression'], 'Work Pressure', 'Depression')
# outliers = {'OverallQual': [2, '>'],
#             'GrLivArea': [4000, '<'],
#             'GarageCars': [4, '<'],
#             'GarageArea': [1200, '<'],
#             'TotalBsmtSF': [3000, '<'],
#             }
# train_data = remove_outliers(train_data, outliers)

# DrawScatterPlot(train_data['GarageArea'],train_data['SalePrice'], 'GarageArea', 'Sale Price')
# DrawPairPlot(train_data[highly_corr_predictors.index])



train_data = dichotomous_encode(train_data, categorical.columns)
test_data = dichotomous_encode(test_data, categorical.columns)
"""
Get training and test data for prediction
"""
# DrawHeatMap(train_data.corr())  # Visualize correlation matrix
# X_train = train_data[numerical.columns.drop('SalePrice')]  # [highly_corr_predictors.index.drop(['SalePrice'])]
# X_train['SaleCondition'] = train_data['SaleCondition']
# X_train['Street'] = train_data['Street']
# y_train = np.log(train_data.SalePrice)

# X_test = test_data[X_train.columns]

X = train_data[highly_corr_predictors.index.drop(['Depression'])]
X = train_data.drop(['Depression'], axis=1)
y = train_data['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

if (type != 'classification'): # Regression models
    """
    Initialize Regression Objects
    """

    print("\n### LINEAR ###")
    lin_reg = LinearRegression()
    MSEs = cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    mean_MSE = np.mean(MSEs)
    print("mean_MSE: ", mean_MSE)
    # print("Model score: ", lin_reg.score(X_test, y_test))

    print("\n### RIDGE ###")
    ridge = Ridge()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 200, 500, 1000]}
    ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    ridge_regressor.fit(X_train, y_train)
    print("ridge_regressor.best_params_: ", ridge_regressor.best_params_)
    print("ridge_regressor.best_score_: ", ridge_regressor.best_score_)

    ridge = Ridge(alpha=ridge_regressor.best_params_['alpha'])
    ridge_regressor = ridge.fit(X_train, y_train)


    print("\n### LASSO ###")
    lasso = Lasso()
    parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 50, 100, 200, 500, 1000]}  # {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
    lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
    lasso_regressor.fit(X_train, y_train)
    print("lasso_regressor.best_params_: ", lasso_regressor.best_params_)
    print("lasso_regressor.best_score_: ", lasso_regressor.best_score_)

    lasso = Lasso(alpha=lasso_regressor.best_params_['alpha'])
    lasso_regressor = lasso.fit(X_train, y_train)

    print("\n### REGRESSION TREE ###")
    reg_tree = DecisionTreeRegressor(max_depth=14)
    reg_tree.fit(X_train, y_train)

    # print(X_test.describe())

    """
    Evaluate prediction results
    """
    pred_df = pd.DataFrame()
    model_info = pd.DataFrame(index=['Score', 'Intercept', 'Co-efficient'])
    regressor = [lin_reg, ridge_regressor, lasso_regressor, reg_tree]
    df_cols = ["Linear", "Ridge", "Lasso", "Regression Tree"]

    X_pred = test_data[X_train.columns]
    for (reg, col) in zip(regressor, df_cols):
        print('\n### {:s} Regression ###'.format(col))
        reg.fit(X_train, y_train)
        scores = cross_val_score(reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        print("Score: ", reg.score(X_test, y_test))
        if col != "Regression Tree":
            model_info[col] = [reg.score(X_test, y_test), reg.intercept_, reg.coef_]
            print("Intercept: ", reg.intercept_)
            print("Co-efficient: ", reg.coef_)
        else:
            model_info[col] = [reg.score(X_test, y_test), 0, 0]
        y_pred = reg.predict(X_pred)
        pred_df[col] = y_pred
else: #Classification Model
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

    # Predict with actual test data
    X_pred = test_data[X_train.columns]
    y_pred = model.predict(X_pred)


print("Model Info\n", model_info)
model_info.to_csv("model.csv", index=False)
pred_df.to_csv("predicted.csv", index=False)

