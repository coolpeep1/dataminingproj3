import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df.head())

# Pre-processing Techniques:

# remove the id column
df = df.drop(columns=["id"])
# fill missing values in the bmi column with the mean
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
# converted categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Classification Techniques

# preprocessing the data for classification - removed id and converted categorical variables to numbers
df_clf = pd.read_csv("healthcare-dataset-stroke-data.csv")
df_clf = df_clf.drop(columns=["id"])
df_clf = pd.get_dummies(df_clf, drop_first=True)

# features and target
X_clf = df_clf.drop(columns=["stroke"])
y_clf = df_clf["stroke"]

# split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, random_state=42)

# DummyClassifier with the most frequent strategy
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = dummy_clf.predict(X_test_clf)

print("\nDummy Classifier (Most Frequent) Results:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Precision:", precision_score(y_test_clf, y_pred_clf))
print("Recall:", recall_score(y_test_clf, y_pred_clf))

# decision Tree Classifier
tree_clf = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_clf.fit(X_train_clf, y_train_clf)
y_pred_tree_clf = tree_clf.predict(X_test_clf)

print("\nDecision Tree (Gini)) Results:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_tree_clf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_tree_clf))

# decision Tree Classifier with the entropy criterion
tree_clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_clf_entropy.fit(X_train_clf, y_train_clf)
y_pred_clf_entropy = tree_clf_entropy.predict(X_test_clf)

print("\nDecision Tree (Entropy) Results:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf_entropy))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf_entropy))

# max_depth

# Regression Techniques

# Majority Class Classifier:

# pre-processing the data for regression - reomved id, dropped rows with missing bmi values, and converted categorical variables to numbers
df_reg = pd.read_csv("healthcare-dataset-stroke-data.csv")
df_reg = df_reg.drop(columns=["id"])
df_reg = df_reg.dropna(subset=["bmi"])
df_reg = pd.get_dummies(df_reg, drop_first=True)

# X is the feature set and y is the target variable
X = df_reg.drop(columns=["bmi"])
y = df_reg["bmi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# predicts the average bmi value for all instances in the test set
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
# makes predictions on the data
y_pred = dummy_regr.predict(X_test)

# calculates the RMSE and MAE for the Dummy Regressor
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# prints results
print("\nDummy Regressor Results:")
print("RMSE:", rmse)
print("MAE:", mae)

# Linear Regression:

reg = LinearRegression()
reg.fit(X_train, y_train)

# makes predictions on the dataset
y_pred = reg.predict(X_test)

# calculates the RMSE and MAE 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
# calculates correlation coefficient
correlation = np.corrcoef(y_test, y_pred)[0, 1]

# print results
print("\nLinear Regression Results:")
print("RMSE:", rmse)
print("MAE:", mae)
print("Correlation Coefficient:", correlation)

# Regression Trees:

tree_reg = DecisionTreeRegressor(random_state=42)
# makes predictions using 10-fold cross-validation
y_pred = cross_val_predict(tree_reg, X, y, cv=10)

# calculates the RMSE and MAE 
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
# calculates correlation coefficient
correlation = np.corrcoef(y, y_pred)[0, 1]

# print results
print("\nDecision Tree Regressor Results (10-fold CV):")
print("RMSE:", rmse)
print("MAE:", mae)
print("Correlation Coefficient:", correlation)