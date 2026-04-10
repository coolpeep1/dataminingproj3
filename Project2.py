import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, recall_score, classification_report,
                             accuracy_score, precision_score, recall_score, roc_auc_score,
                            confusion_matrix)
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
print("\nDecision Tree with max_depth Experiments:")
depth_values = [None, 3, 5, 10]

for depth in depth_values:
    tree_clf_depth = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree_clf_depth, X_clf, y_clf, cv=10, scoring="accuracy")
    print(f"Max Depth={depth}, Accuracy={scores.mean():.4f}")
    
# min_samples_split
print("\nDecision Tree with min_samples_split Experiments:")
split_values = [2, 5, 10]

for split in split_values:
    tree_clf_split = DecisionTreeClassifier(min_samples_split=split, random_state=42)
    scores = cross_val_score(tree_clf_split, X_clf, y_clf, cv=10, scoring="accuracy")
    print(f"Min Samples Split={split}, Accuracy={scores.mean():.4f}")

# random forests

X_rf = df.drop(columns=["stroke"])
y_rf = df["stroke"]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
def evaluate_rf_classification(model_name, model):
    start = time.time()
    y_pred = cross_val_predict(model, X_rf, y_rf, cv=cv, method="predict")
    y_prob = cross_val_predict(model, X_rf, y_rf, cv=cv, method="predict_proba")[:, 1]
    build_time = time.time() - start

    # fit once on full data to report model size
    model.fit(X_rf, y_rf)
    num_trees = len(model.estimators_)
    avg_nodes = sum(tree.tree_.node_count for tree in model.estimators_) / num_trees

    print(f"\n{model_name}")
    print("Accuracy:", round(accuracy_score(y_rf, y_pred), 4))
    print("Precision:", round(precision_score(y_rf, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_rf, y_pred, zero_division=0), 4))
    print("ROC AUC:", round(roc_auc_score(y_rf, y_prob), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_rf, y_pred))
    print("Time to construct (sec):", round(build_time, 4))
    print("Number of trees:", num_trees)
    print("Average nodes/tree:", round(avg_nodes, 2))

# baseline comparison
dummy_clf = DummyClassifier(strategy="most_frequent")
evaluate_rf_classification("most frequent Dummy Classifier ", dummy_clf)

#   experiments
rf_default = RandomForestClassifier(random_state=42)
evaluate_rf_classification("Random Forest - Default", rf_default)
rf_n200 = RandomForestClassifier(n_estimators=200, random_state=42)
evaluate_rf_classification("Random Forest - n_estimators=200", rf_n200)
rf_n300 = RandomForestClassifier(n_estimators=300, random_state=42)
evaluate_rf_classification("Random Forest - n_estimators=300", rf_n300)
rf_depth5 = RandomForestClassifier(max_depth=5, random_state=42)
evaluate_rf_classification("Random Forest - max_depth=5", rf_depth5)
rf_depth10 = RandomForestClassifier(max_depth=10, random_state=42)
evaluate_rf_classification("Random Forest - max_depth=10", rf_depth10)

# Regression Techniques

# Dummy Regressor:

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
dummy_y_pred = dummy_regr.predict(X_test)

# calculates the RMSE and MAE for the Dummy Regressor
dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_y_pred))
dummy_mae = mean_absolute_error(y_test, dummy_y_pred)

# prints results
print("\nDummy Regressor Results:")
print("RMSE:", dummy_rmse)
print("MAE:", dummy_mae)

# Linear Regression:

linear_reg = LinearRegression()

# makes predictions on the dataset
linear_y_pred = cross_val_predict(linear_reg, X, y, cv=10)

# calculates the RMSE and MAE 
linear_rmse = np.sqrt(mean_squared_error(y, linear_y_pred))
linear_mae = mean_absolute_error(y, linear_y_pred)
# calculates correlation coefficient
linear_correlation = np.corrcoef(y, linear_y_pred)[0, 1]

# print results
print("\nLinear Regression Results:")
print("RMSE:", linear_rmse)
print("MAE:", linear_mae)
print("Correlation Coefficient:", linear_correlation)

# Experiment 2 - Linear Regression with Scaling:
linear_reg_scaled = make_pipeline(StandardScaler(), LinearRegression())

linear_y_pred_scaled = cross_val_predict(linear_reg_scaled, X, y, cv=10)

linear_rmse_scaled = np.sqrt(mean_squared_error(y, linear_y_pred_scaled))
linear_mae_scaled = mean_absolute_error(y, linear_y_pred_scaled)
linear_corr_scaled = np.corrcoef(y, linear_y_pred_scaled)[0, 1]

print("\nLinear Regression with Scaling Results:")
print("RMSE:", linear_rmse_scaled)
print("MAE:", linear_mae_scaled)
print("Correlation Coefficient:", linear_corr_scaled)

# Regression Trees:

tree_reg = DecisionTreeRegressor(random_state=42)
# makes predictions using 10-fold cross-validation
tree_y_pred = cross_val_predict(tree_reg, X, y, cv=10)

# calculates the RMSE and MAE 
tree_rmse = np.sqrt(mean_squared_error(y, tree_y_pred))
tree_mae = mean_absolute_error(y, tree_y_pred)
# calculates correlation coefficient
tree_correlation = np.corrcoef(y, tree_y_pred)[0, 1]

# print results
print("\nDecision Tree Regressor Results (10-fold CV):")
print("RMSE:", tree_rmse)
print("MAE:", tree_mae)
print("Correlation Coefficient:", tree_correlation)

# Experiment 2: Decision Tree (max_depth = 5)
tree_reg_5 = DecisionTreeRegressor(max_depth=5, random_state=42)

tree_y_pred = cross_val_predict(tree_reg_5, X, y, cv=10)

print("\nDecision Tree (max_depth=5):")
print("RMSE:", np.sqrt(mean_squared_error(y, tree_y_pred)))
print("MAE:", mean_absolute_error(y, tree_y_pred))
print("Correlation:", np.corrcoef(y, tree_y_pred)[0, 1])

# Experiment 3: Decision Tree (max_depth = 10)
tree_reg_10 = DecisionTreeRegressor(max_depth=10, random_state=42)

tree_y_pred = cross_val_predict(tree_reg_10, X, y, cv=10)

print("\nDecision Tree (max_depth=10):")
print("RMSE:", np.sqrt(mean_squared_error(y, tree_y_pred)))
print("MAE:", mean_absolute_error(y, tree_y_pred))
print("Correlation:", np.corrcoef(y, tree_y_pred)[0, 1])

# Experiment 4: Decision Tree (max_depth = 20)
tree_reg_20 = DecisionTreeRegressor(max_depth=20, random_state=42)

tree_y_pred = cross_val_predict(tree_reg_20, X, y, cv=10)

print("\nDecision Tree (max_depth=20):")
print("RMSE:", np.sqrt(mean_squared_error(y, tree_y_pred)))
print("MAE:", mean_absolute_error(y, tree_y_pred))
print("Correlation:", np.corrcoef(y, tree_y_pred)[0, 1])

# random forest regressor:

X_reg_rf = df_reg.drop(columns=["bmi"])
y_reg_rf = df_reg["bmi"]
cv_reg = KFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_rf_regression(model_name, model):
    start = time.time()
    y_pred = cross_val_predict(model, X_reg_rf, y_reg_rf, cv=cv_reg, n_jobs=-1)
    build_time = time.time() - start
    rmse = np.sqrt(mean_squared_error(y_reg_rf, y_pred))
    mae = mean_absolute_error(y_reg_rf, y_pred)
    corr = np.corrcoef(y_reg_rf, y_pred)[0, 1]
    model.fit(X_reg_rf, y_reg_rf)
    num_trees = len(model.estimators_)
    avg_nodes = sum(tree.tree_.node_count for tree in model.estimators_) / num_trees

    print(f"\n{model_name}")
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("Correlation Coefficient:", round(corr, 4))
    print("Time to construct (sec):", round(build_time, 4))
    print("Number of trees:", num_trees)
    print("Average nodes per tree:", round(avg_nodes, 2))

# baseline comparison
dummy_regr = DummyRegressor(strategy="mean")
evaluate_rf_regression("Dummy Regressor (Mean)", dummy_regr)

# random rf experiments
rf_default = RandomForestRegressor(random_state=42)
evaluate_rf_regression("Random Forest Regressor - Default", rf_default)

rf_n200 = RandomForestRegressor(n_estimators=200, random_state=42)
evaluate_rf_regression("Random Forest Regressor - n_estimators=200", rf_n200)

rf_n300 = RandomForestRegressor(n_estimators=300, random_state=42)
evaluate_rf_regression("Random Forest Regressor - n_estimators=300", rf_n300)

rf_depth5 = RandomForestRegressor(max_depth=5, random_state=42)
evaluate_rf_regression("Random Forest Regressor - max_depth=5", rf_depth5)

rf_depth10 = RandomForestRegressor(max_depth=10, random_state=42)
evaluate_rf_regression("Random Forest Regressor - max_depth=10", rf_depth10)