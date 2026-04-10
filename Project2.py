import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, recall_score, 
                            accuracy_score, precision_score, recall_score, roc_auc_score,
                            confusion_matrix)
import time
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

# Classification Techniques

# Random Forests

X_cls = df.drop(columns=["stroke"])
y_cls = df["stroke"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split( X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

def evaluate_classification_model(model_name, model):
    start = time.time()
    model.fit(X_train_cls, y_train_cls)
    train_time = time.time() - start

    y_pred = model.predict(X_test_cls)
    y_prob = model.predict_proba(X_test_cls)[:, 1]

    print(f"\n{model_name}")
    print("Accuracy:", round(accuracy_score(y_test_cls, y_pred), 4))
    print("Precision:", round(precision_score(y_test_cls, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_test_cls, y_pred, zero_division=0), 4))
    print("Area under curve:", round(roc_auc_score(y_test_cls, y_prob), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_cls, y_pred))
    print("Time to construct in sec:", round(train_time, 4))

# baseline,  need this for comparison
dummy_clf = DummyClassifier(strategy="most_frequent")
evaluate_classification_model("Most FrequentDummy Classifier ", dummy_clf)

# default random forest 
rf_default = RandomForestClassifier(random_state=42)
evaluate_classification_model("Default Random Forest Classifier", rf_default)

# experiment 1: varied n_estimators 
for n in [100, 200, 300, 500]:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42)
    evaluate_classification_model(f"Random Forest Classifier n_estimators={n}", rf_n)

# experiment 2: varied  max_depth 
for depth in [None, 5, 10, 20]:
    rf_depth = RandomForestClassifier(max_depth=depth, random_state=42)
    evaluate_classification_model(f"Random Forest Classifier max_depth={depth}", rf_depth)