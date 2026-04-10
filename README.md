# Project 2 Instructions

## Logistics

- **Due Date:** Thursday, April 9th, 2026 at 10:00 PM *(see late submission policy at end of page)*
- **Group Size:** Groups of 3 students

> ⚠️ Do not split the project so each student only does a portion. Each student is expected to work on the entire project individually, then meet as a group to clarify doubts, share findings, and combine solutions into one group report. Quizzes and exams will test you individually on what you learned.

- Only **one submission per group**
- Help or assistance from other groups, other people, or online resources is **NOT allowed**

---

## Preparation

- Study **Chapter 3, Sections 4.10.1, 4.10.2, 4.10.6**, and **Appendix D** (online) of the textbook in great detail.
- Study all materials posted on the course Lecture Notes:
  - **Prediction:** Classification, Regression, and Evaluation of Models
  - Know the algorithms to construct **decision trees, regression trees, and model trees** very well — you must be able to construct trees from data **by hand** during quizzes and exams.
- **Thoroughly read and follow the Project Guidelines.**

> ⚠️ You must use the **Project 2 Template** provided. Do NOT change the structure, do NOT exceed page limits, and do NOT decrease the font size. (If you prefer not to use Word, copy the format into another editor while respecting the page structure and page limit.)

---

## Submission Instructions

Submit the following **3 files** via Canvas (submission name: `Project2`):

1. **PDF** — Written report using the provided template and Project Guidelines.
2. **.py or .ipynb** — All Python code written for each project part, organized neatly in order with comments.
3. **PDF** — Answers to the following 5 questions (for in-class discussion highlights):

   **(a)** Your name

   *Reflect on the course material covered in this project (think "big picture"):*

   **(b)** What are the top 3 things you learned about **data mining techniques** from working on this project?

   **(c)** What are the top 2 things you learned about the **data domain (stroke)** from uncovering patterns in this dataset?

   **(d)** What are the top 3 things you want to **learn more about** regarding the topics covered by this project?

   *Reflect on how you worked on this project:*

   **(e)** If you could start working on this project again, what would you do differently?

---

## Data Mining Techniques

Run all experiments in **Python** using the following techniques:

### Pre-processing

- Consider pre-processing techniques discussed in class, the textbook, and used in Project 1 (feature selection, feature creation, dimensionality reduction, noise reduction, attribute transformations, etc.)
- Determine which pre-processing techniques are **necessary** before mining predictive models. Start with the least pre-processing.
- Determine which techniques are **useful (but not necessary)** by running experiments with and without them, and comparing their effect on performance and readability.
- List all pre-processing performed in your report.

---

### Classification Techniques

**Target attribute:** `stroke`

#### Majority Class Classifier (Baseline) - HRITHIKA
- Use `DummyClassifier` from `sklearn.dummy`
- Parameters: `strategy='most_frequent'`, all others default

#### Decision Trees - HRITHIKA
- Use `DecisionTreeClassifier` from `sklearn.tree`
- Experiment with (one at a time, others at default):
  - `criterion`: `gini` (default) and `entropy`
  - `max_depth`: `None`, then determine a good value based on results
  - `min_samples_split`: default (`2`), then determine a good value based on results
 
#### Random Forests - SHRIYA
- Use `RandomForestClassifier` from `sklearn.ensemble`
- Experiment with (one at a time, others at default):
  - `n_estimators`: default (`100`), then determine a good value based on results
  - `max_depth`: `None`, then determine a good value based on results

---

### Regression Techniques

**Target attribute:** `bmi` *(remove instances where BMI = N/A)*

#### Majority Class Regressor (Baseline) - ANGELA
- Use `DummyRegressor` from `sklearn.dummy`
- Parameters: `strategy='mean'`, all others default

#### Linear Regression - ANGELA
- Use `LinearRegression` from `sklearn.linear_model`

#### Regression Trees - ANGELA
- Use `DecisionTreeRegressor` from `sklearn.tree`

#### Random Forests (Regression)
- Use `RandomForestRegressor` from `sklearn.ensemble`
- Experiment with (one at a time, others at default):
  - `n_estimators`: default (`100`), then determine a good value based on results
  - `max_depth`: `None`, then determine a good value based on results

---

## Dataset

Use the **[Stroke Prediction Dataset](https://www.kaggle.com/)** available on Kaggle.

| Task | Target Attribute | Notes |
|------|-----------------|-------|
| Classification | `stroke` | — |
| Regression | `bmi` | Remove instances where BMI = N/A |

---

## Performance Metrics

### Classification
- Classification accuracy
- Precision
- Recall
- ROC Area
- Confusion matrices

### Regression
- Correlation coefficient
- Any appropriate subset of:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Relative Squared Error (RSE)
  - Root Relative Squared Error (RRSE)
  - Relative Absolute Error (RAE)

### Additional Qualitative Criteria (for all models)
- **Size** of the tree
- **Readability** of the tree *(qualitative)*
- **Time** to construct the tree

> Compare all accuracy/error metrics against the appropriate **baseline (Majority Class) classifier or regressor** over the same subset of data instances.