Due date: This project is due on Thursday, April 9th, 2026 at 10:00 pm. (See late submission policy for this project at the end of this page.)
Group Project: This is a group project. Students must work in groups of 3 students.
Please do not split the project in a way that each student does only a portion of the work. Instead each student is expected to work on the entire project individually and then meet with the group to clarify doubts, share findings, and combine the project solutions into one group report. The quizzes and exams will test you individually on what you learned on this Project.
Only one submission per group (see a description of the required submission files below).
Help or assistance from other groups, other people, or online resources is NOT allowed.
Study Chapter 3, Sections 4.10.1, 4.10.2, 4.10.6 and Appendix D (online)Links to an external site. of the textbook in great detail.
Study all the materials posted on the course Lecture Notes:
Prediction: which includes Classification, Regression, and Evaluation of Models
In particular, you should know the algorithms to construct decision trees, regression trees, and model trees very well, and be able to use these algorithms to construct trees from data by hand during quizzes and exams. 
THOROUGHLY READ AND FOLLOW THE PROJECT GUIDELINES. These guidelines contain detailed information about how to structure your project, how to prepare your written report and how to prepare your project highlights for in-class discussion.
*** You must use the Project 2 Template Download Project 2 Templateprovided here for your written report. Do NOT change the structure of the report, do NOT exceed the page limits stated in the template and do NOT decrease the font size ***. (If you prefer not to use Word, you can copy and paste this format in a different editor as long as you respect the stated page structure and page limit.)
Project Submission Instructions: Each group must submit the following 3 files using the Canvas system (submission name: Project2):
one pdf file with your project written report, using the template provided and following the instructions on the Project Guidelines.
one .py or . ipynb file containing all the Python code that you wrote to perform each of the project parts below. Please organize your Python code neatly in the order of the project parts below with comments that help us understand your work.
one pdf file with your answers to the following 5 questions summarizing your highlights for this project (which we'll use for this project's in-class discussion): 
(a) Your name

Reflect on the content of (i.e., the course material covered in) this project. Try to think "big picture" in answering these questions:

(b) What are the top 3 things you learned about data mining techniques from working on this project?

(c) What are the top 2 things you learned about the data domain (i.e., stroke) from uncovering patterns from this dataset? 

(d) What are the top 3 things you want to learn more about the topic covered by this project?
Reflect now on how you worked on this project:
(e) If you could start working on this project again, what would you do differently?
Data Mining Technique(s): Run all project experiments in Python, using the following techniques:
Pre-processing Techniques:
Consider the pre-processing techniques (feature selection, feature creation, dimensionality reduction, noise reduction, attribute transformations, ...) discussed in class, the textbook and used in project 1.
Determine which pre-processing techniques are necessary to pre-process the given dataset before you can mine predictive (either classification or regression) models from this data. The least pre-processing at first, the better. List the necessary pre-processing you performed in your report.
Determine which pre-processing techniques would be useful (though not necessary) for this dataset in order to construct better prediction models. Do this by running experiments with and without applying these pre-processing techniques and comparing how they affect the performance and readability of the prediction models.
Classification Techniques: 
Majority Class Classifier:
Use the DummyClassifierLinks to an external site. class from the sklearn.dummy module with default parameters except for setting strategy=’most_frequent’ so that it will predict the most frequent class in the training data.
Decision Trees:
Use the DecisionTreeClassifierLinks to an external site. class from the sklearn.tree module. Experiment with the following parameters (one at a time, leaving the others with their default values):
criterion: experiment with gini (default value) and with entropy.
max_depth: experiment with None, and then based on your analysis of the results, determine a good value to use for this parameter.
min_samples_split, experiment with the default=2, and then based on your analysis of the results, determine a good value to use for this parameter.
Random Forests:
Use the RandomForestClassifierLinks to an external site. class from the sklearn.ensemble module. Experiment with the following parameters (one at a time, leaving the others with their default values):
n_estimators: experiment with default value (100), and then based on your analysis of the results, determine a good value to use for this parameter.
max_depth: experiment with None, and then based on your analysis of the results, determine a good value to use for this parameter.
Regression Techniques:
Majority Class Classifier:
Use the DummyRegressorLinks to an external site. class from the sklearn.dummy module with default parameters except for setting strategy=’most_frequent’ so that it will predict the average of the target attribute  in the training data.
Linear Regression:
Use the LinearRegressionLinks to an external site. object model from the sklearn.linear_model module
Regression Trees:
Use the DecisionTreeRegressorLinks to an external site. from the sklearn.tree module
Random Forests: again, now for Regression instead of Classification
Use the RandomForestRegressorLinks to an external site. class from the sklearn.ensemble module. Experiment with the following parameters (one at a time, leaving the others with their default values):
n_estimators: experiment with default value (100), and then based on your analysis of the results, determine a good value to use for this parameter.
max_depth: experiment with None, and then based on your analysis of the results, determine a good value to use for this parameter.
Dataset:
Use the Stroke Prediction DatasetLinks to an external site. available at KaggleLinks to an external site..
For classification tasks: Use the stroke attribute as the target attribute.
For regression tasks: Use the BMI attribute as the regression target. For this part of the project, remove the data instances with BMI = N/A.
Performance Metric(s):
Use the following metrics or evaluation methods:
For classification tasks: use classification accuracy, precision, recall, ROC Area, and confusion matrices.
For regression tasks: use correlation coefficient AND any subset of the following error metrics that you find appropriate: mean-squared error, root mean-squared error, mean absolute error, relative squared error, root relative squared error, and relative absolute error. An important part of the data mining evaluation in this project is to try to make sense of these performance metrics and to become familiar with them.
size of the tree,
readability of the tree, as a separate qualitative criterion to evaluate the "goodness" of your models, and
time it took to construct the tree.
Compare each accuracy/error you obtained against those of benchmarking techniques as the Majority Class classifier over the same (sub-)set of data instances you used in the corresponding experiment.
