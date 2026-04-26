# Project 3

## Due Date & Group Info

Due date: This project is due on Tuesday, April 28th, 2026 at 10:00 pm. (See late submission policy for this project at the end of this page.)

Group Project: This is a group project. Students must work in groups of 3 students.

I expect that you work with the same group as in Project 2, but please let me know well in advance if you plan/need to change groups.

Please do not split the project in a way that each student does only a portion of the work. Instead each student is expected to work on the entire project individually and then meet with the group to clarify doubts, share findings, and combine the project solutions into one group report. The quizzes and exams will test you individually on what you learned on this Project.

Only one submission per group (see a description of the required submission files below).

Help or assistance from other groups, other people, unauthorized AI or online resources is NOT allowed.

---

## Study Materials

Study in great detail the following Chapters and Sections from the Textbook:
- Sections 2.4 on "Measures of Similarity and Dissimilarity" (except for Sections 2.4.7 and 2.4.8).
- Chapter 7 on "Cluster Analysis: Basic Concepts and Algorithms".

Study all the materials posted on the course Lecture Notes:
- Clustering
- Anomaly Detection using clustering-based methods

In particular, you should know the definitions of distance and similarity measures, and the algorithms for clustering data including k-means, hierarchical clustering, and density clustering very well, and be able to use these algorithms to cluster data "by hand" during quizzes and exams. Same comment applies to techniques to evaluate resulting clusterings, and to determine data outliers using clustering techniques.

---

## Report Instructions

THOROUGHLY READ AND FOLLOW THE PROJECT GUIDELINES. These guidelines contain detailed information about how to structure your project, how to prepare your written report and how to prepare your project highlights for in-class discussion.

*** You must use the Project 3 Template provided here for your written report. Do NOT change the structure of the report, do NOT exceed the page limits stated in the template and do NOT decrease the font size ***. Your whole report must fit within the page limit. *No* appendices allowed. Analyze your group results together and select as a group only the most significant results and analysis to include in the report. (If you prefer not to use Word, you can copy and paste this format in a different editor as long as you respect the stated page structure and page limit.)

---

## Project Submission Instructions

Each group must submit the following 3 files using the Canvas system (submission name: Project3). Note: it is the responsibility of each teammate to make sure all the correct files are submitted by the deadline. Late penalty applies to the whole team is a project file is not submitted by the deadline.

1. one pdf file with your project written report, using the template provided and following the instructions on the Project Guidelines.
2. one .py or .ipynb file containing all the Python code that you wrote to perform each of the project parts below. Please organize your Python code neatly in the order of the project parts below with comments that help us understand your work.
3. one pdf file with your answers to the following 5 questions summarizing your highlights for this project (which we'll use for this project's in-class discussion):

**(a)** Your names

Reflect on the content of (i.e., the course material covered in) this project. Try to think "big picture" in answering these questions:

**(b)** What are the top 3 things you learned about data mining techniques from working on this project?

**(c)** What are the top 2 things you learned about the data domain (i.e., stroke) from this project?

**(d)** What are the top 3 things you want to learn more about the topic covered by this project?

Reflect now on how you worked on this project:

**(e)** If you could start working on this project again, what would you do differently?

---

## Project Assignment

### Dataset

Use the Stroke Prediction Dataset available at Kaggle.

### Data Mining Technique(s)

Run all project experiments in Python, using the following techniques:

---

### Pre-processing Techniques

- Consider the pre-processing techniques (feature selection, feature creation, dimensionality reduction, noise reduction, attribute transformations, ...) discussed in class, the textbook and used in previous projects.
- Determine which pre-processing techniques are necessary to pre-process the given dataset before you can mine predictive (either classification or regression) models from this data. The least pre-processing at first, the better. List the necessary pre-processing you performed in your report.
- Determine which pre-processing techniques would be useful (though not necessary) for this dataset in order to construct better clusterings. Do this by running experiments with and without applying these pre-processing techniques and comparing how they affect the clustering results.
- Remember that it is a good idea in general to scale continuous attributes so that they are all in the same range so that an attribute with a large range doesn't have a disproportionate influence on the distance/similarity measurement.

---

### Clustering Techniques - HRITHIKA

Use the scikit learn clustering library: https://scikit-learn.org/stable/modules/clustering.html. Note that this page contains very useful illustrations and overviews of the clustering methods, worth reading!

**K-means clustering:**
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

**Hierarchical clustering:** Experiment with "ward", "complete", "average", and "single" linkage:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

**DBSCAN clustering:** Run experiments to find good parameter values.
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN

**General Comments about Clustering:**
- Focus on experimenting with varying the parameters of the clustering algorithms, and providing in-depth evaluation and interpretation of the results.
- For k-means and for density-based-clustering, work on determining good parameter values using the methods described in class: For k-means, plot SSE values of clusterings with k = 1, 2, 3, 4, 5, ... to pick a good value for k. For density-based-clustering, use a similar approach to find good values for the epsilon and min-points parameters.
- Once that you determine the best k value for k-means, compare the k-clustering obtained by k-means with the k-clusterings that can be derived from the hierarchical clustering dendrograms.

---

### Anomaly Detection using Cluster-based methods - SHRIYA

For each of the 3 clustering methods used in this project (k-means, hierarchical, and DBSCAN):
- Select a clustering obtained in one of your experiments.
- Define a sound anomaly score, f(x), based on the cluster method.
- Apply the anomaly score, f(x), to each data instance x in the dataset.
- Determine whether the selected clustering identified the presence of outliers, either based on the resulting clusters themselves or the anomaly score, f(x).
- If outliers are identified, take a close look at them and analyze what characteristics from the dataset domain (i.e., stroke) make them outliers.
- Explain your answers, ideally illustrating with plots and/or visualizations.
- Elaborate on whether the same outliers, if any, were identified by two or all three of the clustering methods.

---

### Clustering Evaluation - ANGELA

A major part of this project is to find meaningful ways of evaluating and interpreting the resulting clusters. Devise a variety of approaches to do so, including but not limited to:

- visualization (e.g., MDS, t-SNE, UMAP and/or others) of the resulting clusters (Python provides a good number of visualization functions);
- inspection of the actual clusters' members to find similarities among data instances in a cluster and dissimilarities with data instances in different clusters; and
- evaluation using the clustering-specific performance metrics/functions described in the textbook and/or https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation (this page contains very useful performance evaluation information, worth reading). Include the following metrics:

**Internal Indices:** These indices evaluate a clustering in terms of itself without external information.
- Sum of Squared Errors (SSE).
- Heatmap of the correlation between the distance (or proximity) matrix and the incidence matrix (as described in the textbook and slides).
- Silhouette Coefficient: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score

**Relative Indices:** These indices are used to compare two different clusterings of a dataset. These two clustering can be the result of using the same clustering method with different parameters (e.g., comparing two k-means clusterings obtained using a different random seed; or comparing a 4-cluster clustering obtained from a single-link dendrogram against a 4-cluster clustering obtained from a complete-link dendrogram); or using two different clustering methods (e.g., comparing a k-means clustering against a 4-cluster clustering obtained from a Ward-link dendrogram).

You can choose what pairs of clusterings to compare. Justify your choices in your written report.

- Sum of Squared Errors (SSE).
- Adjusted Rand score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score
- Normalized mutual information score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
- Adjusted mutual information score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score

**External Indices:** These indices are used to compare a clustering with respect to a target attribute given in the dataset (e.g., stroke or discretized BMI).

You can choose what clusterings and what data attributes to use as labels in this comparison. Justify your choices in your written report. Make sure not to include the attribute that you are using as label among the input attributes used for clustering.

- Homogeneity, completeness and V-measure:
  - Homogeneity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
  - Completeness: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
  - V-measure: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
- Contingency Matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix

Note: Some of the indices above listed under "Relative Indices" can also be used as "External Indices" and some of the indices listed under "External Indices" can also be used as "Relative Indices". This is because a (partitional) clustering of the data can be interpreted as a labeling of the data; and a labeling (classification) of the data can be interpreted as a (partitional) clustering of the data.

The deeper your analysis, the better your project grade. You may consider implementing in Python evaluation/interpretation functionality that you need but doesn't currently exist in Python's libraries.
