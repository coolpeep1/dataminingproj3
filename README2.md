# Project 3 — Clustering & Anomaly Detection on Stroke Prediction Data

**Due:** Tuesday, April 28, 2026 at 10:00 PM  
**Group Size:** 3 students (same groups as Project 2)  
**Submission:** One submission per group via Canvas (`Project3`)

---

## Dataset

[Stroke Prediction Dataset](https://www.kaggle.com/) — available on Kaggle.

---

## Submission Requirements

Submit **3 files** on Canvas:

| File | Description |
|------|-------------|
| `report.pdf` | Written report using the provided Project 3 Template (no appendices, no font-size changes, no page-limit violations) |
| `code.py` or `code.ipynb` | All Python code, organized by project part with comments |
| `highlights.pdf` | Answers to the 5 highlight questions listed below |

---

## Highlight Questions (highlights.pdf)

**(a)** Your names  
**(b)** Top 3 things learned about data mining techniques from this project  
**(c)** Top 2 things learned about the stroke domain from this project  
**(d)** Top 3 things you want to learn more about  
**(e)** What you would do differently if starting over  

---

## Project Parts

### 1. Pre-processing

- Apply the minimum necessary pre-processing to make the data usable (less is more at first).
- Experiment with additional pre-processing (e.g., feature selection, dimensionality reduction, transformations) to improve clustering quality — compare results with and without.
- **Scale all continuous attributes** to the same range to avoid distance distortion.

---

### 2. Clustering

Use [scikit-learn's clustering library](https://scikit-learn.org/stable/modules/clustering.html).

#### K-Means
- Plot SSE for k = 1, 2, 3, 4, 5, ... to select the best k (elbow method).
- [`sklearn.cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

#### Hierarchical Clustering
- Experiment with **ward**, **complete**, **average**, and **single** linkage.
- Compare k-cluster cuts from dendrograms against k-means results.
- [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

#### DBSCAN
- Tune `epsilon` and `min_samples` using a k-distance plot approach.
- [`sklearn.cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

### 3. Anomaly Detection (for each of the 3 methods)

For each clustering method:
1. Select one clustering from your experiments.
2. Define a sound **anomaly score `f(x)`** based on that method.
3. Apply `f(x)` to every data instance.
4. Determine whether outliers are present (via clusters or scores).
5. Analyze what stroke-domain characteristics make identified outliers unusual.
6. Illustrate with plots/visualizations.
7. Compare: do all three methods identify the same outliers?

---

### 4. Clustering Evaluation

#### Visualization
- MDS, t-SNE, UMAP, or other dimensionality-reduction-based visualizations of clusters.

#### Internal Indices *(no external labels needed)*
- **SSE** (Sum of Squared Errors)
- **Heatmap** of the correlation between the distance/proximity matrix and the incidence matrix
- **Silhouette Coefficient** — [`silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

#### Relative Indices *(compare two clusterings)*
Compare two clusterings of your choice (justify your selection). Options: same method/different params, or different methods.
- SSE
- **Adjusted Rand Score** — [`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
- **Normalized Mutual Information** — [`normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html)
- **Adjusted Mutual Information** — [`adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html)

#### External Indices *(compare clustering against a ground-truth label)*
Choose a target attribute (e.g., `stroke`, discretized BMI) — **do not include it as a clustering input**. Justify your choice.
- **Homogeneity** — [`homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html)
- **Completeness** — [`completeness_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html)
- **V-Measure** — [`v_measure_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html)
- **Contingency Matrix** — [`contingency_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html)

---

## Academic Integrity

Collaboration is limited to your group of 3. Help from other groups, unauthorized AI tools, or online resources is **not allowed**.

---

## Late Submission

See the course page for the late submission policy.
