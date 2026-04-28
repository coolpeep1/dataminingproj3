import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
import seaborn as sns
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

# Pre-processing Techniques

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
y = df["stroke"]

# Removed the id column
df = df.drop(columns=["id"])

# Filled missing values in the bmi column with the mean
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Removed stroke from clustering input
X = df.drop(columns=["stroke"])

# Converted columns with text to numbers
X = pd.get_dummies(X, drop_first=True)

# Scaled data using StandardScaler to ensure that all features contribute equally to the clustering process
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering Techniques

# K-Means: find best k using elbow method for sum of squared errors (SSE)
sse = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=4, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
    
plt.plot(k_values, sse, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for K-Means")
plt.show()

# choose best k based on elbow
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=4, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# K-means anomaly detection
kmeans_distances = np.linalg.norm(
    X_scaled - kmeans.cluster_centers_[kmeans_labels], axis=1)

distance_frame = pd.DataFrame(
    {"cluster": kmeans_labels, "distance": kmeans_distances})
cluster_rms = distance_frame.groupby("cluster")["distance"].transform( lambda values: np.sqrt(np.mean(np.square(values))))
kmeans_anomaly_score = kmeans_distances / (cluster_rms + 1e-12)

anomaly_df = df.copy()
anomaly_df["cluster"] = kmeans_labels
anomaly_df["anomaly_score"] = kmeans_anomaly_score
top_n = 10
top_outliers = anomaly_df.sort_values("anomaly_score", ascending=False).head(top_n)
threshold = np.quantile(kmeans_anomaly_score, 0.99)
anomaly_df["is_outlier"] = anomaly_df["anomaly_score"] >= threshold

print("\nK-means anomaly score summary:")
print(anomaly_df["anomaly_score"].describe())
print(f"K-means outlier threshold (99th percentile): {threshold:.4f}")
print("Number of flagged outliers:", anomaly_df["is_outlier"].sum())
print(f"\nTop {top_n} K-means outliers:")
print(top_outliers[["cluster", "anomaly_score", "stroke"]])

# Hierarchical Clustering
linkage_methods = ["ward", "complete", "average", "single"]
hierarchical_labels = {}

for linkage in linkage_methods:
    model = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    hierarchical_labels[linkage] = labels
    
    score = silhouette_score(X_scaled, labels)
    print(f"{linkage} silhouette:", score)
    
# choose best linkage method based on silhouette score
hierarchical_labels = hierarchical_labels["ward"]

# Hierarchical anomaly detection
# Compute cluster centers for hierarchical clustering
hierarchical_centers = np.vstack([X_scaled[hierarchical_labels==c].mean(axis=0) for c in sorted(np.unique(hierarchical_labels))])
hierarchical_distances = np.linalg.norm(
    X_scaled - hierarchical_centers[hierarchical_labels], axis=1)

hier_distance_frame = pd.DataFrame(
    {"cluster": hierarchical_labels, "distance": hierarchical_distances})
hier_cluster_rms = hier_distance_frame.groupby("cluster")["distance"].transform(
    lambda values: np.sqrt(np.mean(np.square(values))))
hierarchical_anomaly_score = hierarchical_distances / (hier_cluster_rms + 1e-12)

anomalies_hier = pd.DataFrame({
    "cluster": hierarchical_labels,
    "anomaly_score": hierarchical_anomaly_score
})

hier_threshold = np.quantile(hierarchical_anomaly_score, 0.99)
hier_is_outlier = hierarchical_anomaly_score >= hier_threshold

print("\nHierarchical anomaly score summary:")
print(pd.Series(hierarchical_anomaly_score).describe())
print(f"Hierarchical outlier threshold (99th percentile): {hier_threshold:.4f}")
print("Number of flagged outliers:", hier_is_outlier.sum())
print(f"\nTop {top_n} Hierarchical outliers:")
top_hier_outliers = anomaly_df.copy()
top_hier_outliers["hier_anomaly_score"] = hierarchical_anomaly_score
print(top_hier_outliers.nlargest(top_n, "hier_anomaly_score")[["cluster", "hier_anomaly_score", "stroke"]])

# DBSCAN Parammeter Search
eps_values = [0.5, 1, 1.5, 2, 2.5, 3]
min_samples_values = (5, 10, 15)

best_dbscan_score = -1
best_parameters = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # skip if only one cluster
        if len(set(labels)) <= 2:
            continue
        
        score = silhouette_score(X_scaled, labels)
        print(f"DBSCAN eps={eps}, min_samples={min_samples} silhouette={score}")
        
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_parameters = (eps, min_samples)

print("Best DBSCAN parameters:", best_parameters)

#final DBSCAN with best parameters
dbscan = DBSCAN(eps=best_parameters[0], min_samples=best_parameters[1])
dbscan_labels = dbscan.fit_predict(X_scaled)

# DBSCAN anomaly detection 
# # for core points compute distance to cluster center 
dbscan_centers = np.vstack([X_scaled[dbscan_labels==c].mean(axis=0) 
for c in sorted(np.unique(dbscan_labels)) if c != -1])
dbscan_anomaly_score = np.zeros(len(X_scaled))

for c in sorted(np.unique(dbscan_labels)):
    if c == -1:
        dbscan_anomaly_score[dbscan_labels == -1] = np.inf  # Noise points are strongest anomalies
    else:
        mask = dbscan_labels == c
        dists = np.linalg.norm(X_scaled[mask] - dbscan_centers[c], axis=1)
        rms_dist = np.sqrt(np.mean(np.square(dists)))
        dbscan_anomaly_score[mask] = dists / (rms_dist + 1e-12)

#flagging noise points as outliers  
dbscan_is_noise = dbscan_labels == -1
if dbscan_is_noise.any():
    dbscan_is_outlier = dbscan_is_noise
else:
    dbscan_is_outlier = dbscan_anomaly_score >= np.quantile(dbscan_anomaly_score, 0.99)

print("\nDBSCAN anomaly detection:")
print(f"Number of noise points: {dbscan_is_noise.sum()}")
if not np.isinf(dbscan_anomaly_score).all():
    print(pd.Series(dbscan_anomaly_score[~np.isinf(dbscan_anomaly_score)]).describe())
print("Number of flagged outliers:", dbscan_is_outlier.sum())
print(f"\nTop {top_n} DBSCAN outliers (noise points first):")
top_dbscan = anomaly_df.copy()
top_dbscan["dbscan_anomaly_score"] = dbscan_anomaly_score
top_dbscan["is_noise"] = dbscan_is_noise
print(top_dbscan[dbscan_is_outlier].head(top_n)[["cluster", "dbscan_anomaly_score", "is_noise", "stroke"]])

# cross method comparisonof outliers
print("\n" + "="*60)
print("CROSS-METHOD OUTLIER COMPARISON")
print("="*60)

kmeans_outliers = anomaly_df["is_outlier"].values
hierarchical_outliers = hier_is_outlier
dbscan_outliers = dbscan_is_outlier

overlap_all_three = (kmeans_outliers & hierarchical_outliers & dbscan_outliers).sum()
overlap_k_h = (kmeans_outliers & hierarchical_outliers).sum()
overlap_k_d = (kmeans_outliers & dbscan_outliers).sum()
overlap_h_d = (hierarchical_outliers & dbscan_outliers).sum()

print(f"Outliers flagged by K-means only: {(kmeans_outliers & ~hierarchical_outliers & ~dbscan_outliers).sum()}")
print(f"Outliers flagged by Hierarchical only: {(~kmeans_outliers & hierarchical_outliers & ~dbscan_outliers).sum()}")
print(f"Outliers flagged by DBSCAN only: {(~kmeans_outliers & ~hierarchical_outliers & dbscan_outliers).sum()}")
print(f"Outliers flagged by K-means AND Hierarchical: {overlap_k_h}")
print(f"Outliers flagged by K-means AND DBSCAN: {overlap_k_d}")
print(f"Outliers flagged by Hierarchical AND DBSCAN: {overlap_h_d}")
print(f"Outliers flagged by ALL THREE methods: {overlap_all_three}")

if overlap_all_three > 0:
    print(f"\nOutliers identified by all three methods:")
    consensus_outliers = kmeans_outliers & hierarchical_outliers & dbscan_outliers
    comparison_df = pd.DataFrame({
        "kmeans_score": anomaly_df["Anomaly_score"],
        "hier_score": pd.Series(hierarchical_anomaly_score),
        "dbscan_score": pd.Series(dbscan_anomaly_score),
        "stroke": anomaly_df["stroke"]})
    print(comparison_df[consensus_outliers].head(10))

# Clustering Evaluation

# Visualization

#make data 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering (PCA)")
plt.show()

# Inspection

# Puts the data into a DataFrame
evaluation_df = pd.DataFrame(X)
# cluster labels
evaluation_df["cluster"] = kmeans_labels
# stroke label
evaluation_df["stroke"] = y

# prints the average values in each cluster
print(evaluation_df.groupby("cluster").mean())
# prints the stroke distribution in each cluster
print(pd.crosstab(evaluation_df["cluster"], evaluation_df["stroke"]))

# Internal Indices

# Sum of Squared Errors (SSE)
print("K-Means SSE:", kmeans.inertia_)

# Heatmap of the correlation between the distance matrix and the incidence matrix

# Distance matrix
distance_matrix = pairwise_distances(X_scaled)
# Incidence matrix (1 if in the same cluster, 0 otherwise)
incidence_matrix = np.equal.outer(kmeans_labels, kmeans_labels).astype(int)

# Distance heatmap
plt.figure(figsize=(6,5))
sns.heatmap(distance_matrix, cmap="viridis")
plt.title("Distance Matrix")
plt.show()

# Incidence heatmap
plt.figure(figsize=(6,5))
sns.heatmap(incidence_matrix, cmap="coolwarm")
plt.title("Incidence Matrix (Same Cluster = 1)")
plt.show()

# Silhouette Scores
print("K-Means Silhouette:", silhouette_score(X_scaled, kmeans_labels))
print("Hierarchical Silhouette:", silhouette_score(X_scaled, hierarchical_labels))

if len(set(dbscan_labels)) > 2:
    print("DBSCAN Silhouette:", silhouette_score(X_scaled, dbscan_labels))
else:
    print("DBSCAN Silhouette: not valid")

# Relative Indices
    
# K-Means compared to Hierarchical
print("K-Means vs Hierarchical:")
print("ARI:", adjusted_rand_score(kmeans_labels, hierarchical_labels))
print("NMI:", normalized_mutual_info_score(kmeans_labels, hierarchical_labels))
print("AMI:", adjusted_mutual_info_score(kmeans_labels, hierarchical_labels))

# K-Means compared to DBSCAN
print("\nK-Means vs DBSCAN:")
print("ARI:", adjusted_rand_score(kmeans_labels, dbscan_labels))
print("NMI:", normalized_mutual_info_score(kmeans_labels, dbscan_labels))
print("AMI:", adjusted_mutual_info_score(kmeans_labels, dbscan_labels))

# External Indices

# K-Means
print("\nK-Means compared to Stroke")
print("Homogeneity:", homogeneity_score(y, kmeans_labels))
print("Completeness:", completeness_score(y, kmeans_labels))
print("V-measure:", v_measure_score(y, kmeans_labels))
print("Contingency Matrix:")
print(contingency_matrix(y, kmeans_labels))

# Hierarchical
print("\nHierarchical compared to Stroke")
print("Homogeneity:", homogeneity_score(y, hierarchical_labels))
print("Completeness:", completeness_score(y, hierarchical_labels))
print("V-measure:", v_measure_score(y, hierarchical_labels))
print("Contingency Matrix:")
print(contingency_matrix(y, hierarchical_labels))

# DBSCAN
print("\nDBSCAN compared to Stroke")
print("Homogeneity:", homogeneity_score(y, dbscan_labels))
print("Completeness:", completeness_score(y, dbscan_labels))
print("V-measure:", v_measure_score(y, dbscan_labels))
print("Contingency Matrix:")
print(contingency_matrix(y, dbscan_labels))