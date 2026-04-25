import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score

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

# Temporary labels for Clustering (waiting for actualy labels from clustering techniques):

# K-Means labels
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical labels
hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# DBSCAN labels
dbscan = DBSCAN(eps=2, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

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
