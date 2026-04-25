import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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