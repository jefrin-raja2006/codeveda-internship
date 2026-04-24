# Task 4: K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Load Dataset (FINAL FIX)
# -----------------------------
# Read whitespace-separated dataset
df = pd.read_csv("data/dataset.csv", header=None, delim_whitespace=True)

print("\nDataset Loaded!")
print(df.head())


# -----------------------------
# 2. Preprocessing
# -----------------------------
df = df.dropna()

print("\nShape:", df.shape)


# -----------------------------
# 3. Scaling
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# -----------------------------
# 4. Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.savefig("outputs/elbow_plot.png")
plt.show()


# -----------------------------
# 5. Apply KMeans
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

df['Cluster'] = clusters


# -----------------------------
# 6. Visualization
# -----------------------------
plt.figure()

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'])

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clusters")

plt.savefig("outputs/clusters.png")
plt.show()