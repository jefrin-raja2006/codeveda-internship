# Task 4: K-Means Clustering

## 📌 Overview

This project implements K-Means Clustering, an unsupervised machine learning algorithm used to group similar data points into clusters.
The goal is to identify patterns in unlabeled data without predefined categories.

---

## 📊 Dataset

* Type: Numeric dataset (House/Boston-style dataset)
* Format: Whitespace-separated values
* Contains multiple numerical features representing real-world attributes

---

## ⚙️ Workflow

### 1. Data Loading

* Dataset loaded using pandas
* Handled whitespace-separated format

### 2. Data Preprocessing

* Removed missing values
* Ensured all features are numeric

### 3. Data Scaling

* Applied StandardScaler to normalize data
* Important for K-Means to treat all features equally

### 4. Elbow Method

* Used to determine optimal number of clusters (K)
* Plotted WCSS vs number of clusters

### 5. Model Building

* Applied K-Means clustering using scikit-learn
* Assigned cluster labels to each data point

### 6. Visualization

* Visualized clusters using 2D scatter plot
* Used first two features for plotting

---

## 📈 Output

Generated outputs:

* Elbow graph:

```text id="8y6k0t"
outputs/elbow_plot.png
```

* Cluster visualization:

```text id="f5s71h"
outputs/clusters.png
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* Matplotlib

---

## ▶️ How to Run

```bash id="x9n8n6"
cd task4-kmeans-clustering
python src/kmeans.py
```

---

## 🧠 Key Learnings

* Understanding unsupervised learning
* Working with real-world datasets without labels
* Importance of scaling data
* Choosing optimal clusters using elbow method
* Visualizing clustering results

---

## 🚀 Future Improvements

* Use PCA for better visualization
* Try different clustering algorithms (DBSCAN, Hierarchical)
* Experiment with different numbers of clusters
* Add 3D visualization
