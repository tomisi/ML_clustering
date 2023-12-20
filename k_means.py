dtype=float
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Standardize the data
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris_data)

# Streamlit UI
st.title("K-Means Clustering with Streamlit")

# Sidebar
st.sidebar.header("Settings")
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
init_method = st.sidebar.selectbox("Initialization Method", ["random", "k-means++"])

# Function to apply k-means clustering
def k_means_clustering(data, num_clusters, init_method):
    kmeans = KMeans(n_clusters=num_clusters, init=init_method, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

# Apply k-means clustering
cluster_labels = k_means_clustering(iris_scaled, num_clusters, init_method)

# Visualize the results (2D plot, considering only the first two features)
plt.scatter(iris_scaled[:, 0], iris_scaled[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k')
plt.title(f'K-Means Clustering of Iris Dataset (Clusters: {num_clusters}, Initialization: {init_method})')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
st.pyplot(plt)

# Display cluster details
st.header("Cluster Details")
cluster_details = pd.DataFrame({
    "Sepal Length": iris_data["sepal length (cm)"],
    "Sepal Width": iris_data["sepal width (cm)"],
    "Petal Length": iris_data["petal length (cm)"],
    "Petal Width": iris_data["petal width (cm)"],
    "Cluster Label": cluster_labels
})
st.write(cluster_details)

# Display cluster statistics
st.header("Cluster Statistics")
cluster_stats = cluster_details.groupby("Cluster Label").mean()
st.write(cluster_stats)
