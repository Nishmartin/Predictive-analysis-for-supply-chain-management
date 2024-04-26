import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Merline\Desktop\FYP\FinalYearProject2\cleaned_dataset_updated.csv"
df = pd.read_csv(file_path)

# Select features for clustering
features = ['Sales', 'Order Item Quantity', 'Benefit per order', 'Order Profit Per Order']  # Updated features
X = df[features]

# Standardize the features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters and fit the K-Means model
optimal_n_clusters = 3  # Adjust based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Optionally, you can analyze the clusters by examining the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=X.columns)
print("Cluster Centers:")
print(cluster_centers_df)

# Visualize the clusters (for 2D or 3D data)
# For 2D data (e.g., feature1 vs. feature2), create a scatter plot and color points by cluster
plt.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('K-Means Clustering')
plt.show()
