# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:58:35 2025

@author: Ajay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data: Purchase history (you can replace this with a real dataset)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualSpending': [15000, 12000, 30000, 40000, 10000, 8000, 20000, 25000, 5000, 45000],
    'NumTransactions': [50, 30, 100, 120, 25, 20, 80, 90, 10, 130]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Extract features for clustering
X = df[['AnnualSpending', 'NumTransactions']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Based on the elbow method, choose the number of clusters (e.g., k=3)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(8, 5))
for cluster in range(k_optimal):
    cluster_points = df[df['Cluster'] == cluster]
    plt.scatter(cluster_points['AnnualSpending'], cluster_points['NumTransactions'], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            s=200, c='red', label='Centroids', marker='X')

plt.xlabel('Annual Spending')
plt.ylabel('Number of Transactions')
plt.title('Customer Clusters')
plt.legend()
plt.show()

# Display the resulting clusters
print("Customer Clusters:")
print(df)
