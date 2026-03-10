import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.title("Customer Segmentation using K-Means Clustering")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

st.subheader("Original Dataset")
st.dataframe(df)

# Data Preprocessing
df = df.drop("Gender", axis=1)

df.rename(columns={
    "Annual Income (k$)": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score"
}, inplace=True)

st.subheader("Processed Dataset")
st.dataframe(df)

# Feature selection
X = df[["Annual_Income", "Spending_Score"]]

# User select K value
k = st.slider("Select number of clusters (K)", 2, 10, 5)

# Train model
model = KMeans(n_clusters=k, random_state=42)
model.fit(X)

clusters = model.labels_

df["Cluster"] = clusters

st.subheader("Clustered Dataset")
st.dataframe(df)

# Cluster Centers
st.subheader("Cluster Centers")

centers = model.cluster_centers_
centers_df = pd.DataFrame(centers, columns=["Income Center","Spending Center"])

st.write(centers_df)

# Silhouette Score
score = silhouette_score(X, clusters)

st.subheader("Model Evaluation")

st.write("Silhouette Score:", round(score,3))

# Scatter plot
st.subheader("Customer Segmentation Graph")

plt.figure()

plt.scatter(X["Annual_Income"], X["Spending_Score"], c=clusters)

plt.scatter(
    centers[:,0],
    centers[:,1],
    s=300,
    c='red',
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

st.pyplot(plt)

from sklearn.metrics import silhouette_score

score = silhouette_score(X, clusters)

st.write("Silhouette Score:", score)

st.write("Inertia:", model.inertia_)