import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Customer Segmentation using K-Means")

data = pd.read_csv("Mall_Customers.csv")

st.write("Dataset Preview")
st.write(data.head())

X = data[['Annual_Income','Spending_Score']]

k = st.slider("Select number of clusters",2,10,5)

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

clusters = kmeans.labels_

data['Cluster'] = clusters

st.write("Clustered Data")
st.write(data.head())

plt.scatter(X['Annual_Income'],X['Spending_Score'],c=clusters)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

st.pyplot(plt)
