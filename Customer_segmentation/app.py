import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

CLUSTER_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")
st.caption("Unsupervised Learning (K-Means) on the Mall Customers Dataset")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    st.header("Clustering Settings")
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=5)
    show_elbow = st.checkbox("Show Elbow Method Chart", value=True)
    show_3d = st.checkbox("Show 3D Cluster View", value=True)


@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", df.shape[0])
col2.metric("Average Age", round(df["Age"].mean(), 1))
col3.metric("Average Income (k$)", round(df["Annual Income (k$)"].mean(), 1))
col4.metric("Average Spending Score", round(df["Spending Score (1-100)"].mean(), 1))

df_clean = df.drop(columns=["CustomerID"]) if "CustomerID" in df.columns else df.copy()

le = LabelEncoder()
df_clean["Gender"] = le.fit_transform(df_clean["Gender"])

features = df_clean.columns.tolist()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[features])

pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

if show_elbow:
    st.subheader("Elbow Method")
    wcss = []
    k_range = range(1, 11)
    for k_val in k_range:
        km = KMeans(n_clusters=k_val, init="k-means++", random_state=42, n_init=10)
        km.fit(scaled_data)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), wcss, marker="o", color=CLUSTER_PALETTE[0], linewidth=2)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Method for Optimal K")
    st.pyplot(fig)

kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

df_clean["Cluster"] = cluster_labels
pca_df["Cluster"] = cluster_labels
df["Cluster"] = cluster_labels

income_median = df_clean["Annual Income (k$)"].median()
spending_median = df_clean["Spending Score (1-100)"].median()


def assign_persona(row):
    high_income = row["Annual Income (k$)"] >= income_median
    high_spending = row["Spending Score (1-100)"] >= spending_median
    if high_income and high_spending:
        return "Premium Target Customers"
    if high_income and not high_spending:
        return "Cautious High Earners"
    if not high_income and high_spending:
        return "Impulsive Budget Spenders"
    if not high_income and not high_spending:
        return "Budget-Conscious Customers"
    return "Average Customers"


cluster_summary = df_clean.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
cluster_summary["Count"] = df_clean["Cluster"].value_counts().sort_index()
cluster_summary["Persona"] = cluster_summary.apply(assign_persona, axis=1)

color_map = {i: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(k)}

st.subheader("Cluster Visualization")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    fig_2d = px.scatter(
        df_clean, x="Annual Income (k$)", y="Spending Score (1-100)",
        color=df_clean["Cluster"].astype(str),
        color_discrete_map={str(i): color_map[i] for i in range(k)},
        title="Income vs Spending Score",
    )
    st.plotly_chart(fig_2d, use_container_width=True)

with viz_col2:
    fig_pca = px.scatter(
        pca_df, x="PC1", y="PC2",
        color=pca_df["Cluster"].astype(str),
        color_discrete_map={str(i): color_map[i] for i in range(k)},
        title="PCA-Reduced View",
    )
    st.plotly_chart(fig_pca, use_container_width=True)

if show_3d:
    fig_3d = px.scatter_3d(
        df_clean, x="Age", y="Annual Income (k$)", z="Spending Score (1-100)",
        color=df_clean["Cluster"].astype(str),
        color_discrete_map={str(i): color_map[i] for i in range(k)},
        title="3D View of Customer Segments",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

st.subheader("Cluster Profiles & Business Personas")
st.dataframe(cluster_summary, use_container_width=True)

st.subheader("Download Segmented Dataset")
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV with Cluster Labels", data=csv_data, file_name="segmented_customers.csv", mime="text/csv")
