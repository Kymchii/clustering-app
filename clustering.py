import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_circles

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap');
        * {
            font-family: "Inter", serif;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load dataset
def load_dataset():
    data = pd.read_csv("dataset/Clustering.csv")
    return data


# Preprocessing dataset
def preprocess_data(data):
    # Hapus duplikasi
    data = data.drop_duplicates()

    # Hapus kolom yang tidak relevan
    non_relevant_columns = [
        "ID",
        "Marital status",
        "Education",
        "Occupation",
        "Settlement size",
        "Sex",
    ]  # Tambahkan nama kolom tidak relevan lainnya jika ada
    data = data.drop(
        columns=[col for col in non_relevant_columns if col in data.columns],
        errors="ignore",
    )

    return data


# Fungsi untuk menampilkan Elbow Method
def plot_elbow_method(data):
    inertia = []
    k_range = range(1, 11)
    for i in k_range:
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    st.pyplot(plt)


def main():
    st.sidebar.markdown(
        """
        <h3 style='font-size: 2.074rem; font-family: "Lora", serif;'>
            Fitur
        </h3>
        """,
        unsafe_allow_html=True
        )
    
    st.markdown(
    """
    <h1 style='font-family: "Lora", serif; font-size: 2.986rem; text-align: center;'>
        Aplikasi Clustering
    </h1>
    """,
    unsafe_allow_html=True
    )
    
    st.markdown(
    """
    <p style='font-weight: 200; text-align: center;'>
        Aplikasi ini mendukung berbagai algoritma clustering seperti K-Means, DBSCAN, dan Hierarchical Clustering.
    </p>
    """,
    unsafe_allow_html=True
    )

    # Load dataset
    data = load_dataset()
    
    st.markdown(
        """
        <h3 style='font-size: 2.074rem; '>
            Dataset Preprocessing
        </h3>
        """,
        unsafe_allow_html=True
        )
    
    st.dataframe(data)
    # Preprocess dataset
    data = preprocess_data(data)
    st.markdown(
        """
        <h3 style='font-size: 2.074rem; '>
            Dataset Afterprocessing
        </h3>
        """,
        unsafe_allow_html=True
        )
    st.dataframe(data)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[["Age", "Income"]])

    # Pilih algoritma clustering
    algorithm = st.sidebar.selectbox(
        "Pilih algoritma clustering:", ["K-Means", "Hierarchical Clustering"]
    )

    if algorithm == "K-Means":
        st.markdown(
        """
        <h3 style='font-size: 2.074rem;'>
            Elbow Method untuk Menentukan Jumlah Cluster
        </h3>
        """,
        unsafe_allow_html=True
        )
        plot_elbow_method(data_scaled)
        n_clusters = st.sidebar.slider(
            "Pilih jumlah cluster:", min_value=2, max_value=10, value=3
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    elif algorithm == "Hierarchical Clustering":
        linkage_method = st.sidebar.selectbox(
            "Pilih metode linkage:", ["single", "complete", "average", "ward"]
        )
        n_clusters = st.sidebar.slider(
            "Pilih jumlah cluster:", min_value=2, max_value=10, value=3
        )
        linkage_matrix = linkage(data, method=linkage_method)
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        centroids_original = None

        # Visualisasi dendrogram
        st.markdown(
        """
        <h3 style='font-size: 2.074rem; '>
            Dendogram
        </h3>
        """,
        unsafe_allow_html=True
        )
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title("Dendrogram (Hierarchical Clustering)")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        st.pyplot(plt)

    # Tambahkan hasil clustering ke data
    data["Cluster"] = labels

    st.markdown(
        """
        <h3 style='font-size: 2.074rem; '>
            Hasil Clustering
        </h3>
        """,
        unsafe_allow_html=True
        )
    st.write(data)

    # Visualisasi
    st.markdown(
        """
        <h3 style='font-size: 2.074rem; '>
            Visualisasi Clustering
        </h3>
        """,
        unsafe_allow_html=True
        )
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data["Age"],
        y=data["Income"],
        hue=data["Cluster"],
        s=100,
        palette="viridis",
        style=data["Cluster"],
    )

    if centroids_original is not None:
        plt.scatter(
            centroids_original[:, data.columns.get_loc("Age")],
            centroids_original[:, data.columns.get_loc("Income")],
            s=100,
            c="red",
            label="Centroid",
        )

    plt.title(f"Visualisasi Clustering ({algorithm})")
    plt.xlabel("Age")
    plt.ylabel("Income")
    plt.legend()
    st.pyplot(plt)


if __name__ == "__main__":
    main()
