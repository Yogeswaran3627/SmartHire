"""
Unsupervised Job Clustering & Topic Modeling for SmartHire.
Implements K-Means clustering, Elbow method, Silhouette analysis,
PCA 2D projection, and LDA topic modeling on job descriptions.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from src.config import CLUSTERING_PATH, RANDOM_STATE

def find_optimal_k(
    jobs_sample: pd.DataFrame,
    k_range: range = range(2, 9),
    max_features: int = 5000
) -> dict:
    """
    Evaluates KMeans over k_range to plot Elbow Curve (Inertia) and Silhouette scores.
    """
    texts = jobs_sample["job_text"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    matrix = vectorizer.fit_transform(texts)

    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        labels = km.fit_predict(matrix)
        inertias.append(float(km.inertia_))
        if 1 < len(set(labels)) < len(texts):
            sil = float(silhouette_score(matrix, labels, sample_size=min(800, len(texts)), random_state=RANDOM_STATE))
            silhouette_scores.append(round(sil, 4))
        else:
            silhouette_scores.append(0.0)

    # Pick optimal k based on highest silhouette score
    best_idx = int(np.argmax(silhouette_scores)) if silhouette_scores else 0
    best_k = k_values[best_idx] if k_values else 5

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "recommended_k": best_k
    }

def cluster_jobs(
    jobs_df: pd.DataFrame,
    n_clusters: int | None = None,
    max_rows: int = 5000,
    save: bool = True
) -> tuple[pd.DataFrame, KMeans, TfidfVectorizer, dict]:
    """
    Clusters job corpus into natural job families using KMeans.
    Extracts top keywords per cluster.
    """
    if len(jobs_df) < 3:
        raise ValueError("At least 3 job postings required for clustering.")

    sample = jobs_df.head(max_rows).copy()
    texts = sample["job_text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts)

    if n_clusters is None:
        # Determine k dynamically between 3 and 7
        n_clusters = min(6, max(3, int(len(sample) ** 0.35)))

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    sample["cluster"] = kmeans.fit_predict(matrix)

    # Compute silhouette score
    sil_score = None
    if 1 < sample["cluster"].nunique() < len(sample):
        sil_score = float(silhouette_score(matrix, sample["cluster"], sample_size=min(1000, len(sample)), random_state=RANDOM_STATE))

    # Extract top keywords per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    cluster_keywords = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        cluster_keywords[i] = top_terms

    # Assign readable cluster theme
    sample["cluster_theme"] = sample["cluster"].apply(
        lambda c: " / ".join(cluster_keywords[c][:3]).title()
    )

    cluster_info = {
        "n_clusters": int(n_clusters),
        "silhouette_score": sil_score,
        "inertia": float(kmeans.inertia_),
        "keywords": cluster_keywords
    }

    if save:
        CLUSTERING_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kmeans": kmeans, "vectorizer": vectorizer, "keywords": cluster_keywords}, CLUSTERING_PATH)

    return sample, kmeans, vectorizer, cluster_info

def compute_pca_projection(matrix, n_components: int = 2) -> np.ndarray:
    """Projects high-dimensional sparse TF-IDF matrix into 2D coordinates using PCA."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    coords = pca.fit_transform(dense)
    return coords

def extract_job_topics(
    jobs_sample: pd.DataFrame,
    n_topics: int = 5,
    top_words: int = 8
) -> list[dict]:
    """
    Extracts latent topics and skill themes using Latent Dirichlet Allocation (LDA).
    """
    texts = jobs_sample["job_text"].fillna("").astype(str)
    tf_vectorizer = CountVectorizer(stop_words="english", max_features=4000, max_df=0.90, min_df=2)
    tf = tf_vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        random_state=RANDOM_STATE,
        learning_method="online"
    )
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        words = [tf_feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]]
        topics.append({
            "topic_id": topic_idx + 1,
            "theme": " / ".join(words[:3]).title(),
            "keywords": words
        })

    return topics
