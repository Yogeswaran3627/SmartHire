"""
Model Evaluation and Metrics Module for SmartHire.
Provides functions for computing classification, clustering, recommendation, and regression metrics.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_classifier_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Computes comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                # Binary ROC-AUC
                pos_probs = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_probs))
            else:
                # Multiclass ROC-AUC
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            metrics["roc_auc"] = None

    return metrics

def evaluate_clustering_metrics(matrix, cluster_labels) -> dict:
    """Computes silhouette score and basic cluster diagnostics."""
    unique_clusters = len(set(cluster_labels))
    if 1 < unique_clusters < matrix.shape[0]:
        sil = float(silhouette_score(matrix, cluster_labels, sample_size=min(1000, matrix.shape[0]), random_state=42))
    else:
        sil = 0.0
    return {
        "n_clusters": unique_clusters,
        "silhouette_score": round(sil, 4)
    }

def evaluate_recommender_precision_at_k(recommended_df: pd.DataFrame, expected_domain: str, k: int = 5) -> float:
    """Precision@K: fraction of top-K recommended jobs that match expected candidate domain."""
    if recommended_df.empty or k <= 0:
        return 0.0
    top_k = recommended_df.head(k)
    expected_words = [w.lower() for w in expected_domain.replace("-", " ").split() if len(w) > 2]
    
    hits = 0
    for _, row in top_k.iterrows():
        text = f"{row.get('title', '')} {row.get('skills', '')} {row.get('job_text', '')}".lower()
        if any(w in text for w in expected_words):
            hits += 1
    return round(hits / len(top_k), 3)

def evaluate_regression_metrics(y_true, y_pred) -> dict:
    """Computes MAE, RMSE, and R2 metrics for salary / score regressions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }
