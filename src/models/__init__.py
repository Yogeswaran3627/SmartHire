"""Machine learning models package for SmartHire."""
from src.models.classifier import train_classifier, load_classifier, predict_category
from src.models.recommender import recommend_jobs, evaluate_precision_at_k
from src.models.clustering import cluster_jobs, find_optimal_k, compute_pca_projection, extract_job_topics
from src.models.fit_predictor import train_fit_predictor, load_fit_predictor, predict_fit_score

__all__ = [
    "train_classifier",
    "load_classifier",
    "predict_category",
    "recommend_jobs",
    "evaluate_precision_at_k",
    "cluster_jobs",
    "find_optimal_k",
    "compute_pca_projection",
    "extract_job_topics",
    "train_fit_predictor",
    "load_fit_predictor",
    "predict_fit_score",
]
