"""
End-to-end model training, data preprocessing, and report figure generation script for SmartHire.
"""
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.config import (
    CLEAN_RESUMES_PATH,
    CLEAN_JOBS_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    MODEL_DIR,
    CLASSIFIER_PATH,
    TFIDF_PATH,
    FIT_PREDICTOR_PATH,
    CLUSTERING_PATH
)
from src.data.load_data import load_raw_resumes, load_processed_data
from src.data.preprocess import preprocess_resumes, merge_job_corpus
from src.features.text_features import fit_and_save_vectorizer
from src.models.classifier import train_classifier
from src.models.clustering import cluster_jobs, find_optimal_k, compute_pca_projection
from src.models.fit_predictor import train_fit_predictor, generate_synthetic_training_pairs
from src.models.recommender import recommend_jobs, evaluate_precision_at_k

def run_pipeline():
    print("==================================================")
    print("[*] STARTING SMARTHIRE ML PIPELINE & TRAINING")
    print("==================================================")


    # 1. PREPROCESSING
    print("\n--- 1. Preprocessing Resumes and Jobs Corpus ---")
    resumes_df = preprocess_resumes(save=True)
    print(f"Resumes: {len(resumes_df)} rows across {resumes_df['category'].nunique()} categories")

    jobs_df = merge_job_corpus(save=True, max_rows=25000)
    print(f"Jobs Corpus: {len(jobs_df)} rows across {jobs_df['source'].nunique()} sources")

    # 2. FIT & SAVE TFIDF VECTORIZER
    print("\n--- 2. Fitting Global TF-IDF Vectorizer ---")
    all_texts = list(resumes_df["resume_text"]) + list(jobs_df["job_text"].sample(min(len(jobs_df), 10000), random_state=42))
    global_vectorizer, _ = fit_and_save_vectorizer(all_texts, save_path=TFIDF_PATH)
    print(f"TF-IDF Vectorizer saved to {TFIDF_PATH}")

    # 3. TRAIN RESUME CATEGORY CLASSIFIER (SUPERVISED)
    print("\n--- 3. Training Resume Category Classifier (Model A) ---")
    clf_model, clf_metrics = train_classifier(resumes_df, save=True)
    print(f"Classifier Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"Classifier Weighted F1: {clf_metrics['f1_weighted']:.4f}")

    # 4. TRAIN K-MEANS CLUSTERING & ELBOW / SILHOUETTE (UNSUPERVISED)
    print("\n--- 4. Unsupervised Job Clustering & Topic Discovery ---")
    clustered_jobs, kmeans_model, cluster_vec, cluster_info = cluster_jobs(jobs_df, max_rows=4000, save=True)
    print(f"Clustering Complete. Clusters: {cluster_info['n_clusters']}, Silhouette: {cluster_info['silhouette_score']:.4f}")
    
    elbow_data = find_optimal_k(jobs_df.head(2000), k_range=range(2, 8))

    # 5. TRAIN SHORTLISTING / FIT PREDICTOR (SUPERVISED)
    print("\n--- 5. Training Candidate Shortlisting / Fit Predictor (Model B) ---")
    fit_pipe, fit_results = train_fit_predictor(
        resumes_df=resumes_df,
        jobs_df=jobs_df,
        vectorizer=global_vectorizer,
        save=True
    )
    print(f"Fit Predictor Trained. Winning Model: {fit_results['best_model_name']}")
    print(f"Winning ROC-AUC: {fit_results['best_roc_auc']:.4f}")
    for name, m in fit_results['comparison_table'].items():
        print(f"   - {name}: Accuracy={m['Accuracy']}, F1={m['F1-Score']}, ROC-AUC={m['ROC-AUC']}")

    # 6. EVALUATE RECOMMENDER PRECISION@K
    print("\n--- 6. Recommender Precision@K Evaluation ---")
    sample_resume = resumes_df[resumes_df["category"] == "INFORMATION-TECHNOLOGY"].iloc[0]["resume_text"]
    recs = recommend_jobs(sample_resume, jobs_df, top_n=10, vectorizer=global_vectorizer)
    p_at_5 = evaluate_precision_at_k(recs, "information technology", k=5)
    print(f"Recommender Precision@5 for IT resume sample: {p_at_5 * 100:.1f}%")

    # 7. GENERATE PUBLICATION FIGURES
    print("\n--- 7. Generating Figures for Reports ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Resume Category Distribution
    plt.figure(figsize=(10, 6))
    cat_counts = resumes_df["category"].value_counts().head(15)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, palette="viridis")
    plt.title("Top Resume Categories in SmartHire Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Resumes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "classifier_class_distribution.png", dpi=200)
    plt.close()

    # Figure 2: Confusion Matrix (Top 10 Classes)
    plt.figure(figsize=(10, 8))
    top_10_classes = resumes_df["category"].value_counts().head(10).index.tolist()
    # Submatrix
    cm = np.array(clf_metrics["confusion_matrix"])
    all_classes = clf_metrics["classes"]
    indices = [all_classes.index(c) for c in top_10_classes if c in all_classes]
    sub_cm = cm[np.ix_(indices, indices)]
    sns.heatmap(sub_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c[:12] for c in top_10_classes],
                yticklabels=[c[:12] for c in top_10_classes])
    plt.title("Resume Classifier Confusion Matrix (Top Categories)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Figure 3: KMeans Elbow and Silhouette Curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(elbow_data["k_values"], elbow_data["inertias"], 'b-o', label="Inertia (Elbow)")
    ax2.plot(elbow_data["k_values"], elbow_data["silhouette_scores"], 'g-s', label="Silhouette Score")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color='b')
    ax2.set_ylabel("Silhouette Score", color='g')
    plt.title("Job Clustering: Elbow Method & Silhouette Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow_silhouette.png", dpi=200)
    plt.close()

    # Figure 4: PCA Cluster 2D Scatter Plot
    plt.figure(figsize=(9, 6))
    texts = clustered_jobs["job_text"].fillna("").astype(str)
    sample_mat = cluster_vec.transform(texts)
    coords = compute_pca_projection(sample_mat, n_components=2)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=clustered_jobs["cluster"], cmap="tab10", alpha=0.6, s=15)
    plt.title("Job Postings 2D PCA Cluster Projection", fontsize=14, fontweight="bold")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "job_clusters_pca.png", dpi=200)
    plt.close()

    # Figure 5: Fit Predictor Feature Importances
    plt.figure(figsize=(9, 5))
    feat_imp = fit_results["feature_importances"]
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    sns.barplot(x=[v for _, v in sorted_feats], y=[k.replace("_", " ").title() for k, _ in sorted_feats], palette="mako")
    plt.title("Candidate Shortlisting Model: Feature Importance", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance / Weight")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_predictor_feature_importance.png", dpi=200)
    plt.close()

    # Figure 6: Model Comparison Table Bar Chart
    plt.figure(figsize=(8, 4.5))
    comp_df = pd.DataFrame(fit_results["comparison_table"]).T[["Accuracy", "F1-Score", "ROC-AUC"]]
    comp_df.plot(kind="bar", figsize=(8, 4.5), colormap="crest")
    plt.title("Shortlisting Predictor: Model Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0.7, 1.02)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_predictor_roc_curve.png", dpi=200)
    plt.close()

    print(f"All 6 figures successfully saved in {FIGURES_DIR}")
    print("\n==================================================")
    print("[OK] ML PIPELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_pipeline()
