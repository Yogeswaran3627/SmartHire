"""
Supervised Shortlisting / Fit Predictor Model for SmartHire.
Predicts candidate-job fit probability based on engineered features:
skill overlap, text similarity, experience match, and education match.
Compares Logistic Regression vs XGBoost / Random Forest and saves fit_predictor.pkl.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import FIT_PREDICTOR_PATH, MODEL_DIR, RANDOM_STATE
from src.features.match_features import compute_match_features, FEATURE_COLUMNS

def generate_synthetic_training_pairs(
    resumes_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    n_pairs: int = 1500,
    vectorizer=None
) -> pd.DataFrame:
    """
    Constructs labeled (resume, job) feature dataset for supervised fit prediction:
    - Label 1 (Fit / Shortlisted): Resume matched to aligned job domain with skill overlap.
    - Label 0 (Not Fit): Resume cross-paired with orthogonal domain job.
    """
    rows = []
    rng = np.random.RandomState(RANDOM_STATE)

    sample_resumes = resumes_df.sample(min(len(resumes_df), 250), random_state=RANDOM_STATE)
    sample_jobs = jobs_df.sample(min(len(jobs_df), 500), random_state=RANDOM_STATE)

    resume_records = sample_resumes.to_dict("records")
    job_records = sample_jobs.to_dict("records")

    pairs_collected = 0
    while pairs_collected < n_pairs:
        r = resume_records[rng.randint(0, len(resume_records))]
        j = job_records[rng.randint(0, len(job_records))]

        r_text = str(r.get("resume_text", ""))
        j_text = str(j.get("job_text", ""))
        r_cat = str(r.get("category", "")).lower().replace("-", " ")
        j_title = str(j.get("title", "")).lower()

        features = compute_match_features(r_text, j_text, vectorizer=vectorizer)

        # Domain alignment heuristic
        cat_match = any(word in j_title for word in r_cat.split() if len(word) > 3)
        high_sim = (features["skill_coverage"] >= 0.35) or (features["text_similarity"] >= 0.22)

        if cat_match or high_sim:
            label = 1
        elif (features["skill_coverage"] <= 0.15) and (features["text_similarity"] <= 0.12):
            label = 0
        else:
            # Ambiguous sample, assign probabilistic label
            label = 1 if (features["skill_overlap"] * 0.5 + features["text_similarity"] * 0.5) > 0.15 else 0

        features["label"] = label
        rows.append(features)
        pairs_collected += 1

    return pd.DataFrame(rows)

def train_fit_predictor(
    train_df: pd.DataFrame | None = None,
    resumes_df: pd.DataFrame | None = None,
    jobs_df: pd.DataFrame | None = None,
    vectorizer=None,
    save: bool = True
) -> tuple[object, dict]:
    """
    Trains and compares Logistic Regression vs XGBoost / Random Forest on match features.
    Saves the best performing model to models/fit_predictor.pkl.
    """
    if train_df is None:
        if resumes_df is None or jobs_df is None:
            raise ValueError("Provide train_df or both resumes_df and jobs_df.")
        train_df = generate_synthetic_training_pairs(resumes_df, jobs_df, n_pairs=1600, vectorizer=vectorizer)

    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models_to_test = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE))
        ])
    }

    if HAS_XGBOOST:
        models_to_test["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss"))
        ])

    comparison = {}
    best_name = None
    best_auc = -1.0
    best_pipeline = None

    for name, pipe in models_to_test.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else preds

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)

        comparison[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Confusion Matrix": confusion_matrix(y_test, preds).tolist()
        }

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipe

    # Feature Importances if tree-based model
    feature_importances = {}
    clf_step = best_pipeline.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        for col, imp in zip(FEATURE_COLUMNS, clf_step.feature_importances_):
            feature_importances[col] = round(float(imp), 4)
    elif hasattr(clf_step, "coef_"):
        for col, coef in zip(FEATURE_COLUMNS, clf_step.coef_[0]):
            feature_importances[col] = round(float(abs(coef)), 4)

    results = {
        "best_model_name": best_name,
        "best_roc_auc": float(best_auc),
        "comparison_table": comparison,
        "feature_importances": feature_importances,
        "feature_columns": FEATURE_COLUMNS
    }

    if save:
        FIT_PREDICTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": best_pipeline, "metadata": results}, FIT_PREDICTOR_PATH)
        print(f"Saved winning fit predictor ({best_name}) to {FIT_PREDICTOR_PATH}")

    return best_pipeline, results

def load_fit_predictor():
    """Loads saved fit predictor model from models/fit_predictor.pkl."""
    if FIT_PREDICTOR_PATH.exists():
        try:
            bundle = joblib.load(FIT_PREDICTOR_PATH)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle["pipeline"]
            return bundle
        except Exception:
            return None
    return None

def predict_fit_score(
    resume_text: str,
    job_text: str,
    model=None,
    vectorizer=None
) -> dict:
    """
    Computes fit / shortlisting probability for a candidate resume against a job.
    Returns probability (0.0 to 1.0), shortlisting status, and feature explanations.
    """
    features = compute_match_features(resume_text, job_text, vectorizer=vectorizer)
    feature_df = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    if model is None:
        model = load_fit_predictor()

    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(feature_df)[0][1])
            else:
                prob = float(model.predict(feature_df)[0])
        except Exception:
            # Fallback heuristic
            prob = float(features["skill_coverage"] * 0.45 + features["text_similarity"] * 0.35 + features["experience_match"] * 0.20)
    else:
        prob = float(features["skill_coverage"] * 0.45 + features["text_similarity"] * 0.35 + features["experience_match"] * 0.20)

    prob = float(np.clip(prob, 0.05, 0.98))
    shortlisted = prob >= 0.55

    return {
        "fit_probability": round(prob, 4),
        "fit_percent": round(prob * 100, 1),
        "is_shortlisted": shortlisted,
        "status": "Recommended for Shortlisting" if shortlisted else "Requires Skill Enhancement",
        "features": features
    }
