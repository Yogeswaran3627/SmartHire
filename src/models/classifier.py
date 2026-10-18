"""
Supervised Resume Category Classifier for SmartHire.
Implements TF-IDF vectorization and multiclass classification (Logistic Regression / LinearSVC / Random Forest).
Saves models to models/classifier.pkl and models/tfidf_vectorizer.pkl.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from src.config import (
    MODEL_DIR,
    CLASSIFIER_PATH,
    TFIDF_PATH,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES
)

def train_classifier(
    df: pd.DataFrame,
    save: bool = True,
    model_type: str = "logistic_regression"
) -> tuple[Pipeline, dict]:
    """
    Trains a supervised multiclass text classification pipeline on labeled resumes.
    df must contain 'resume_text' and 'category'.
    """
    counts = df["category"].value_counts()
    valid = df[df["category"].isin(counts[counts >= 2].index)].copy()
    X = valid["resume_text"].astype(str)
    y = valid["category"].astype(str)

    if len(valid) < 4 or y.nunique() < 2:
        raise ValueError("Need at least two categories with sufficient samples to train.")

    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)

    model = Pipeline([
        ("tfidf", tfidf),
        ("classifier", clf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    classes = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "num_classes": int(y.nunique()),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # Save primary classifier
        joblib.dump(model, CLASSIFIER_PATH)
        # Also maintain compatibility with classifier.joblib
        joblib.dump(model, MODEL_DIR / "classifier.joblib")
        # Save standalone fitted TF-IDF vectorizer
        joblib.dump(model.named_steps["tfidf"], TFIDF_PATH)
        print(f"Saved classifier to {CLASSIFIER_PATH} and vectorizer to {TFIDF_PATH}")

    return model, metrics

def load_classifier() -> Pipeline | None:
    """Loads saved classifier from classifier.pkl (or classifier.joblib as fallback)."""
    if CLASSIFIER_PATH.exists():
        try:
            return joblib.load(CLASSIFIER_PATH)
        except Exception:
            pass
    joblib_path = MODEL_DIR / "classifier.joblib"
    if joblib_path.exists():
        try:
            return joblib.load(joblib_path)
        except Exception:
            pass
    return None

def predict_category(
    model: Pipeline,
    resume_text: str,
    top_k: int = 5
) -> dict:
    """
    Predicts the target job category with probability distribution for top-K classes.
    """
    predicted_cat = str(model.predict([str(resume_text)])[0])
    top_predictions = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([str(resume_text)])[0]
        classes = model.named_steps["classifier"].classes_
        sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {"category": str(cat), "probability": round(float(prob), 4), "percentage": round(float(prob) * 100, 1)}
            for cat, prob in sorted_pairs[:top_k]
        ]
    else:
        top_predictions = [{"category": predicted_cat, "probability": 1.0, "percentage": 100.0}]

    confidence = top_predictions[0]["probability"] if top_predictions else 1.0

    return {
        "predicted_category": predicted_cat,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "top_predictions": top_predictions
    }
