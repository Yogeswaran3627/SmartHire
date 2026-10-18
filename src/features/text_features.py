"""
Text feature extraction module for SmartHire.
Provides TF-IDF vectorization, vocabulary management, and text-similarity calculations.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TFIDF_PATH, TFIDF_MAX_FEATURES

def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """Instantiates a tuned TF-IDF Vectorizer with English stop words and sublinear scaling."""
    return TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b[a-zA-Z0-9_.+#-]+\b"
    )

def fit_and_save_vectorizer(
    texts: list[str],
    save_path: Path | str = TFIDF_PATH,
    max_features: int = TFIDF_MAX_FEATURES
) -> tuple[TfidfVectorizer, object]:
    """Fits TF-IDF vectorizer on a corpus of text and saves to disk as a pickle file."""
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(texts)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, save_path)
    return vectorizer, matrix

def load_tfidf_vectorizer(path: Path | str = TFIDF_PATH) -> TfidfVectorizer | None:
    """Loads saved TF-IDF vectorizer if present."""
    p = Path(path)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception:
            return None
    return None

def compute_text_similarity(
    vectorizer: TfidfVectorizer,
    text1: str,
    text2: str
) -> float:
    """Computes cosine similarity between two text strings."""
    vecs = vectorizer.transform([str(text1), str(text2)])
    sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
    return float(np.clip(sim, 0.0, 1.0))

def extract_top_keywords(
    vectorizer: TfidfVectorizer,
    text: str,
    top_k: int = 10
) -> list[tuple[str, float]]:
    """Extracts the top TF-IDF keywords and their weights for a given document."""
    if not hasattr(vectorizer, "vocabulary_"):
        return []
    vec = vectorizer.transform([str(text)])
    feature_names = vectorizer.get_feature_names_out()
    nonzero_idx = vec.nonzero()[1]
    scores = [(feature_names[i], float(vec[0, i])) for i in nonzero_idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
