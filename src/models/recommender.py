"""
Content-based Job Recommendation Engine for SmartHire.
Matches resumes to job postings using TF-IDF vectorization and Cosine Similarity ranking.
Supports Precision@K evaluation and filtering.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TFIDF_MAX_FEATURES, DEFAULT_TOP_N

def fit_job_vectorizer(job_texts: list[str]) -> tuple[TfidfVectorizer, object]:
    """Fits TF-IDF vectorizer on job descriptions corpus."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(job_texts)
    return vectorizer, matrix

def recommend_jobs(
    resume_text: str,
    jobs_df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
    filter_role: str | None = None,
    filter_location: str | None = None,
    vectorizer: TfidfVectorizer | None = None
) -> pd.DataFrame:
    """
    Ranks job postings by cosine similarity to candidate resume text.
    Returns DataFrame containing top-N matching jobs with similarity metrics.
    """
    if jobs_df.empty:
        return jobs_df.copy()

    df = jobs_df.copy()
    if filter_role and filter_role.strip():
        mask = df["title"].str.contains(filter_role.strip(), case=False, na=False)
        if mask.any():
            df = df[mask].reset_index(drop=True)

    if filter_location and filter_location.strip():
        mask = df["location"].str.contains(filter_location.strip(), case=False, na=False)
        if mask.any():
            df = df[mask].reset_index(drop=True)

    texts = df["job_text"].fillna("").astype(str).tolist()
    if not texts:
        return pd.DataFrame()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        all_texts = texts + [str(resume_text)]
        matrix = vectorizer.fit_transform(all_texts)
        resume_vec = matrix[-1]
        job_matrix = matrix[:-1]
    else:
        job_matrix = vectorizer.transform(texts)
        resume_vec = vectorizer.transform([str(resume_text)])

    scores = cosine_similarity(resume_vec, job_matrix).ravel()

    result = df.copy()
    result["match_score"] = scores.round(4)
    result["match_percent"] = (result["match_score"] * 100).round(1)
    result = result.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result

def evaluate_precision_at_k(
    recommendations_df: pd.DataFrame,
    target_category: str,
    k: int = 5
) -> float:
    """
    Evaluates Precision@K by checking whether top-K jobs match the target role/domain.
    """
    if recommendations_df.empty or k <= 0:
        return 0.0

    top_k = recommendations_df.head(k)
    target_terms = [t.lower() for t in target_category.replace("-", " ").split()]
    
    hits = 0
    for _, row in top_k.iterrows():
        combined = f"{row.get('title', '')} {row.get('skills', '')} {row.get('description', '')}".lower()
        if any(term in combined for term in target_terms if len(term) > 2):
            hits += 1

    return round(hits / len(top_k), 3)
