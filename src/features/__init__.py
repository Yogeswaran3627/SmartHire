"""Feature extraction and matching package."""
from src.features.text_features import (
    build_tfidf_vectorizer,
    fit_and_save_vectorizer,
    load_tfidf_vectorizer,
    compute_text_similarity,
    extract_top_keywords,
)
from src.features.match_features import (
    extract_skills,
    extract_years_experience,
    extract_education_level,
    skill_gap,
    compute_match_features,
    FEATURE_COLUMNS,
    SKILL_TAXONOMY,
)

__all__ = [
    "build_tfidf_vectorizer",
    "fit_and_save_vectorizer",
    "load_tfidf_vectorizer",
    "compute_text_similarity",
    "extract_top_keywords",
    "extract_skills",
    "extract_years_experience",
    "extract_education_level",
    "skill_gap",
    "compute_match_features",
    "FEATURE_COLUMNS",
    "SKILL_TAXONOMY",
]
