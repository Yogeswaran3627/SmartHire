"""
Unit test suite for SmartHire feature engineering, models, and parser.
"""
import unittest
import pandas as pd
from src.features.match_features import (
    extract_skills,
    extract_years_experience,
    extract_education_level,
    skill_gap,
    compute_match_features,
    FEATURE_COLUMNS
)
from src.features.text_features import build_tfidf_vectorizer, compute_text_similarity
from src.models.recommender import recommend_jobs, evaluate_precision_at_k
from src.models.classifier import load_classifier, predict_category
from src.models.fit_predictor import load_fit_predictor, predict_fit_score

class TestSmartHireFeatures(unittest.TestCase):
    def test_extract_skills(self):
        text = "Candidate has 5 years of experience in Python, SQL, Docker, Scikit-learn and Machine Learning."
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("docker", skills)
        self.assertIn("scikit-learn", skills)
        self.assertIn("machine learning", skills)

    def test_skill_gap(self):
        cand = "Expert in Python, Pandas, and MySQL."
        job = "Seeking Senior Data Scientist skilled in Python, SQL, Pandas, TensorFlow and Docker."
        result = skill_gap(cand, job)
        self.assertIn("python", result["matched_skills"])
        self.assertIn("pandas", result["matched_skills"])
        self.assertIn("docker", result["missing_skills"])
        self.assertIn("tensorflow", result["missing_skills"])
        self.assertGreater(result["coverage"], 0.0)
        self.assertLessEqual(result["coverage"], 1.0)

    def test_experience_extraction(self):
        text = "Overall 6.5 years of experience in backend development."
        exp = extract_years_experience(text)
        self.assertEqual(exp, 6.5)

    def test_education_level(self):
        text = "Completed Master of Science (M.Sc) in Computer Science after B.Tech."
        level = extract_education_level(text)
        self.assertEqual(level, 3)  # Master's level is 3

    def test_match_features_vector(self):
        cand = "Python developer with Django and PostgreSQL experience."
        job = "Looking for Python Django web engineer."
        features = compute_match_features(cand, job)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features)
            self.assertIsInstance(features[col], float)

    def test_text_similarity(self):
        vec = build_tfidf_vectorizer()
        vec.fit(["python data science machine learning", "software engineering web development"])
        sim = compute_text_similarity(vec, "python machine learning", "python data science")
        self.assertGreater(sim, 0.0)

    def test_recommender(self):
        jobs_df = pd.DataFrame([
            {"title": "Python Backend Engineer", "company": "Acme", "location": "Remote",
             "skills": "python django rest api", "experience": "3 years", "salary": "$100k",
             "job_text": "Python Backend Engineer with Django and REST APIs."},
            {"title": "Registered Nurse", "company": "HealthCare", "location": "NY",
             "skills": "healthcare nursing patient care", "experience": "2 years", "salary": "$80k",
             "job_text": "Registered Nurse providing clinical patient care."}
        ])
        cand = "Experienced Python software engineer proficient in Django and REST APIs."
        recs = recommend_jobs(cand, jobs_df, top_n=2)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs.iloc[0]["title"], "Python Backend Engineer")
        self.assertGreater(recs.iloc[0]["match_percent"], recs.iloc[1]["match_percent"])

    def test_trained_classifier_if_available(self):
        model = load_classifier()
        if model is not None:
            res = predict_category(model, "Accountant with experience in financial audit, general ledger, and tax reporting.")
            self.assertIn("predicted_category", res)
            self.assertIn("confidence_pct", res)
            self.assertGreater(len(res["top_predictions"]), 0)

    def test_trained_fit_predictor_if_available(self):
        cand = "Full stack developer with React, Node.js and MongoDB."
        job = "Hiring MERN Stack Engineer skilled in React, Node.js, Express, MongoDB."
        out = predict_fit_score(cand, job)
        self.assertIn("fit_percent", out)
        self.assertIn("status", out)
        self.assertIn("features", out)
        self.assertGreater(out["fit_percent"], 50.0)

if __name__ == "__main__":
    unittest.main()
