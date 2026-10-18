"""
Match feature engineering module for SmartHire.
Extracts domain skills, computes skill-gap analytics, experience match,
education match, and builds feature vectors for the Fit Predictor model.
"""
import re
from src.features.text_features import compute_text_similarity

# Comprehensive curated taxonomy across tech, data, business, finance, healthcare, design, sales
SKILL_TAXONOMY = [
    # Programming & Software
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "ruby", "php", "golang", "go",
    "rust", "swift", "kotlin", "scala", "dart", "perl", "bash", "shell scripting", "powershell",
    
    # Web & Frameworks
    "html", "html5", "css", "css3", "sass", "bootstrap", "tailwind", "react", "react.js", "next.js",
    "angular", "vue", "vue.js", "node.js", "express", "express.js", "django", "flask", "fastapi",
    "spring", "spring boot", "asp.net", "laravel", "ruby on rails", "graphql", "rest api", "microservices",
    
    # Data Science & Machine Learning
    "machine learning", "deep learning", "artificial intelligence", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "data science", "data analysis", "data analytics",
    "data mining", "predictive modeling", "statistical analysis", "pandas", "numpy", "scipy",
    "scikit-learn", "tensorflow", "keras", "pytorch", "xgboost", "lightgbm", "catboost", "opencv",
    "spacy", "nltk", "hugging face", "transformers", "llm", "genai", "prompt engineering",
    
    # Databases & Big Data
    "sql", "mysql", "postgresql", "sqlite", "oracle", "sql server", "mongodb", "cassandra",
    "redis", "elasticsearch", "neo4j", "spark", "apache spark", "hadoop", "hive", "kafka",
    "snowflake", "bigquery", "databricks", "etl", "data pipeline", "data warehousing",
    
    # Cloud, DevOps & Tools
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "ci/cd", "jenkins", "git", "github", "gitlab", "terraform", "ansible", "linux", "unix",
    "jira", "confluence", "agile", "scrum", "kanban", "devops",
    
    # BI & Visualization
    "power bi", "tableau", "looker", "excel", "advanced excel", "vba", "matplotlib", "seaborn", "plotly",
    
    # Cybersecurity & Networks
    "cybersecurity", "penetration testing", "ethical hacking", "network security", "information security",
    "firewall", "siem", "cryptography", "wireshark", "soc",
    
    # Business, Marketing & Operations
    "project management", "product management", "business analysis", "stakeholder management",
    "crm", "salesforce", "sap", "erp", "seo", "sem", "digital marketing", "content marketing",
    "social media marketing", "google analytics", "email marketing", "lead generation",
    "supply chain", "logistics", "procurement", "operations management",
    
    # Finance & Accounting
    "accounting", "financial analysis", "financial modeling", "budgeting", "forecasting", "auditing",
    "taxation", "quickbooks", "tally", "risk management", "investment banking", "portfolio management",
    "corporate finance", "compliance",
    
    # Human Resources & Soft Skills
    "recruitment", "talent acquisition", "hr operations", "employee relations", "onboarding",
    "payroll", "performance management", "training and development", "human resources",
    "leadership", "communication", "problem solving", "critical thinking", "teamwork", "negotiation"
]

EDUCATION_KEYWORDS = {
    "doctorate": ["phd", "ph.d", "doctorate", "doctoral"],
    "master": ["master", "masters", "m.tech", "mtech", "ms", "m.s", "mba", "mca", "m.sc", "msc", "m.e", "me"],
    "bachelor": ["bachelor", "bachelors", "b.tech", "btech", "b.e", "be", "bs", "b.s", "bba", "bca", "b.sc", "bsc", "b.com", "bcom"],
    "diploma": ["diploma", "associate", "polytechnic"]
}

def extract_skills(text: str) -> list[str]:
    """Extracts known technical and domain skills from arbitrary text using regex boundary matching."""
    text_lower = str(text).lower()
    found = set()
    for skill in SKILL_TAXONOMY:
        # Match skill with word boundary or boundary symbols
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)

def extract_years_experience(text: str) -> float:
    """Attempts to parse total years of professional experience from resume or job text."""
    text_lower = str(text).lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*years?",
        r"(\d+(?:\.\d+)?)\+?\s*years?(?:\s+of)?\s+(?:experience|exp)",
        r"(?:experience|exp)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*\+\s*years?"
    ]
    vals = []
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        for m in matches:
            if isinstance(m, tuple):
                for val in m:
                    if val:
                        try:
                            f = float(val)
                            if 0.5 <= f <= 40:
                                vals.append(f)
                        except ValueError:
                            pass
            else:
                try:
                    f = float(m)
                    if 0.5 <= f <= 40:
                        vals.append(f)
                except ValueError:
                    pass
    if vals:
        return float(max(vals))
    return 0.0

def extract_education_level(text: str) -> int:
    """
    Ordinal scale for highest detected education level:
    4: Doctorate, 3: Master's, 2: Bachelor's, 1: Diploma, 0: Unspecified
    """
    text_lower = str(text).lower()
    for level, words in [("doctorate", 4), ("master", 3), ("bachelor", 2), ("diploma", 1)]:
        for kw in EDUCATION_KEYWORDS[level]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return words
    return 0

def skill_gap(resume_text: str, job_text: str) -> dict:
    """
    Performs comprehensive skill-gap analysis comparing resume against job requirements.
    Returns matched skills, missing skills, coverage ratio, and tailored action advice.
    """
    cand_skills = set(extract_skills(resume_text))
    req_skills = set(extract_skills(job_text))

    matched = sorted(cand_skills & req_skills)
    missing = sorted(req_skills - cand_skills)
    coverage = len(matched) / len(req_skills) if req_skills else 1.0

    if req_skills:
        note = f"Candidate covers {len(matched)} of {len(req_skills)} detected job skills ({coverage * 100:.1f}%)."
    else:
        note = "No specific skills detected in job text from taxonomy. Qualitative fit applies."

    return {
        "candidate_skills": sorted(cand_skills),
        "required_skills": sorted(req_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "coverage": round(coverage, 3),
        "note": note
    }

def compute_match_features(
    resume_text: str,
    job_text: str,
    vectorizer=None
) -> dict:
    """
    Extracts the feature set used by the Shortlisting / Fit Predictor model:
    - skill_overlap (Jaccard)
    - skill_coverage
    - num_matched_skills
    - num_missing_skills
    - text_similarity (cosine similarity)
    - experience_match
    - education_match
    - length_ratio
    """
    gap = skill_gap(resume_text, job_text)
    cand_skills = set(gap["candidate_skills"])
    req_skills = set(gap["required_skills"])

    union = cand_skills | req_skills
    jaccard = len(cand_skills & req_skills) / len(union) if union else 0.0

    exp_resume = extract_years_experience(resume_text)
    exp_job = extract_years_experience(job_text)
    if exp_job > 0:
        exp_match = min(1.0, exp_resume / exp_job) if exp_resume >= 0.5 else 0.5
    else:
        exp_match = 0.8  # neutral default

    edu_resume = extract_education_level(resume_text)
    edu_job = extract_education_level(job_text)
    if edu_job > 0:
        edu_match = 1.0 if edu_resume >= edu_job else max(0.4, edu_resume / edu_job)
    else:
        edu_match = 0.85

    if vectorizer is not None:
        text_sim = compute_text_similarity(vectorizer, resume_text, job_text)
    else:
        # Fallback quick token overlap
        t1 = set(resume_text.lower().split())
        t2 = set(job_text.lower().split())
        text_sim = len(t1 & t2) / max(1, len(t1 | t2))

    len_ratio = min(1.0, len(resume_text) / max(100, len(job_text)))

    return {
        "skill_overlap": float(jaccard),
        "skill_coverage": float(gap["coverage"]),
        "num_matched_skills": float(len(gap["matched_skills"])),
        "num_missing_skills": float(len(gap["missing_skills"])),
        "text_similarity": float(text_sim),
        "experience_match": float(exp_match),
        "education_match": float(edu_match),
        "length_ratio": float(len_ratio)
    }

FEATURE_COLUMNS = [
    "skill_overlap",
    "skill_coverage",
    "num_matched_skills",
    "num_missing_skills",
    "text_similarity",
    "experience_match",
    "education_match",
    "length_ratio"
]
