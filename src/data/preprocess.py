"""
Data preprocessing and corpus merging pipeline for SmartHire.
Cleans raw resume text and merges heterogeneous job datasets into a unified corpus.
"""
import re
import html
from pathlib import Path
import pandas as pd
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLEAN_RESUMES_PATH,
    CLEAN_JOBS_PATH,
    MAX_JOB_ROWS,
)
from src.data.load_data import load_raw_resumes, discover_raw_files, read_dataset, find_column

def clean_text(text: str) -> str:
    """Strips HTML entities, tags, and excessive whitespace from text."""
    if not isinstance(text, str):
        return ""
    # Unescape HTML
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace common formatting characters
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove non-printable or corrupt Unicode characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_resumes(raw_df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """Cleans and validates labeled resume records."""
    if raw_df is None:
        raw_df, _ = load_raw_resumes()
    if raw_df is None or raw_df.empty:
        raise ValueError("No resume dataset available to preprocess.")

    df = raw_df.copy()
    df["resume_text"] = df["resume_text"].astype(str).apply(clean_text)
    df["category"] = df["category"].astype(str).str.strip().str.upper()

    # Drop empty or overly short resumes
    df = df[df["resume_text"].str.len() >= 30]
    df = df[df["category"].str.len() > 0]
    df = df.drop_duplicates(subset=["resume_text"]).reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_RESUMES_PATH, index=False, encoding="utf-8")
        print(f"Saved {len(df):,} cleaned resumes to {CLEAN_RESUMES_PATH}")

    return df

def merge_job_corpus(save: bool = True, max_rows: int = MAX_JOB_ROWS) -> pd.DataFrame:
    """
    Merges job listings from jobs.csv, Naukri ldjson, and LinkedIn postings
    into a standardized job corpus matching Section 2 of project specification:
    Columns: [title, company, location, skills, description, experience, salary, job_text, source]
    """
    records = []
    
    # 1. Process jobs/jobs.csv if present
    jobs_csv = RAW_DATA_DIR / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        try:
            df_jobs = pd.read_csv(jobs_csv, nrows=15000, encoding="utf-8", low_memory=False)
            t_col = find_column(df_jobs.columns, ["Job Title", "title"])
            s_col = find_column(df_jobs.columns, ["Key Skills", "skills"])
            e_col = find_column(df_jobs.columns, ["Job Experience Required", "experience"])
            sal_col = find_column(df_jobs.columns, ["Job Salary", "salary"])
            c_col = find_column(df_jobs.columns, ["Industry", "Functional Area", "Role Category"])

            out1 = pd.DataFrame()
            out1["title"] = df_jobs[t_col].fillna("Unknown Role").astype(str).apply(clean_text) if t_col else "Unknown Role"
            out1["company"] = df_jobs[c_col].fillna("Not disclosed").astype(str).apply(clean_text) if c_col else "Not disclosed"
            out1["location"] = "India"
            out1["skills"] = df_jobs[s_col].fillna("").astype(str).apply(clean_text) if s_col else ""
            out1["experience"] = df_jobs[e_col].fillna("Not specified").astype(str).apply(clean_text) if e_col else "Not specified"
            out1["salary"] = df_jobs[sal_col].fillna("Not disclosed").astype(str).apply(clean_text) if sal_col else "Not disclosed"
            out1["description"] = out1["title"] + " role. Required skills: " + out1["skills"] + ". Experience: " + out1["experience"]
            out1["source"] = "jobs.csv"
            records.append(out1)
            print(f"Loaded {len(out1):,} jobs from jobs.csv")
        except Exception as e:
            print(f"Notice: jobs.csv load note: {e}")

    # 2. Process Naukri ldjson if present
    naukri_files = list(RAW_DATA_DIR.rglob("*.ldjson")) + list(RAW_DATA_DIR.rglob("*naukri*.json*"))
    for n_file in naukri_files:
        try:
            df_naukri = read_dataset(n_file, nrows=15000)
            t_col = find_column(df_naukri.columns, ["job_title", "title"])
            c_col = find_column(df_naukri.columns, ["company_name", "company"])
            l_col = find_column(df_naukri.columns, ["city", "inferred_city", "location"])
            d_col = find_column(df_naukri.columns, ["job_description", "description"])
            cat_col = find_column(df_naukri.columns, ["category", "job_type"])
            sal_col = find_column(df_naukri.columns, ["salary_offered", "salary"])

            if t_col and d_col:
                out2 = pd.DataFrame()
                out2["title"] = df_naukri[t_col].fillna("Unknown Role").astype(str).apply(clean_text)
                out2["company"] = df_naukri[c_col].fillna("Confidential").astype(str).apply(clean_text) if c_col else "Confidential"
                out2["location"] = df_naukri[l_col].fillna("India").astype(str).apply(clean_text) if l_col else "India"
                out2["skills"] = df_naukri[cat_col].fillna("").astype(str).apply(clean_text) if cat_col else ""
                out2["experience"] = "Not specified"
                out2["salary"] = df_naukri[sal_col].fillna("Not disclosed").astype(str).apply(clean_text) if sal_col else "Not disclosed"
                out2["description"] = df_naukri[d_col].fillna("").astype(str).apply(clean_text)
                out2["source"] = "naukri"
                records.append(out2)
                print(f"Loaded {len(out2):,} jobs from {n_file.name}")
                break
        except Exception as e:
            print(f"Notice: Naukri load note: {e}")

    # 3. Process LinkedIn postings if present
    li_file = RAW_DATA_DIR / "linkedin" / "postings.csv"
    if li_file.exists():
        try:
            df_li = pd.read_csv(li_file, nrows=15000, encoding="utf-8", low_memory=False)
            t_col = find_column(df_li.columns, ["title", "job_title"])
            c_col = find_column(df_li.columns, ["company_name", "company"])
            l_col = find_column(df_li.columns, ["location", "city"])
            d_col = find_column(df_li.columns, ["description", "job_description"])
            s_col = find_column(df_li.columns, ["skills_desc", "skills"])
            e_col = find_column(df_li.columns, ["formatted_experience_level", "experience"])

            if t_col and d_col:
                out3 = pd.DataFrame()
                out3["title"] = df_li[t_col].fillna("Unknown Role").astype(str).apply(clean_text)
                out3["company"] = df_li[c_col].fillna("Not disclosed").astype(str).apply(clean_text) if c_col else "Not disclosed"
                out3["location"] = df_li[l_col].fillna("Remote / Various").astype(str).apply(clean_text) if l_col else "Remote / Various"
                out3["skills"] = df_li[s_col].fillna("").astype(str).apply(clean_text) if s_col else ""
                out3["experience"] = df_li[e_col].fillna("Mid-Senior").astype(str).apply(clean_text) if e_col else "Not specified"
                out3["salary"] = "Not disclosed"
                out3["description"] = df_li[d_col].fillna("").astype(str).apply(clean_text)
                out3["source"] = "linkedin"
                records.append(out3)
                print(f"Loaded {len(out3):,} jobs from LinkedIn postings.csv")
        except Exception as e:
            print(f"Notice: LinkedIn load note: {e}")

    if not records:
        raise ValueError("Could not find or load any raw job postings datasets.")

    corpus = pd.concat(records, ignore_index=True)
    # Composite job text used for TF-IDF vectorization and recommendation
    corpus["job_text"] = (
        corpus["title"] + " " +
        corpus["skills"] + " " +
        corpus["experience"] + " " +
        corpus["location"] + " " +
        corpus["description"]
    ).apply(clean_text)

    # Filter out empty or truncated listings
    corpus = corpus[corpus["job_text"].str.len() >= 30]
    corpus = corpus.drop_duplicates(subset=["title", "company", "location"]).reset_index(drop=True)

    if len(corpus) > max_rows:
        corpus = corpus.sample(n=max_rows, random_state=42).reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        corpus.to_csv(CLEAN_JOBS_PATH, index=False, encoding="utf-8")
        print(f"Saved {len(corpus):,} unified jobs corpus to {CLEAN_JOBS_PATH}")

    return corpus

def build_full_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs end-to-end preprocessing pipeline for both resumes and jobs."""
    resumes_df = preprocess_resumes(save=True)
    jobs_df = merge_job_corpus(save=True)
    return resumes_df, jobs_df
