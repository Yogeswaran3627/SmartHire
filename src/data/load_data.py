"""
Data loading utilities for SmartHire.
Reads raw resume datasets and job postings datasets with robust format support.
"""
from pathlib import Path
import pandas as pd
import os
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLEAN_RESUMES_PATH,
    CLEAN_JOBS_PATH,
    SUPPORTED_EXTENSIONS,
    MAX_JOB_ROWS,
)

def read_dataset(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    """Reads dataset from CSV, Excel, JSON, or LDJSON with multi-encoding fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_error = None
        for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                return pd.read_csv(path, nrows=nrows, encoding=enc, low_memory=False)
            except Exception as e:
                last_error = e
        # pyrefly: ignore [bad-raise]
        raise last_error

    if suffix == ".xlsx":
        return pd.read_excel(path, nrows=nrows)

    if suffix in {".jsonl", ".ldjson"}:
        try:
            return pd.read_json(path, lines=True, nrows=nrows)
        except (ValueError, TypeError):
            # In case nrows is not supported in some pandas version for jsonl
            df = pd.read_json(path, lines=True)
            return df.head(nrows) if nrows else df

    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported file format: {suffix}")

def discover_raw_files() -> list[Path]:
    """Discover all supported dataset files under data/raw/."""
    if not RAW_DATA_DIR.exists():
        return []
    return sorted(
        p for p in RAW_DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$") and not p.name.startswith(".")
    )

def norm_name(s: str) -> str:
    """Normalize string for column matching."""
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive fuzzy match candidate column names."""
    lookup = {norm_name(c): c for c in columns}
    for cand in candidates:
        if norm_name(cand) in lookup:
            return lookup[norm_name(cand)]
    for c in columns:
        for cand in candidates:
            if norm_name(cand) in norm_name(c) and len(norm_name(cand)) >= 4:
                return c
    return None

def load_raw_resumes(nrows: int | None = None) -> tuple[pd.DataFrame | None, Path | None]:
    """
    Identifies and loads the primary resume dataset with text and category label.
    Prefers Resume.csv with Resume_str / Category.
    """
    candidates = []
    for path in discover_raw_files():
        if "resume" in path.name.lower() and path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=50, encoding="utf-8", errors="ignore")
                text_col = find_column(sample.columns, ["Resume_str", "resume_text", "resume", "CV", "text"])
                cat_col = find_column(sample.columns, ["Category", "category", "label", "target", "role"])
                if text_col and cat_col:
                    candidates.append((10, path, text_col, cat_col))
            except Exception:
                continue

    if not candidates:
        # Fallback to any file with matching columns
        for path in discover_raw_files():
            try:
                sample = read_dataset(path, nrows=50)
                text_col = find_column(sample.columns, ["Resume_str", "resume_text", "resume", "CV", "text"])
                cat_col = find_column(sample.columns, ["Category", "category", "label", "target"])
                if text_col and cat_col and text_col != cat_col:
                    candidates.append((5, path, text_col, cat_col))
            except Exception:
                continue

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_path, text_col, cat_col = candidates[0]
    df = read_dataset(best_path, nrows=nrows)
    df = df[[text_col, cat_col]].rename(columns={text_col: "resume_text", cat_col: "category"})
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    return df, best_path

def load_raw_jobs(nrows: int = MAX_JOB_ROWS) -> list[pd.DataFrame]:
    """Loads raw job frames from discoverable job files."""
    frames = []
    for path in discover_raw_files():
        if "resume" in path.name.lower():
            continue
        try:
            df = read_dataset(path, nrows=nrows)
            frames.append(df)
        except Exception:
            continue
    return frames

def load_processed_data() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Loads pre-processed and model-ready resumes and jobs datasets if available.
    """
    resumes_df = None
    jobs_df = None
    if CLEAN_RESUMES_PATH.exists():
        try:
            resumes_df = pd.read_csv(CLEAN_RESUMES_PATH, encoding="utf-8")
        except Exception:
            pass

    if CLEAN_JOBS_PATH.exists():
        try:
            jobs_df = pd.read_csv(CLEAN_JOBS_PATH, encoding="utf-8")
        except Exception:
            pass

    return resumes_df, jobs_df

