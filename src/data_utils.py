from openpyxl.drawing import spreadsheet_drawing
from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS, MAX_JOB_ROWS, CLEAN_JOBS_PATH
from src.data.load_data import read_dataset, discover_raw_files as discover_files, find_column, norm_name as norm

def discover_files():
    if not RAW_DATA_DIR.exists():
        return []
    return sorted(p for p in RAW_DATA_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                  and not p.name.startswith("~$"))

def norm(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

def find_column(columns, candidates):
    lookup = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    for c in columns:
        for candidate in candidates:
            if norm(candidate) in norm(c) and len(norm(candidate)) >= 4:
                return c
    return None

def inspect_all():
    results = []
    for path in discover_files():
        try:
            df = read_dataset(path, nrows=1000)
            results.append({"file": str(path.relative_to(RAW_DATA_DIR)),
                            "sample_rows": len(df), "columns": list(df.columns),
                            "missing": int(df.isna().sum().sum()), "error": ""})
        except Exception as exc:
            results.append({"file": str(path.relative_to(RAW_DATA_DIR)),
                            "sample_rows": 0, "columns": [], "missing": 0,
                            "error": str(exc)})
    return results

def identify_resume_dataset():
    candidates = []
    for path in discover_files():
        try:
            sample = read_dataset(path, nrows=5000)
        except Exception:
            continue
        text_col = find_column(sample.columns, ["Resume_str", "resume_text", "resume", "CV", "text", "content"])
        label_col = find_column(sample.columns, ["Category", "job category", "label", "target", "profession", "role"])
        if text_col and label_col and text_col != label_col:
            score = (3 if "resume" in path.name.lower() else 0) + (2 if "category" in norm(label_col) else 0)
            candidates.append((score, path, text_col, label_col))
    if not candidates:
        return None
    _, path, text_col, label_col = max(candidates, key=lambda item: item[0])
    df = read_dataset(path)
    df = df[[text_col, label_col]].rename(columns={text_col: "resume_text", label_col: "category"})
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df = df[(df["resume_text"].str.len() > 20) & (df["category"] != "")]
    df = df.drop_duplicates(subset=["resume_text"]).reset_index(drop=True)
    return {"path": path, "data": df, "text_col": text_col, "label_col": label_col}

def load_job_corpus():
    records, notes = [], []
    for path in discover_files():
        try:
            df = read_dataset(path, nrows=MAX_JOB_ROWS)
        except Exception as exc:
            notes.append(f"{path.name}: read error: {exc}")
            continue
        # Skip obvious labelled-resume data.
        if find_column(df.columns, ["Resume_str", "resume_text"]) and find_column(df.columns, ["Category"]):
            continue
        title = find_column(df.columns, ["title", "job_title", "position", "role", "designation"])
        desc = find_column(df.columns, ["description", "job_description", "job_details", "summary", "details"])
        if not title and not desc:
            continue
        mapping = {
            "title": (title, "Job title not provided"),
            "company": (find_column(df.columns, ["company", "company_name", "employer"]), "Not provided"),
            "location": (find_column(df.columns, ["location", "city", "job_location"]), "Not provided"),
            "description": (desc, ""),
            "skills": (find_column(df.columns, ["skills", "required_skills", "job_skills", "skillset"]), ""),
            "experience": (find_column(df.columns, ["experience", "experience_level", "years_experience"]), "Not specified"),
            "salary": (find_column(df.columns, ["salary", "salary_range", "pay", "compensation"]), "Not provided"),
        }
        # pyrefly: ignore [unknown-name]
        out = pd.DataFrame(index=df.index)
        for target, (source, default) in mapping.items():
            out[target] = df[source].fillna("").astype(str) if source else default
        out["source_file"] = str(path.relative_to(RAW_DATA_DIR))
        out["job_text"] = (out["title"] + " " + out["description"] + " " + out["skills"] + " " + out["experience"]).str.strip()
        out = out[out["job_text"].str.len() > 5]
        if len(out):
            records.append(out)
            notes.append(f"{path.name}: loaded {len(out):,} job rows")
    if not records:
        # pyrefly: ignore [unknown-name]
        return pd.DataFrame(columns=["title","company","location","description","skills","experience","salary","source_file","job_text"]), notes
    # pyrefly: ignore [unknown-name]
    corpus = pd.concat(records, ignore_index=True)
    corpus = corpus.drop_duplicates(subset=["title", "company", "location", "job_text"]).reset_index(drop=True)
    return corpus, notes
