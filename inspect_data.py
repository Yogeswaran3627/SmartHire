import os
from src.config import RAW_DATA_DIR
from src.data_utils import discover_files, read_dataset

print("\n========== SMARTHIRE DATA INSPECTION ==========\n")
paths = discover_files()
if not paths:
    print("No CSV/XLSX/JSON files found under:", RAW_DATA_DIR)
    print("Extract the Kaggle ZIPs and put the extracted files in data/raw/.")
    raise SystemExit
for path in paths:
    print("\n" + "=" * 65)
    print("FILE:", path.relative_to(RAW_DATA_DIR))
    print("SIZE:", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")
    try:
        df = read_dataset(path, nrows=1000)
        print("SAMPLE SHAPE:", df.shape, "(up to 1,000 rows)")
        print("COLUMNS:", list(df.columns))
        print("MISSING VALUES IN SAMPLE:")
        print(df.isna().sum().to_string())
        print("FIRST 3 ROWS:")
        print(df.head(3).to_string(index=False, max_cols=15))
    except Exception as exc:
        print("Could not read file:", exc)
