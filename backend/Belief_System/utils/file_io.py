import os
import pandas as pd
from pathlib import Path
 
 
# ── load_txt ──────────────────────────────────────────────────────────────────
 
def load_txt(path: str | Path) -> str:
    """
    Load a plain-text file and return its content as a string.
    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is empty.
    """
    txt_path = Path(path)
 
    if not txt_path.is_file():
        raise FileNotFoundError(f"Text file not found: {txt_path}")
 
    text = txt_path.read_text(encoding="utf-8").strip()
 
    if not text:
        raise ValueError(f"Text file is empty: {txt_path}")
 
    print(f"[load_txt] Characters loaded : {len(text):,}")
    print(f"[load_txt] Preview (first 300 chars):")
    print("-" * 60)
    print(text[:300])
 
    return text
 
 
# ── load_linkedin_xlsx ────────────────────────────────────────────────────────
 
def load_linkedin_xlsx(path: str | Path) -> pd.DataFrame:
    """
    Load a LinkedIn Campaign Manager export (.xlsx) and return
    a filtered DataFrame of organic posts only.
 
    The LinkedIn export uses a two-row header structure:
    - Row 1: metadata string (skipped via header=1)
    - Row 2: actual column names
 
    Only posts with Post type == 'Organic' and a non-empty Post title
    are retained.
 
    Returns columns: Post title, Created date, Engagement rate
    """
    xlsx_path = Path(path)
 
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"LinkedIn XLSX file not found: {xlsx_path}")
 
    # header=1 skips the LinkedIn metadata row
    df_raw = pd.read_excel(xlsx_path, sheet_name="All posts", header=1)
 
    print(f"[load_linkedin_xlsx] Raw rows loaded   : {len(df_raw):,}")
    print(f"[load_linkedin_xlsx] Columns available : {df_raw.columns.tolist()}")
 
    # Validate required columns
    REQUIRED_COLS = {"Post title", "Post type", "Created date", "Engagement rate"}
    missing = REQUIRED_COLS - set(df_raw.columns)
 
    if missing:
        raise KeyError(
            f"Expected columns not found in 'All posts' sheet: {missing}. "
            f"Available: {list(df_raw.columns)}"
        )
 
    print("[load_linkedin_xlsx] Column validation passed.")
 
    # Filter: organic posts with non-empty titles only
    df_organic = (
        df_raw[df_raw["Post type"] == "Organic"]
        .dropna(subset=["Post title"])
        [["Post title", "Created date", "Engagement rate"]]
        .reset_index(drop=True)
    )
 
    print(f"[load_linkedin_xlsx] Organic posts retained : {len(df_organic):,}")
 
    return df_organic
 
 
# ── save_txt ──────────────────────────────────────────────────────────────────
 
def save_txt(text: str, path: str | Path) -> None:
    """
    Write a string to a plain-text file.
    Creates parent directories if they do not exist.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[save_txt] Written : {out_path}  ({out_path.stat().st_size:,} bytes)")
 
 
# ── save_csv ──────────────────────────────────────────────────────────────────
 
def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to a CSV file.
    Creates parent directories if they do not exist.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[save_csv] Written : {out_path}  ({out_path.stat().st_size:,} bytes)")
 
 
# ── save_json ─────────────────────────────────────────────────────────────────
 
def save_json(data: list | dict, path: str | Path) -> None:
    """
    Write a list or dict to a JSON file with readable indentation.
    Creates parent directories if they do not exist.
    """
    import json
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save_json] Written : {out_path}  ({out_path.stat().st_size:,} bytes)")
 
 
# ── load_json ─────────────────────────────────────────────────────────────────
 
def load_json(path: str | Path) -> list | dict:
    """
    Load and return the contents of a JSON file.
    Raises FileNotFoundError if the file does not exist.
    """
    import json
    json_path = Path(path)
 
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
 
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"[load_json] Loaded : {json_path}")
    return data