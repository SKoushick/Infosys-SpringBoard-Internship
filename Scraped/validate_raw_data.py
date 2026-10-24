from pathlib import Path
import json
import pandas as pd


def audit_raw_dataset(file_path: Path):
    if not file_path.exists():
        print(f"[ERROR] Target file not found at: {file_path}")
        return

    print("=" * 60)
    print(f"AUDITING RAW DATASET: {file_path.name}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    # 1. Dimensions
    total_records, total_columns = df.shape
    print(f"\n1. RECORD COUNT & SHAPE")
    print(f"   - Total rows: {total_records}")
    print(f"   - Total columns: {total_columns}")

    # 2. Schema and Null Values
    print(f"\n2. COLUMN INTEGRITY & MISSING VALUES")
    null_summary = df.isnull().sum()
    for col in df.columns:
        null_count = null_summary[col]
        null_pct = (null_count / total_records) * 100
        print(
            f"   - {col:18} | Type: {str(df[col].dtype):8} | Missing: {null_count:5} ({null_pct:5.2f}%)"
        )

    # 3. Key Content Checks
    print(f"\n3. CONTENT-BASED RECOMMENDER READINESS")
    has_title = "title" in df.columns
    has_transcript = "transcript" in df.columns
    has_tags = "tags" in df.columns

    print(f"   - Title field present:       {has_title}")
    print(f"   - Tags field present:        {has_tags}")
    print(f"   - Transcript field present:  {has_transcript}")

    if has_transcript:
        non_empty_transcripts = df["transcript"].dropna().str.strip()
        actual_transcripts = non_empty_transcripts[
            non_empty_transcripts.str.len() > 50
        ].count()
        print(
            f"   - Usable transcripts (>50 chars): {actual_transcripts} / {total_records}"
        )

    # 4. Identifier and Duplication Checks
    print(f"\n4. UNIQUENESS CHECKS")
    if "url" in df.columns:
        dup_urls = df["url"].duplicated().sum()
        print(f"   - Duplicate URLs: {dup_urls}")
    if "title" in df.columns:
        dup_titles = df["title"].duplicated().sum()
        print(f"   - Duplicate titles: {dup_titles}")

    print("\n" + "=" * 60)
    print("SAMPLE RECORD (FIRST ROW)")
    print("=" * 60)
    sample = df.iloc[0].to_dict()
    for k, v in sample.items():
        val_str = str(v)
        if len(val_str) > 100:
            val_str = val_str[:97] + "..."
        print(f"{k:18}: {val_str}")
    print("=" * 60)


if __name__ == "__main__":
    # Path relative to script execution in VS Code root
    raw_path = Path("Scraped/Data/Raw/ted_talks.csv")
    if not raw_path.exists():
        # Fallback if executed directly from inside Scraped/
        raw_path = Path("Data/Raw/ted_talks.csv")
    audit_raw_dataset(raw_path)