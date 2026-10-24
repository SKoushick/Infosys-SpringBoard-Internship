import ast
from pathlib import Path
import re
import pandas as pd


def parse_string_list(val):
    """Safely converts stringified lists/dicts to native Python objects."""
    if pd.isna(val):
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def clean_text_for_nlp(text: str) -> str:
    """Normalizes transcript and description text for NLP feature extraction."""
    if not isinstance(text, str):
        return ""

    # 1. Remove TED transcription annotations like (Laughter), (Applause), (Music)
    text = re.sub(r"\([A-Za-z\s]+\)", " ", text)

    # 2. Lowercase conversion
    text = text.lower()

    # 3. Remove punctuation and non-alphanumeric characters (keep basic whitespace)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 4. Collapse multiple spaces, newlines, and tabs into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_tag_names(tags_raw):
    """Extracts a clean list of lowercased tag names."""
    parsed = parse_string_list(tags_raw)
    if isinstance(parsed, list):
        return [str(t).strip().lower() for t in parsed if str(t).strip()]
    return []


def extract_related_talk_ids(related_raw):
    """Extracts integer IDs of the ground-truth related talks."""
    parsed = parse_string_list(related_raw)
    ids = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                ids.append(item["id"])
    return ids


def run_preprocessing_pipeline(raw_path: Path, output_path: Path):
    print("=" * 60)
    print("RUNNING MODULE 2 PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load Raw Dataset
    print(f"[1/6] Loading raw dataset: {raw_path}...")
    df = pd.read_csv(raw_path)
    total_talks = len(df)
    print(f"      Loaded {total_talks} records.")

    # 2. Handle Missing Categorical Values
    print("[2/6] Imputing missing metadata fields...")
    df["speaker_occupation"] = df["speaker_occupation"].fillna("Unknown")
    df["main_speaker"] = df["main_speaker"].fillna("Unknown")
    df["description"] = df["description"].fillna("")

    # 3. Parse Nested Columns
    print(
        "[3/6] Parsing nested list structures (tags, related_talks, ratings)..."
    )
    df["clean_tags"] = df["tags"].apply(extract_tag_names)
    df["tags_joined"] = df["clean_tags"].apply(lambda tags: " ".join(tags))
    df["related_talk_ids"] = df["related_talks"].apply(extract_related_talk_ids)

    # 4. Clean Speech Transcripts & Build Fallback Strategy
    print("[4/6] Normalizing speech transcripts & handling nulls...")
    df["has_transcript"] = df["transcript"].notna() & (
        df["transcript"].str.strip().str.len() > 50
    )
    missing_transcript_count = (~df["has_transcript"]).sum()
    print(
        f"      Transcripts present: {df['has_transcript'].sum()} | Missing fallback needed: {missing_transcript_count}"
    )

    df["clean_transcript"] = df["transcript"].fillna("").apply(clean_text_for_nlp)
    df["clean_description"] = df["description"].apply(clean_text_for_nlp)
    df["clean_title"] = df["title"].apply(clean_text_for_nlp)
    df["clean_speaker"] = (
        df["main_speaker"].astype(str).str.lower().str.replace(" ", "_")
    )

    # 5. Build Comprehensive Content Soup
    print(
        "[5/6] Engineering unified 'content_soup' for TF-IDF / Embedding vectorization..."
    )

    def assemble_soup(row):
        # Weighting: Title (x2) + Speaker + Tags (x3) + Description + Transcript
        tags_weighted = " ".join([f"{tag}" for tag in row["clean_tags"]] * 2)

        # If transcript is available, use description + transcript; otherwise rely on description heavily
        if row["has_transcript"]:
            body_text = f"{row['clean_description']} {row['clean_transcript']}"
        else:
            body_text = f"{row['clean_description']} {row['clean_description']}"

        soup = f"{row['clean_title']} {row['clean_title']} {row['clean_speaker']} {tags_weighted} {body_text}"
        return re.sub(r"\s+", " ", soup).strip()

    df["content_soup"] = df.apply(assemble_soup, axis=1)

    # Calculate word and token metrics for EDA
    df["transcript_word_count"] = df["clean_transcript"].apply(
        lambda x: len(x.split()) if x else 0
    )
    df["soup_word_count"] = df["content_soup"].apply(lambda x: len(x.split()))
    df["duration_minutes"] = (df["duration"] / 60).round(2)

    # Convert timestamps to ISO readable dates
    df["published_date_formatted"] = pd.to_datetime(
        df["published_date"], unit="s", errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # 6. Save Clean Dataset
    print(f"[6/6] Saving processed dataset to: {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nPREPROCESSING COMPLETE!")
    print(f"- Output Rows:    {len(df)}")
    print(f"- Output Columns: {df.shape[1]}")
    print(f"- Missing Values in content_soup: {df['content_soup'].isna().sum()}")
    print(
        f"- Avg Transcript Words: {df[df['has_transcript']]['transcript_word_count'].mean():.1f}"
    )
    print(f"- Avg Content Soup Words: {df['soup_word_count'].mean():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "Data" / "Raw" / "ted_talks.csv"
    processed_csv = base_dir / "Data" / "Processed" / "ted_talks_clean.csv"

    run_preprocessing_pipeline(raw_csv, processed_csv)