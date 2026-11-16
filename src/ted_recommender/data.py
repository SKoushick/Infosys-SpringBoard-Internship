"""
Data loading, validation, and preprocessing utilities.

Handles:
- Loading raw, cleaned, and feature-engineered datasets
- Schema validation
- Missing value analysis
- Row alignment verification
- Data quality reporting
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    FEATURES_CSV_PATH,
)

logger = logging.getLogger(__name__)

# ── Required column schemas ─────────────────────────────────────────────
RAW_REQUIRED_COLUMNS = [
    "talk_id", "title", "main_speaker", "description",
    "tags", "url", "views", "duration",
]

FEATURE_REQUIRED_COLUMNS = [
    "talk_id", "title", "main_speaker", "description",
    "tags", "url", "views", "duration",
    "title_processed", "description_processed",
    "tags_processed", "transcript_processed",
    "combined_text", "has_transcript",
]


def load_dataset(
    path: Optional[Path] = None,
    stage: str = "features",
) -> pd.DataFrame:
    """
    Load a TED Talks dataset at the specified processing stage.

    Args:
        path: Explicit path to a CSV file. If None, uses default for stage.
        stage: One of 'raw', 'clean', 'preprocessed', 'features'.

    Returns:
        DataFrame with loaded data.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing.
    """
    if path is None:
        stage_paths = {
            "raw": RAW_DATA_PATH,
            "clean": CLEAN_DATA_PATH,
            "preprocessed": PREPROCESSED_DATA_PATH,
            "features": FEATURES_CSV_PATH,
        }
        if stage not in stage_paths:
            raise ValueError(f"Unknown stage '{stage}'. Use: {list(stage_paths.keys())}")
        path = stage_paths[stage]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run the appropriate module notebook to generate it."
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df)} records from {path.name}")

    # Validate required columns based on stage
    required = RAW_REQUIRED_COLUMNS if stage == "raw" else FEATURE_REQUIRED_COLUMNS
    if stage in ("raw", "features"):
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Dataset at {path} is missing required columns: {missing_cols}"
            )

    return df


def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive data quality checks on a TED Talks dataset.

    Returns a dictionary with validation results suitable for reporting.
    """
    report = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "duplicate_talk_ids": 0,
        "missing_titles": 0,
        "missing_descriptions": 0,
        "missing_tags": 0,
        "missing_transcripts": 0,
        "missing_urls": 0,
        "duplicate_urls": 0,
        "empty_text_fields": 0,
        "issues": [],
    }

    if "talk_id" in df.columns:
        dups = df["talk_id"].duplicated().sum()
        report["duplicate_talk_ids"] = int(dups)
        if dups > 0:
            report["issues"].append(f"{dups} duplicate talk_id values found")

    for col, key in [
        ("title", "missing_titles"),
        ("description", "missing_descriptions"),
        ("tags", "missing_tags"),
        ("transcript", "missing_transcripts"),
        ("url", "missing_urls"),
    ]:
        if col in df.columns:
            missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            report[key] = int(missing)
            if missing > 0 and col != "transcript":
                report["issues"].append(f"{missing} records missing '{col}'")

    if "url" in df.columns:
        dup_urls = df["url"].dropna().duplicated().sum()
        report["duplicate_urls"] = int(dup_urls)
        if dup_urls > 0:
            report["issues"].append(f"{dup_urls} duplicate URLs found")

    # Check for empty processed text fields
    text_cols = ["title_processed", "description_processed", "combined_text"]
    for col in text_cols:
        if col in df.columns:
            empty = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
            if empty > 0:
                report["empty_text_fields"] += int(empty)
                report["issues"].append(f"{empty} empty values in '{col}'")

    report["is_valid"] = len(report["issues"]) == 0
    return report


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a deterministic SHA-256 fingerprint from the sequence of talk IDs
    and essential columns to guarantee exact version compatibility.
    """
    import hashlib
    data_str = ",".join(str(int(tid)) for tid in df["talk_id"].values)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]


def verify_talk_id_alignment(
    df: pd.DataFrame,
    saved_talk_ids: np.ndarray,
    artifact_name: str = "artifact",
) -> bool:
    """
    Verify exact 1-to-1 ordered talk ID alignment between dataset and saved artifact.
    Guarantees that row i in a feature matrix or embedding array corresponds
    strictly to row i in the DataFrame.
    """
    if len(df) != len(saved_talk_ids):
        raise ValueError(
            f"Row count mismatch in {artifact_name}: DataFrame has {len(df)} rows, "
            f"but {artifact_name} has {len(saved_talk_ids)} talk IDs."
        )
    df_ids = df["talk_id"].values.astype(int)
    saved_ids = np.asarray(saved_talk_ids, dtype=int)
    mismatches = np.where(df_ids != saved_ids)[0]
    if len(mismatches) > 0:
        first_bad = mismatches[0]
        raise ValueError(
            f"Row ID mismatch in {artifact_name} at index {first_bad}: "
            f"DataFrame talk_id={df_ids[first_bad]} vs artifact talk_id={saved_ids[first_bad]}. "
            f"Total {len(mismatches)} misaligned rows. Re-run artifact generation."
        )
    return True


def verify_row_alignment(
    df: pd.DataFrame,
    matrix_rows: int,
    artifact_name: str = "matrix",
) -> bool:
    """
    Verify that a matrix has the same number of rows as the dataset.

    Args:
        df: The DataFrame used to create the matrix.
        matrix_rows: Number of rows in the matrix.
        artifact_name: Name for error messages.

    Returns:
        True if aligned, raises ValueError otherwise.
    """
    if len(df) != matrix_rows:
        raise ValueError(
            f"Row count mismatch: DataFrame has {len(df)} rows but "
            f"{artifact_name} has {matrix_rows} rows. "
            f"Regenerate the artifact from the current dataset."
        )
    return True


# ── Text preprocessing functions ────────────────────────────────────────

def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def clean_urls(text: str) -> str:
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def normalise_unicode(text: str) -> str:
    """Normalise Unicode characters to NFC form."""
    if not isinstance(text, str):
        return ""
    return unicodedata_normalize("NFC", text)


def clean_text_light(text: str) -> str:
    """
    Light text cleaning that preserves meaningful content.

    - Removes HTML tags and URLs
    - Lowercases
    - Removes TED transcription annotations like (Laughter), (Applause)
    - Collapses whitespace
    - Preserves punctuation, negation, and meaningful phrases
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = clean_html(text)
    text = clean_urls(text)
    # Remove TED transcription annotations
    text = re.sub(r"\([A-Za-z\s]+\)", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_aggressive(text: str) -> str:
    """
    Aggressive cleaning for TF-IDF: remove all non-alphanumeric characters.

    Use this for TF-IDF vectorisation where punctuation doesn't help.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = clean_html(text)
    text = clean_urls(text)
    text = re.sub(r"\([A-Za-z\s]+\)", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_tags(tags_raw) -> List[str]:
    """Parse stringified tag list into a clean list of lowercased tag names."""
    if pd.isna(tags_raw):
        return []
    if isinstance(tags_raw, list):
        return [str(t).strip().lower() for t in tags_raw if str(t).strip()]
    try:
        parsed = ast.literal_eval(str(tags_raw))
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed if str(t).strip()]
    except (ValueError, SyntaxError):
        pass
    # Fallback: split by comma
    return [t.strip().lower() for t in str(tags_raw).split(",") if t.strip()]


def tags_to_text(tags: List[str]) -> str:
    """Convert tag list to space-separated text for vectorisation."""
    return " ".join(tags)


def get_content_for_vectorisation(
    row: pd.Series,
    use_fallback: bool = True,
) -> str:
    """
    Build combined text content for a talk row, with fallback for missing transcripts.

    Priority: title + description + tags + transcript
    Fallback (no transcript): title + description(x2) + tags
    """
    parts = []

    title = str(row.get("title_processed", row.get("title", ""))).strip()
    if title:
        parts.append(title)

    desc = str(row.get("description_processed", row.get("description", ""))).strip()
    if desc:
        parts.append(desc)

    tags = str(row.get("tags_processed", row.get("tags_text", ""))).strip()
    if tags:
        parts.append(tags)

    transcript = str(row.get("transcript_processed", row.get("transcript", ""))).strip()
    has_transcript = bool(row.get("has_transcript", bool(transcript and len(transcript) > 50)))

    if has_transcript and transcript:
        parts.append(transcript)
    elif use_fallback and desc:
        # Double the description weight as fallback
        parts.append(desc)

    return " ".join(parts)


# Needed for normalise_unicode
try:
    from unicodedata import normalize as unicodedata_normalize
except ImportError:
    def unicodedata_normalize(form, text):
        return text

