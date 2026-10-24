import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure clear logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directory paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = BASE_DIR / "Data" / "External"
RAW_DIR = BASE_DIR / "Data" / "Raw"
OUTPUT_CSV_PATH = RAW_DIR / "ted_talks.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Authorized Public Archive Mirrors (CC BY-NC-SA 4.0 TED Talks Dataset: Metadata + Transcripts)
TED_MAIN_URLS = [
    "https://raw.githubusercontent.com/cpethe/TED_Talks/master/ted_main.csv",
    "https://raw.githubusercontent.com/MerkulovDaniil/optim/master/assets/Notebooks/ted_main.csv",
]

TRANSCRIPTS_URLS = [
    "https://raw.githubusercontent.com/cpethe/TED_Talks/master/transcripts.csv",
]


def ensure_directories() -> None:
    """Ensure External and Raw directories exist."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def extract_live_nextjs_sample() -> None:
    """
    Demonstrate parsing embedded __NEXT_DATA__ from the public ted.com/talks page
    while strictly respecting robots.txt (no /graphql, /api/, or /_next/data/ calls).
    Saves a proof-of-concept JSON snapshot in Scraped/Data/External/.
    """
    url = "https://www.ted.com/talks"
    sample_path = EXTERNAL_DIR / "ted_live_nextjs_sample.json"
    logger.info("Fetching live Next.js pageProps sample from %s ...", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        script_tag = soup.find("script", id="__NEXT_DATA__")
        if script_tag and script_tag.string:
            payload = json.loads(script_tag.string)
            talks_data = payload.get("props", {}).get("pageProps", {}).get("talks", [])
            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "source_url": url,
                        "robots_txt_compliant": True,
                        "talks_count_on_initial_page": len(talks_data) if isinstance(talks_data, list) else 0,
                        "sample_talks": talks_data[:5] if isinstance(talks_data, list) else talks_data,
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved live Next.js sample payload to %s", sample_path)
    except Exception as exc:
        logger.warning("Live Next.js sample extraction skipped or failed: %s", exc)


def download_with_fallback(urls: list[str], dest_path: Path) -> Path:
    """Download a file from a list of mirror URLs if not already cached locally."""
    if dest_path.exists() and dest_path.stat().st_size > 10_000:
        logger.info("Using cached file: %s (%.2f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))
        return dest_path

    for url in urls:
        try:
            logger.info("Downloading %s from %s ...", dest_path.name, url)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            logger.info(
                "Successfully downloaded %s (%.2f MB)",
                dest_path.name,
                dest_path.stat().st_size / (1024 * 1024),
            )
            return dest_path
        except requests.RequestException as exc:
            logger.warning("Mirror failed (%s): %s", url, exc)

    raise RuntimeError(f"Could not download {dest_path.name} from any mirror.")


def normalize_url(url_val: str) -> str:
    """Normalize TED talk URLs so metadata and transcripts join cleanly."""
    if pd.isna(url_val):
        return ""
    cleaned = str(url_val).strip().lower()
    cleaned = cleaned.replace("http://", "https://")
    cleaned = cleaned.split("?")[0].rstrip("/")
    return cleaned


def unix_to_iso_date(ts) -> str:
    """Convert Unix timestamp to YYYY-MM-DD string safely."""
    try:
        if pd.isna(ts):
            return ""
        return datetime.fromtimestamp(int(float(ts)), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


def build_raw_dataset() -> Path:
    """Download, merge, standardize, and save Scraped/Data/Raw/ted_talks.csv."""
    ensure_directories()
    extract_live_nextjs_sample()

    ted_main_path = download_with_fallback(TED_MAIN_URLS, EXTERNAL_DIR / "ted_main.csv")
    transcripts_path = download_with_fallback(TRANSCRIPTS_URLS, EXTERNAL_DIR / "transcripts.csv")

    logger.info("Loading ted_main.csv and transcripts.csv into pandas...")
    df_main = pd.read_csv(ted_main_path, encoding="utf-8", on_bad_lines="skip")
    df_trans = pd.read_csv(transcripts_path, encoding="utf-8", on_bad_lines="skip")

    logger.info("Raw shapes -> ted_main: %s, transcripts: %s", df_main.shape, df_trans.shape)

    # Normalize URLs for joining
    df_main["norm_url"] = df_main["url"].apply(normalize_url)
    df_trans["norm_url"] = df_trans["url"].apply(normalize_url)

    # Deduplicate transcripts on normalized URL before merging
    df_trans = df_trans[df_trans["norm_url"] != ""].drop_duplicates(subset=["norm_url"], keep="first")

    # Left-join transcripts onto main metadata
    merged = pd.merge(
        df_main,
        df_trans[["norm_url", "transcript"]],
        on="norm_url",
        how="left",
    )

    # Deduplicate by normalized URL
    merged = merged[merged["norm_url"] != ""].drop_duplicates(subset=["norm_url"], keep="first").reset_index(drop=True)

    # Build standardized schema required for Modules 1-6
    standardized = pd.DataFrame({
        "talk_id": range(1, len(merged) + 1),
        "title": merged["title"].fillna("").astype(str).str.strip(),
        "speaker": merged.get("main_speaker", pd.Series([""] * len(merged))).fillna("Unknown Speaker").astype(str).str.strip(),
        "speaker_occupation": merged.get("speaker_occupation", pd.Series([""] * len(merged))).fillna("").astype(str).str.strip(),
        "description": merged.get("description", pd.Series([""] * len(merged))).fillna("").astype(str).str.strip(),
        "tags": merged.get("tags", pd.Series(["[]"] * len(merged))).fillna("[]").astype(str).str.strip(),
        "transcript": merged["transcript"].fillna("").astype(str).str.strip(),
        "duration": pd.to_numeric(merged.get("duration"), errors="coerce").fillna(0).astype(int),
        "published_date": merged.get("published_date", pd.Series([None] * len(merged))).apply(unix_to_iso_date),
        "views": pd.to_numeric(merged.get("views"), errors="coerce").fillna(0).astype(int),
        "comments": pd.to_numeric(merged.get("comments"), errors="coerce").fillna(0).astype(int),
        "ratings": merged.get("ratings", pd.Series(["[]"] * len(merged))).fillna("[]").astype(str).str.strip(),
        "event": merged.get("event", pd.Series([""] * len(merged))).fillna("").astype(str).str.strip(),
        "url": merged["url"].fillna("").astype(str).str.strip(),
        "source": "TED_Public_Archive_CC_BY_NC_SA_4.0",
    })

    # Safe backup if raw dataset already exists
    if OUTPUT_CSV_PATH.exists() and OUTPUT_CSV_PATH.stat().st_size > 0:
        backup_path = RAW_DIR / "ted_talks_backup.csv"
        shutil.copy2(OUTPUT_CSV_PATH, backup_path)
        logger.info("Backed up existing raw dataset to %s", backup_path)

    # Save to Scraped/Data/Raw/ted_talks.csv
    standardized.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    logger.info("Saved standardized dataset to %s", OUTPUT_CSV_PATH)

    # Verify readability
    verify_df = pd.read_csv(OUTPUT_CSV_PATH, encoding="utf-8")
    non_empty_transcripts = (verify_df["transcript"].fillna("").str.len() > 0).sum()
    logger.info("VERIFICATION SUCCESS:")
    logger.info("  -> File location : %s", OUTPUT_CSV_PATH)
    logger.info("  -> Total records : %d", len(verify_df))
    logger.info("  -> With transcripts: %d (%.1f%%)", non_empty_transcripts, 100 * non_empty_transcripts / len(verify_df))
    logger.info("  -> Columns (%d)  : %s", len(verify_df.columns), list(verify_df.columns))

    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    build_raw_dataset()