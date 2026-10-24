from pathlib import Path
import pandas as pd


def build_master_dataset(
    main_path: Path, transcripts_path: Path, output_path: Path
):
    print("=" * 60)
    print("BUILDING MASTER RAW TED TALKS DATASET")
    print("=" * 60)


    print(f"[1/4] Loading {main_path.name}...")
    df_main = pd.read_csv(main_path)
    print(
        f"      Found {len(df_main)} records across {df_main.shape[1]} columns."
    )

    print(f"[2/4] Loading {transcripts_path.name}...")
    df_transcripts = pd.read_csv(transcripts_path)
    print(f"      Found {len(df_transcripts)} transcripts.")

    df_main["url"] = df_main["url"].astype(str).str.strip()
    df_transcripts["url"] = df_transcripts["url"].astype(str).str.strip()

    # Drop duplicate transcript URLs if any exist in the external file
    df_transcripts = df_transcripts.drop_duplicates(subset=["url"])

    print("[3/4] Merging datasets on talk URL...")
    merged_df = pd.merge(df_main, df_transcripts, on="url", how="left")

    merged_df.insert(0, "talk_id", range(1, len(merged_df) + 1))
    merged_df["source"] = "TED_Public_Archive_CC_BY_NC_SA_4.0"

    primary_columns = [
        "talk_id",
        "title",
        "main_speaker",
        "speaker_occupation",
        "description",
        "tags",
        "transcript",
        "related_talks",
        "ratings",
        "views",
        "comments",
        "duration",
        "published_date",
        "event",
        "url",
        "source",
    ]

    final_cols = [c for c in primary_columns if c in merged_df.columns]
    extra_cols = [c for c in merged_df.columns if c not in final_cols]
    final_df = merged_df[final_cols + extra_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[4/4] Master raw dataset saved successfully to:\n      {output_path}")
    print(f"      Total records: {len(final_df)}")
    print(f"      Total features: {final_df.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    main_file = base_dir / "Data" / "External" / "ted_main.csv"
    transcripts_file = base_dir / "Data" / "External" / "transcripts.csv"
    raw_output = base_dir / "Data" / "Raw" / "ted_talks.csv"

    build_master_dataset(main_file, transcripts_file, raw_output)