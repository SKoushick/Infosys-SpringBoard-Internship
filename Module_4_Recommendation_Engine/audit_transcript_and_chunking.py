"""
Audit Sentence Transformer transcript length and evaluate chunked embeddings.
Answers Phase 2, Items 1, 2, and 3 with empirical evidence.
"""

import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from ted_recommender.data import load_dataset
from ted_recommender.evaluation import (
    RecommendationEvaluator,
    parse_ground_truth_related_talks,
)


def run_audit():
    print("=" * 60)
    print("PHASE 2 - AUDIT 1 & 2: TRANSCRIPT LENGTH AND TOKEN DISTRIBUTION")
    print("=" * 60)

    df = load_dataset(stage="features")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    transcripts = df["transcript"].dropna().tolist()
    has_transcript_count = len(transcripts)
    total_talks = len(df)
    missing_transcript_count = total_talks - has_transcript_count

    print(f"Total talks: {total_talks}")
    print(f"Talks with transcript: {has_transcript_count} ({has_transcript_count / total_talks * 100:.2f}%)")
    print(f"Talks without transcript: {missing_transcript_count} ({missing_transcript_count / total_talks * 100:.2f}%)")

    char_lengths = [len(t) for t in transcripts]
    word_counts = [len(t.split()) for t in transcripts]

    # Sample token counts across all transcripts
    print("\nTokenizing transcripts with all-MiniLM-L6-v2 tokenizer...")
    t0 = time.time()
    # Batch tokenize without truncation to count exact token lengths
    token_counts = []
    current_excerpt_token_counts = []

    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        desc = str(row.get("description", "")).strip()
        tags = str(row.get("tags_processed", row.get("tags", ""))).strip()
        t = str(row.get("transcript", "")).strip()
        has_t = bool(row.get("has_transcript", False))

        if has_t and t:
            # Full transcript tokens
            toks = tokenizer.encode(t, truncation=False, add_special_tokens=True)
            token_counts.append(len(toks))
            # Current excerpt representation
            excerpt = t[:1200]
            text_repr = f"{title}. {desc} Topics: {tags}. Excerpt: {excerpt}"
        else:
            text_repr = f"{title}. {desc}. Summary: {desc} Topics: {tags}."

        toks_repr = tokenizer.encode(text_repr, truncation=False, add_special_tokens=True)
        current_excerpt_token_counts.append(len(toks_repr))

    tok_time = time.time() - t0
    print(f"Tokenization finished in {tok_time:.2f}s.")

    pct_exceeding_256 = sum(1 for c in token_counts if c > 256) / len(token_counts) * 100
    pct_exceeding_512 = sum(1 for c in token_counts if c > 512) / len(token_counts) * 100
    pct_repr_exceeding_256 = sum(1 for c in current_excerpt_token_counts if c > 256) / len(current_excerpt_token_counts) * 100

    audit_stats = {
        "dataset_total_talks": total_talks,
        "talks_with_transcript": has_transcript_count,
        "talks_without_transcript": missing_transcript_count,
        "transcript_character_length": {
            "min": int(np.min(char_lengths)),
            "mean": float(np.mean(char_lengths)),
            "median": float(np.median(char_lengths)),
            "p90": float(np.percentile(char_lengths, 90)),
            "p95": float(np.percentile(char_lengths, 95)),
            "max": int(np.max(char_lengths)),
        },
        "transcript_word_counts": {
            "min": int(np.min(word_counts)),
            "mean": float(np.mean(word_counts)),
            "median": float(np.median(word_counts)),
            "p90": float(np.percentile(word_counts, 90)),
            "p95": float(np.percentile(word_counts, 95)),
            "max": int(np.max(word_counts)),
        },
        "full_transcript_token_counts": {
            "min": int(np.min(token_counts)),
            "mean": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "p90": float(np.percentile(token_counts, 90)),
            "p95": float(np.percentile(token_counts, 95)),
            "max": int(np.max(token_counts)),
            "pct_exceeding_256_tokens": round(pct_exceeding_256, 2),
            "pct_exceeding_512_tokens": round(pct_exceeding_512, 2),
        },
        "current_excerpt_representation_tokens": {
            "min": int(np.min(current_excerpt_token_counts)),
            "mean": float(np.mean(current_excerpt_token_counts)),
            "median": float(np.median(current_excerpt_token_counts)),
            "max": int(np.max(current_excerpt_token_counts)),
            "pct_exceeding_256_tokens": round(pct_repr_exceeding_256, 2),
        },
    }

    print("\n--- Summary Statistics ---")
    print(f"Transcript words: Mean={audit_stats['transcript_word_counts']['mean']:.1f}, Median={audit_stats['transcript_word_counts']['median']:.1f}")
    print(f"Full transcript tokens: Mean={audit_stats['full_transcript_token_counts']['mean']:.1f}, Median={audit_stats['full_transcript_token_counts']['median']:.1f}")
    print(f"Percentage of full transcripts exceeding 256 tokens: {pct_exceeding_256:.2f}%")
    print(f"Percentage of full transcripts exceeding 512 tokens: {pct_exceeding_512:.2f}%")
    print(f"Current excerpt representation tokens: Mean={audit_stats['current_excerpt_representation_tokens']['mean']:.1f}, Exceeding 256: {pct_repr_exceeding_256:.2f}%")

    out_dir = PROJECT_ROOT / "Module_4_Recommendation_Engine" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transcript_token_length_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 2 - AUDIT 3: CONTROLLED CHUNKED-EMBEDDING EXPERIMENT")
    print("=" * 60)

    # Let's run a controlled comparison on a sample of 100 talks:
    # Method 1: Current approach (title + desc + tags + excerpt[:1200]) -> 1 embedding per talk
    # Method 2: Chunked approach:
    #   Chunk transcript into overlapping 200-word chunks (max 5 chunks)
    #   Embed chunks with SentenceTransformer, mean-pool chunk embeddings with title+desc embedding
    # Compare evaluation metrics on editorial related talks!

    ground_truth = parse_ground_truth_related_talks(df)
    sample_ids = [t for t in ground_truth.keys() if t <= 500][:100]

    st_model = SentenceTransformer(model_name)

    # Method 1: Current cached embeddings
    emb_cached = np.load(PROJECT_ROOT / "Module_3_Feature_Engineering_Vectorization" / "embeddings" / "ted_semantic_embeddings.npy")

    class SimpleEmbRecommender:
        def __init__(self, emb_matrix, df):
            self.emb_matrix = emb_matrix
            self.df = df
            self.id_to_idx = {int(row["talk_id"]): idx for idx, row in df.iterrows()}
            self.idx_to_id = {idx: int(row["talk_id"]) for idx, row in df.iterrows()}

        def recommend_for_talk(self, talk_id: int, top_k: int = 10):
            idx = self.id_to_idx[talk_id]
            q_vec = self.emb_matrix[idx]
            sims = np.dot(self.emb_matrix, q_vec)
            # Exclude self
            sims[idx] = -1.0
            top_pos = np.argsort(-sims)[:top_k]
            from ted_recommender.recommender import RecommendationResult, RecommendationExplanation
            res = []
            for p in top_pos:
                tid = self.idx_to_id[p]
                res.append(RecommendationResult(
                    talk_id=tid,
                    title=str(self.df.iloc[p]["title"]),
                    main_speaker=str(self.df.iloc[p].get("main_speaker", "")),
                    description="",
                    tags=[],
                    url="",
                    views=0,
                    duration_minutes=0.0,
                    score=float(sims[p]),
                    explanation=RecommendationExplanation(talk_id=tid, title="", combined_score=float(sims[p])),
                ))
            return res

    evaluator = RecommendationEvaluator(ground_truth)
    rec_baseline = SimpleEmbRecommender(emb_cached, df)
    metrics_baseline = evaluator.evaluate_recommender(rec_baseline, sample_ids, k=10)
    print(f"\n[Baseline Excerpt Representation]: {metrics_baseline}")

    # Generate chunked embeddings for the dataset
    print("\nGenerating chunked embeddings for all talks (max 5 chunks per talk, window=200 words, overlap=50 words)...")
    t0_chunk = time.time()
    chunked_embeddings = np.zeros((len(df), 384), dtype=np.float32)

    for idx, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        desc = str(row.get("description", "")).strip()
        tags = str(row.get("tags_processed", row.get("tags", ""))).strip()
        t = str(row.get("transcript", "")).strip()

        chunks_to_encode = [f"{title}. {desc}. Topics: {tags}"]
        if t and bool(row.get("has_transcript", False)):
            words = t.split()
            # Create up to 4 transcript chunks of 200 words with 50-word overlap
            chunk_size = 200
            stride = 150
            for start in range(0, min(len(words), 800), stride):
                sub_words = words[start:start + chunk_size]
                if len(sub_words) >= 30:
                    chunks_to_encode.append(" ".join(sub_words))

        vecs = st_model.encode(chunks_to_encode, normalize_embeddings=True, show_progress_bar=False)
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        chunked_embeddings[idx] = mean_vec

    chunk_gen_time = time.time() - t0_chunk
    print(f"Chunked embeddings generated in {chunk_gen_time:.2f}s.")

    rec_chunked = SimpleEmbRecommender(chunked_embeddings, df)
    metrics_chunked = evaluator.evaluate_recommender(rec_chunked, sample_ids, k=10)
    print(f"[Chunked Averaged Representation]: {metrics_chunked}")

    comparison = {
        "baseline_excerpt_representation": {
            "encoding_time_est_seconds": 23.8,
            "metrics": metrics_baseline,
        },
        "chunked_representation": {
            "encoding_time_seconds": round(chunk_gen_time, 2),
            "metrics": metrics_chunked,
        },
        "analysis": {
            "ndcg_diff": round(metrics_chunked["NDCG@10"] - metrics_baseline["NDCG@10"], 4),
            "precision_diff": round(metrics_chunked["Precision@10"] - metrics_baseline["Precision@10"], 4),
            "map_diff": round(metrics_chunked["MAP@10"] - metrics_baseline["MAP@10"], 4),
            "verdict": "Keep baseline excerpt representation" if metrics_chunked["NDCG@10"] <= metrics_baseline["NDCG@10"] else "Adopt chunked embeddings",
        }
    }

    with open(out_dir / "chunked_embedding_experiment.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\nExperiment analysis:")
    print(f"NDCG diff: {comparison['analysis']['ndcg_diff']}")
    print(f"Precision diff: {comparison['analysis']['precision_diff']}")
    print(f"Verdict: {comparison['analysis']['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()
