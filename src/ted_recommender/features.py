"""
Feature engineering and vectorisation pipeline for TED Talks.

Handles:
- TF-IDF vectorisation for individual text fields (title, description, tags, transcript)
- Pretrained semantic embeddings generation using SentenceTransformers
- Combination and weighting of feature representations
- Artifact persistence and verification (row alignment with dataset)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import (
    FEATURES_CSV_PATH,
    TFIDF_MODELS_DIR,
    TFIDF_FEATURES_DIR,
    EMBEDDING_CACHE_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    FieldWeights,
    TfidfConfig,
)
from .data import get_content_for_vectorisation, verify_row_alignment

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Manages text vectorisation and semantic embeddings for TED Talks.
    Guarantees exact row-level alignment with the source dataset.
    """

    def __init__(
        self,
        tfidf_config: Optional[TfidfConfig] = None,
        field_weights: Optional[FieldWeights] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.tfidf_config = tfidf_config or TfidfConfig()
        self.field_weights = field_weights or FieldWeights()
        self.embedding_model_name = embedding_model_name

        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.tfidf_matrices: Dict[str, sparse.csr_matrix] = {}
        self.combined_tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        self.talk_ids: Optional[np.ndarray] = None

    def fit_transform_tfidf(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, sparse.csr_matrix], sparse.csr_matrix]:
        """
        Fit TF-IDF vectorizers across all text fields for all rows in df.
        Ensures NO rows are dropped (empty/missing fields become empty strings).
        """
        n_rows = len(df)
        self.talk_ids = df["talk_id"].values.copy()

        field_cols = {
            "title": "title_processed" if "title_processed" in df.columns else "title",
            "description": "description_processed" if "description_processed" in df.columns else "description",
            "tags": "tags_processed" if "tags_processed" in df.columns else "tags",
            "transcript": "transcript_processed" if "transcript_processed" in df.columns else "transcript",
        }

        max_features_map = {
            "title": self.tfidf_config.title_max_features,
            "description": self.tfidf_config.description_max_features,
            "tags": self.tfidf_config.tags_max_features,
            "transcript": self.tfidf_config.transcript_max_features,
        }

        weighted_matrices = []
        norm_weights = self.field_weights.normalised()
        weights_dict = {
            "title": norm_weights.title,
            "description": norm_weights.description,
            "tags": norm_weights.tags,
            "transcript": norm_weights.transcript,
        }

        for field_name, col_name in field_cols.items():
            logger.info(f"Fitting TF-IDF vectorizer for {field_name} (column: {col_name})...")
            # Fill NA with empty string to preserve exact row count
            corpus = df[col_name].fillna("").astype(str).tolist()

            vec = TfidfVectorizer(
                max_features=max_features_map[field_name],
                ngram_range=self.tfidf_config.ngram_range,
                min_df=self.tfidf_config.min_df,
                max_df=self.tfidf_config.max_df,
                sublinear_tf=self.tfidf_config.sublinear_tf,
                stop_words=self.tfidf_config.stop_words,
                dtype=self.tfidf_config.dtype,
            )

            mat = vec.fit_transform(corpus)
            self.vectorizers[field_name] = vec
            self.tfidf_matrices[field_name] = mat

            # Weight and normalize each field matrix
            weight = weights_dict[field_name]
            weighted_matrices.append(mat.multiply(weight))

        # Horizontally stack all weighted matrices
        combined = sparse.hstack(weighted_matrices, format="csr")
        # Apply L2 normalization across the stacked features for standard cosine similarity
        self.combined_tfidf_matrix = normalize(combined, norm="l2", copy=False)

        verify_row_alignment(df, self.combined_tfidf_matrix.shape[0], "combined_tfidf_matrix")
        logger.info(f"Combined TF-IDF matrix generated: {self.combined_tfidf_matrix.shape}")
        return self.tfidf_matrices, self.combined_tfidf_matrix

    def transform_query_tfidf(self, query_text: str, field: str = "combined") -> sparse.csr_matrix:
        """
        Transform arbitrary input text into TF-IDF vector space using fitted vectorizers.
        """
        if not self.vectorizers:
            raise RuntimeError("Vectorizers must be fitted or loaded before transforming query.")

        norm_weights = self.field_weights.normalised()
        weights_dict = {
            "title": norm_weights.title,
            "description": norm_weights.description,
            "tags": norm_weights.tags,
            "transcript": norm_weights.transcript,
        }

        if field in self.vectorizers:
            vec = self.vectorizers[field]
            mat = vec.transform([query_text])
            return normalize(mat, norm="l2")

        # Combined representation: query text transformed through each field vectorizer
        parts = []
        for f_name, vec in self.vectorizers.items():
            w = weights_dict[f_name]
            mat = vec.transform([query_text]).multiply(w)
            parts.append(mat)

        combined = sparse.hstack(parts, format="csr")
        return normalize(combined, norm="l2")

    def generate_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        force_regenerate: bool = False,
    ) -> np.ndarray:
        """
        Generate semantic sentence embeddings using SentenceTransformers.
        Uses cached embeddings if already available on disk.
        Uses fallback strategy for talks without transcripts (title + description + tags).
        """
        cache_file = EMBEDDING_CACHE_DIR / "ted_semantic_embeddings.npy"
        if cache_file.exists() and not force_regenerate:
            logger.info(f"Loading cached embeddings from {cache_file}...")
            embeddings = np.load(cache_file)
            if embeddings.shape[0] == len(df):
                self.embedding_matrix = embeddings
                return self.embedding_matrix
            logger.warning(f"Cached embedding row count ({embeddings.shape[0]}) does not match dataset ({len(df)}). Regenerating...")

        logger.info(f"Loading SentenceTransformer model '{self.embedding_model_name}'...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

        # Construct meaningful semantic summary texts for each talk:
        # Title + Description + Tags + first 1000 characters of transcript (if present)
        texts_to_embed = []
        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            desc = str(row.get("description", "")).strip()
            tags = str(row.get("tags_processed", row.get("tags", ""))).strip()
            transcript = str(row.get("transcript", "")).strip()
            has_t = bool(row.get("has_transcript", False))

            if has_t and transcript:
                # Truncate transcript to excerpt to keep semantic density high
                excerpt = transcript[:1200]
                text = f"{title}. {desc} Topics: {tags}. Excerpt: {excerpt}"
            else:
                text = f"{title}. {desc}. Summary: {desc} Topics: {tags}."
            texts_to_embed.append(text)

        logger.info(f"Encoding {len(texts_to_embed)} talk documents with batch size {batch_size}...")
        embeddings = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # Unit length for cosine similarity via dot product
        )

        self.embedding_matrix = np.asarray(embeddings, dtype=np.float32)
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, self.embedding_matrix)
        logger.info(f"Saved semantic embeddings to {cache_file}: shape {self.embedding_matrix.shape}")
        return self.embedding_matrix

    def transform_query_embedding(self, query_text: str) -> np.ndarray:
        """Embed an unseen text query using the pretrained embedding model."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.embedding_model_name)
        vec = model.encode([query_text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

    def save_artifacts(
        self,
        models_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Save fitted vectorizers, combined matrix, and artifact manifest."""
        m_dir = models_dir or TFIDF_MODELS_DIR
        f_dir = features_dir or TFIDF_FEATURES_DIR
        m_dir.mkdir(parents=True, exist_ok=True)
        f_dir.mkdir(parents=True, exist_ok=True)

        saved = {}
        for name, vec in self.vectorizers.items():
            p = m_dir / f"{name}_tfidf_vectorizer.joblib"
            joblib.dump(vec, p)
            saved[f"{name}_vectorizer"] = str(p)

        if self.combined_tfidf_matrix is not None:
            p = f_dir / "combined_tfidf_matrix.npz"
            sparse.save_npz(p, self.combined_tfidf_matrix)
            saved["combined_tfidf_matrix"] = str(p)

        if self.talk_ids is not None:
            p = f_dir / "talk_ids_manifest.npy"
            np.save(p, self.talk_ids)
            saved["talk_ids_manifest"] = str(p)

        manifest = {
            "n_records": int(self.combined_tfidf_matrix.shape[0]) if self.combined_tfidf_matrix is not None else 0,
            "feature_dim": int(self.combined_tfidf_matrix.shape[1]) if self.combined_tfidf_matrix is not None else 0,
            "fields": list(self.vectorizers.keys()),
            "saved_paths": saved,
        }
        with open(f_dir / "vectorization_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return saved

    def load_artifacts(
        self,
        models_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
    ) -> bool:
        """Load saved vectorizers and combined matrix from disk."""
        m_dir = models_dir or TFIDF_MODELS_DIR
        f_dir = features_dir or TFIDF_FEATURES_DIR

        fields = ["title", "description", "tags", "transcript"]
        for f in fields:
            p = m_dir / f"{f}_tfidf_vectorizer.joblib"
            if not p.exists():
                return False
            self.vectorizers[f] = joblib.load(p)

        matrix_p = f_dir / "combined_tfidf_matrix.npz"
        if not matrix_p.exists():
            return False
        self.combined_tfidf_matrix = sparse.load_npz(matrix_p)

        manifest_ids = f_dir / "talk_ids_manifest.npy"
        if manifest_ids.exists():
            self.talk_ids = np.load(manifest_ids)

        emb_p = EMBEDDING_CACHE_DIR / "ted_semantic_embeddings.npy"
        if emb_p.exists():
            self.embedding_matrix = np.load(emb_p)

        return True

