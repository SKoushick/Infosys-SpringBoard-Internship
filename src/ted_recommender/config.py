"""
Centralised configuration for the TED Talks Recommendation System.

All paths are relative to the project root. No hard-coded personal paths.
All hyperparameters are documented with rationale.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import json
import numpy as np


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains requirements.txt)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "requirements.txt").exists():
            return current
        current = current.parent
    # Fallback: assume two levels up from src/ted_recommender/
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _find_project_root()

# ── Module directories ──────────────────────────────────────────────────
MODULE_1_DIR = PROJECT_ROOT / "Module_1_Data_Collection"
MODULE_2_DIR = PROJECT_ROOT / "Module_2_Data_Exploration_Preprocessing"
MODULE_3_DIR = PROJECT_ROOT / "Module_3_Feature_Engineering_Vectorization"
MODULE_4_DIR = PROJECT_ROOT / "Module_4_Recommendation_Engine"

# ── Data paths ──────────────────────────────────────────────────────────
RAW_DATA_PATH = MODULE_1_DIR / "Scraped" / "Data" / "Raw" / "ted_talks.csv"
CLEAN_DATA_PATH = MODULE_1_DIR / "Scraped" / "Data" / "Processed" / "ted_talks_clean.csv"
PREPROCESSED_DATA_PATH = MODULE_1_DIR / "Scraped" / "Data" / "Processed" / "ted_talks_preprocessed.csv"
FEATURES_CSV_PATH = MODULE_3_DIR / "features" / "ted_talks_features.csv"

# ── Artifact paths ──────────────────────────────────────────────────────
TFIDF_MODELS_DIR = MODULE_3_DIR / "models"
TFIDF_FEATURES_DIR = MODULE_3_DIR / "features"
EMBEDDING_CACHE_DIR = MODULE_3_DIR / "embeddings"
MODULE_4_MODELS_DIR = MODULE_4_DIR / "models"
MODULE_4_OUTPUTS_DIR = MODULE_4_DIR / "outputs"

# ── Random seed ─────────────────────────────────────────────────────────
RANDOM_SEED = 42


@dataclass
class TfidfConfig:
    """
    TF-IDF vectorisation parameters.

    Rationale:
    - max_features=30000: Sufficient vocabulary for 2550 talks; captures
      long-tail terms without excessive dimensionality.
    - ngram_range=(1,2): Bigrams capture meaningful phrases like
      "climate change", "artificial intelligence".
    - min_df=2: Terms appearing in only 1 document are likely noise/typos.
    - max_df=0.90: Terms in >90% of documents are near-stopwords.
    - sublinear_tf=True: Dampens the effect of high term frequencies
      (uses 1 + log(tf)), improving performance on long transcripts.
    - L2 normalisation is applied post-hoc for cosine similarity.
    """
    max_features: int = 30000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.90
    sublinear_tf: bool = True
    stop_words: str = "english"
    dtype: type = field(default=np.float32, repr=False)

    # Field-specific overrides (smaller max_features for short fields)
    title_max_features: int = 5000
    description_max_features: int = 10000
    tags_max_features: int = 5000
    transcript_max_features: int = 30000

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "stop_words": self.stop_words,
            "title_max_features": self.title_max_features,
            "description_max_features": self.description_max_features,
            "tags_max_features": self.tags_max_features,
            "transcript_max_features": self.transcript_max_features,
        }


@dataclass
class FieldWeights:
    """
    Weights for combining different text field similarities.

    These are initial engineering estimates, NOT optimised values.
    Rationale:
    - Tags (0.30): Strongest topical signal; curated by TED editors.
    - Description (0.25): Concise summary of talk content.
    - Transcript (0.25): Rich content signal but noisy (filler words, etc.).
    - Title (0.20): Short but highly informative; users scan titles first.

    These weights are configurable and should be tuned with relevance
    judgments when available (see Module 5).
    """
    title: float = 0.20
    description: float = 0.25
    tags: float = 0.30
    transcript: float = 0.25

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "transcript": self.transcript,
        }

    def normalised(self) -> "FieldWeights":
        """Return a copy with weights normalised to sum to 1.0."""
        total = self.title + self.description + self.tags + self.transcript
        if total == 0:
            return FieldWeights(0.25, 0.25, 0.25, 0.25)
        return FieldWeights(
            title=self.title / total,
            description=self.description / total,
            tags=self.tags / total,
            transcript=self.transcript / total,
        )


@dataclass
class RecommenderConfig:
    """Global configuration for the recommendation engine."""
    # Default number of recommendations
    top_k: int = 10

    # Diversity: MMR lambda parameter (1.0 = pure relevance, 0.0 = pure diversity)
    mmr_lambda: float = 0.7

    # Hybrid weights between TF-IDF and semantic similarity
    tfidf_weight: float = 0.5
    semantic_weight: float = 0.5

    # Sentence transformer model for semantic embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_max_seq_length: int = 256

    # Field weights for content-based similarity
    field_weights: FieldWeights = field(default_factory=FieldWeights)

    # TF-IDF configuration
    tfidf_config: TfidfConfig = field(default_factory=TfidfConfig)

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "mmr_lambda": self.mmr_lambda,
            "tfidf_weight": self.tfidf_weight,
            "semantic_weight": self.semantic_weight,
            "embedding_model_name": self.embedding_model_name,
            "embedding_max_seq_length": self.embedding_max_seq_length,
            "field_weights": self.field_weights.to_dict(),
            "tfidf_config": self.tfidf_config.to_dict(),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save configuration to JSON."""
        if path is None:
            path = PROJECT_ROOT / "config" / "recommender_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RecommenderConfig":
        """Load configuration from JSON, falling back to defaults."""
        if path is None:
            path = PROJECT_ROOT / "config" / "recommender_config.json"
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        fw = FieldWeights(**d.get("field_weights", {}))
        tc = TfidfConfig(**{
            k: v for k, v in d.get("tfidf_config", {}).items()
            if k != "ngram_range"
        })
        if "ngram_range" in d.get("tfidf_config", {}):
            tc.ngram_range = tuple(d["tfidf_config"]["ngram_range"])
        return cls(
            top_k=d.get("top_k", 10),
            mmr_lambda=d.get("mmr_lambda", 0.7),
            tfidf_weight=d.get("tfidf_weight", 0.5),
            semantic_weight=d.get("semantic_weight", 0.5),
            embedding_model_name=d.get("embedding_model_name", cls.embedding_model_name),
            embedding_max_seq_length=d.get("embedding_max_seq_length", 256),
            field_weights=fw,
            tfidf_config=tc,
        )

