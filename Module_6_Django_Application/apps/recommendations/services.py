"""
Recommender Service Layer for TEDMind Django Application.
Module 6: Django Frontend & Demonstration.

Integrates the completed Modules 1-5 recommendation engine into Django.
Features:
- Thread-safe singleton lifecycle: loads precomputed models and embeddings once per process.
- Zero retraining or expensive re-computation during request-response cycles.
- Graceful error handling for missing talk IDs or edge cases.
- Exposes clean Python dictionary interfaces for views and templates.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset, parse_tags
from ted_recommender.features import FeaturePipeline
from ted_recommender.recommender import (
    ContentBasedRecommender,
    SemanticRecommender,
    HybridRecommender,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


class RecommenderService:
    """Singleton service wrapping the core ML recommendation models."""

    _instance: Optional["RecommenderService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.is_initialized = False
        self.df: Optional[pd.DataFrame] = None
        self.pipeline: Optional[FeaturePipeline] = None
        self.config: Optional[RecommenderConfig] = None
        self.tfidf_rec: Optional[ContentBasedRecommender] = None
        self.semantic_rec: Optional[SemanticRecommender] = None
        self.hybrid_rec: Optional[HybridRecommender] = None
        self.id_to_talk: Dict[int, Dict[str, Any]] = {}
        self.top_tags: List[Tuple[str, int]] = []
        self._initialize()

    @classmethod
    def get_instance(cls) -> "RecommenderService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _initialize(self):
        """Loads dataset and precomputed vectorizer/embedding artifacts once."""
        if self.is_initialized:
            return

        logger.info("Initializing TEDMind Recommender Service...")
        try:
            # 1. Load feature-engineered dataset
            self.df = load_dataset(stage="features")
            logger.info(f"Loaded dataset: {len(self.df)} TED talks.")

            # 2. Load configuration & feature pipeline
            self.config = RecommenderConfig.load()
            self.pipeline = FeaturePipeline(
                tfidf_config=self.config.tfidf_config,
                field_weights=self.config.field_weights,
            )
            loaded = self.pipeline.load_artifacts()
            if not loaded:
                logger.warning("Precomputed artifacts could not be loaded; features will need fitting.")

            # 3. Instantiate recommenders
            self.tfidf_rec = ContentBasedRecommender(self.df, self.pipeline, self.config)
            self.semantic_rec = SemanticRecommender(self.df, self.pipeline, self.config)
            self.hybrid_rec = HybridRecommender(self.df, self.pipeline, self.config)

            # 4. Precompute fast in-memory talk dictionary
            tag_counter: Dict[str, int] = {}
            for _, row in self.df.iterrows():
                tid = int(row["talk_id"])
                tags = parse_tags(row.get("tags_processed", row.get("tags", "[]")))
                for t in tags:
                    tag_counter[t] = tag_counter.get(t, 0) + 1

                views = int(row.get("views", 0))
                duration = float(row.get("duration_minutes", round(float(row.get("duration", 0)) / 60.0, 1)))

                self.id_to_talk[tid] = {
                    "talk_id": tid,
                    "title": str(row.get("title", "")),
                    "main_speaker": str(row.get("main_speaker", "")),
                    "speaker_occupation": str(row.get("speaker_occupation", "")),
                    "description": str(row.get("description", "")),
                    "tags": tags,
                    "tags_preview": tags[:4],
                    "url": str(row.get("url", "")),
                    "views": views,
                    "views_formatted": f"{views:,.0f}",
                    "duration_minutes": duration,
                    "published_year": row.get("published_year", ""),
                    "event": str(row.get("event", "TED")),
                    "has_transcript": bool(row.get("has_transcript", True)),
                }

            # Top 30 most frequent tags
            self.top_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:30]
            self.is_initialized = True
            logger.info("TEDMind Recommender Service initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize RecommenderService: {e}", exc_info=True)
            self.is_initialized = False

    def get_talk_by_id(self, talk_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve pre-formatted talk metadata by talk ID."""
        return self.id_to_talk.get(int(talk_id))

    def get_all_talk_titles(self) -> List[Dict[str, Any]]:
        """Returns minimal talk ID and title list for fast searchable dropdowns."""
        return [
            {"talk_id": t["talk_id"], "title": t["title"], "main_speaker": t["main_speaker"]}
            for t in self.id_to_talk.values()
        ]

    def get_featured_talks(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Return curated high-profile talks across diverse categories."""
        featured_ids = [1, 2, 3, 7, 26, 4]  # Ken Robinson, Al Gore, David Pogue, etc.
        talks = [self.id_to_talk[tid] for tid in featured_ids if tid in self.id_to_talk]
        if len(talks) < limit:
            # Fallback to top viewed
            talks = self.get_popular_talks(limit=limit)
        return talks[:limit]

    def get_popular_talks(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Return talks sorted by view count."""
        all_talks = list(self.id_to_talk.values())
        all_talks.sort(key=lambda x: x["views"], reverse=True)
        return all_talks[:limit]

    def search_talks(
        self,
        query: str = "",
        tag: str = "",
        speaker: str = "",
        page: int = 1,
        per_page: int = 12,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search and filter talks by keyword, tag, or speaker with pagination.
        """
        filtered = list(self.id_to_talk.values())

        if query and query.strip():
            q_clean = query.strip().lower()
            filtered = [
                t for t in filtered
                if q_clean in t["title"].lower() or q_clean in t["description"].lower()
            ]

        if speaker and speaker.strip():
            spk_clean = speaker.strip().lower()
            filtered = [t for t in filtered if spk_clean in t["main_speaker"].lower()]

        if tag and tag.strip():
            t_clean = tag.strip().lower()
            filtered = [
                t for t in filtered
                if any(t_clean == t_tag.lower() or t_clean in t_tag.lower() for t_tag in t["tags"])
            ]

        total_count = len(filtered)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_talks = filtered[start_idx:end_idx]

        return paginated_talks, total_count

    def _format_result(self, r: RecommendationResult) -> Dict[str, Any]:
        """Transform RecommendationResult into rich presentation dict."""
        t_data = self.id_to_talk.get(r.talk_id, {})
        explanation = r.explanation
        score = float(r.score)

        return {
            "talk_id": r.talk_id,
            "title": r.title,
            "main_speaker": r.main_speaker,
            "description": r.description,
            "tags": r.tags,
            "tags_preview": r.tags[:4],
            "url": r.url,
            "views": r.views,
            "views_formatted": f"{r.views:,.0f}",
            "duration_minutes": round(float(r.duration_minutes), 1),
            "score": round(score, 4),
            "match_pct": min(100, max(1, int(score * 100))) if score > 0 else 0,
            "explanation": {
                "rationale": explanation.rationale,
                "shared_tags": explanation.shared_tags,
                "speaker_match": explanation.speaker_match,
                "combined_score": round(float(explanation.combined_score), 4),
                "tfidf_score": round(float(explanation.tfidf_score), 4) if explanation.tfidf_score is not None else None,
                "semantic_score": round(float(explanation.semantic_score), 4) if explanation.semantic_score is not None else None,
            },
            "event": t_data.get("event", "TED"),
            "published_year": t_data.get("published_year", ""),
        }

    def recommend_for_talk(
        self,
        seed_talk_id: int,
        method: str = "hybrid",
        top_k: int = 10,
        apply_mmr: bool = True,
        filter_tags: Optional[List[str]] = None,
        exclude_talk_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations similar to a seed talk using the chosen model.
        """
        if not self.is_initialized or seed_talk_id not in self.id_to_talk:
            return []

        top_k = max(1, min(20, int(top_k)))
        method = method.lower().strip()

        if method == "tfidf":
            raw_recs = self.tfidf_rec.recommend_for_talk(
                talk_id=seed_talk_id,
                top_k=top_k,
                filter_tags=filter_tags,
                exclude_talk_ids=exclude_talk_ids,
                apply_mmr=apply_mmr,
            )
        elif method == "semantic":
            raw_recs = self.semantic_rec.recommend_for_talk(
                talk_id=seed_talk_id,
                top_k=top_k,
                filter_tags=filter_tags,
                exclude_talk_ids=exclude_talk_ids,
                apply_mmr=apply_mmr,
            )
        else:  # hybrid (default)
            raw_recs = self.hybrid_rec.recommend_for_talk(
                talk_id=seed_talk_id,
                top_k=top_k,
                filter_tags=filter_tags,
                exclude_talk_ids=exclude_talk_ids,
                apply_mmr=apply_mmr,
            )

        return [self._format_result(r) for r in raw_recs]

    def recommend_by_interests(
        self,
        user_interests: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        liked_talk_ids: Optional[List[int]] = None,
        disliked_talk_ids: Optional[List[int]] = None,
        method: str = "hybrid",
        top_k: int = 10,
        apply_mmr: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations using user interests, query text, or liked talks.
        """
        if not self.is_initialized:
            return []

        top_k = max(1, min(20, int(top_k)))
        method = method.lower().strip()

        # If pure free text query with TF-IDF or Semantic specified and no liked profile
        if query_text and not liked_talk_ids and not user_interests:
            if method == "tfidf":
                raw_recs = self.tfidf_rec.recommend_for_query(
                    query_text=query_text,
                    top_k=top_k,
                )
                return [self._format_result(r) for r in raw_recs]
            elif method == "semantic":
                raw_recs = self.semantic_rec.recommend_for_query(
                    query_text=query_text,
                    top_k=top_k,
                )
                return [self._format_result(r) for r in raw_recs]

        # Use Hybrid Recommender for multi-signal fusion
        w_tfidf = 0.5
        w_semantic = 0.5
        if method == "tfidf":
            w_tfidf, w_semantic = 1.0, 0.0
        elif method == "semantic":
            w_tfidf, w_semantic = 0.0, 1.0

        raw_recs = self.hybrid_rec.recommend(
            liked_talk_ids=liked_talk_ids,
            disliked_talk_ids=disliked_talk_ids,
            user_interests=user_interests,
            query_text=query_text,
            top_k=top_k,
            apply_mmr=apply_mmr,
            tfidf_weight=w_tfidf,
            semantic_weight=w_semantic,
        )

        return [self._format_result(r) for r in raw_recs]


# Global service access helper
def get_recommender_service() -> RecommenderService:
    return RecommenderService.get_instance()
