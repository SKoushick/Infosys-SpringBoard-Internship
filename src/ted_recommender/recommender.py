"""
Recommendation Engine for TED Talks.
Module 4 Implementation for Infosys Springboard Milestone 2.

Provides:
- Content-based TF-IDF Recommender
- Semantic Pretrained Embedding Recommender
- Hybrid Multi-Signal Recommender
- Session-based Personalisation (multi-liked, negative talk exclusions, user tags)
- Maximal Marginal Relevance (MMR) Ranking for Diversity
- Explainability Engine (transparent reasons for each recommended item)
- Resilient Cold-start and Edge-case Handlers
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .config import RecommenderConfig
from .data import parse_tags
from .features import FeaturePipeline

logger = logging.getLogger(__name__)


@dataclass
class RecommendationExplanation:
    """Detailed breakdown explaining why a talk was recommended."""
    talk_id: int
    title: str
    combined_score: float
    tfidf_score: Optional[float] = None
    semantic_score: Optional[float] = None
    shared_tags: List[str] = field(default_factory=list)
    speaker_match: bool = False
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "talk_id": self.talk_id,
            "title": self.title,
            "combined_score": round(float(self.combined_score), 4),
            "tfidf_score": round(float(self.tfidf_score), 4) if self.tfidf_score is not None else None,
            "semantic_score": round(float(self.semantic_score), 4) if self.semantic_score is not None else None,
            "shared_tags": self.shared_tags,
            "speaker_match": self.speaker_match,
            "rationale": self.rationale,
        }


@dataclass
class RecommendationResult:
    """Recommended talk item with metadata and explanation."""
    talk_id: int
    title: str
    main_speaker: str
    description: str
    tags: List[str]
    url: str
    views: int
    duration_minutes: float
    score: float
    explanation: RecommendationExplanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "talk_id": self.talk_id,
            "title": self.title,
            "main_speaker": self.main_speaker,
            "description": self.description,
            "tags": self.tags,
            "url": self.url,
            "views": self.views,
            "duration_minutes": round(self.duration_minutes, 2),
            "score": round(float(self.score), 4),
            "explanation": self.explanation.to_dict(),
        }


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_features: np.ndarray,
    relevance_scores: np.ndarray,
    top_k: int = 10,
    mmr_lambda: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) re-ranking for recommendation diversity.
    
    Formula:
    argmax [ lambda * Sim(query, d_i) - (1 - lambda) * max_{d_j in Selected} Sim(d_i, d_j) ]

    Args:
        query_vector: 1D normalized query representation
        candidate_indices: Row indices of candidate items in candidate_features
        candidate_features: 2D unit-normalized feature matrix (dense ndarray)
        relevance_scores: 1D array of precomputed similarity scores with query
        top_k: Number of diversified items to return
        mmr_lambda: Balance parameter [0, 1]. 1.0 = purely relevant, 0.0 = purely diverse.

    Returns:
        List of selected candidate row indices in ranked order.
    """
    if len(candidate_indices) <= top_k:
        # Return sorted by relevance if candidates fewer than requested K
        sorted_order = np.argsort(-relevance_scores)
        return [candidate_indices[i] for i in sorted_order]

    selected: List[int] = []
    selected_features: List[np.ndarray] = []
    candidates = list(range(len(candidate_indices)))

    # Select the first item with highest relevance score
    best_first = int(np.argmax(relevance_scores))
    selected.append(candidate_indices[best_first])
    selected_features.append(candidate_features[candidate_indices[best_first]])
    candidates.remove(best_first)

    while len(selected) < top_k and candidates:
        sub_features = candidate_features[candidate_indices[candidates]]  # (N_cand, Dim)
        rel_scores = relevance_scores[candidates]

        # Compute maximum similarity between each candidate and already selected items
        sel_matrix = np.array(selected_features)  # (N_sel, Dim)
        # Cosine similarity is simple dot product as vectors are unit normalized
        inter_sims = np.dot(sub_features, sel_matrix.T)  # (N_cand, N_sel)
        max_inter_sims = np.max(inter_sims, axis=1)

        # Compute MMR score for remaining candidates
        mmr_scores = mmr_lambda * rel_scores - (1.0 - mmr_lambda) * max_inter_sims
        best_cand_idx = int(np.argmax(mmr_scores))

        selected_actual_idx = candidate_indices[candidates[best_cand_idx]]
        selected.append(selected_actual_idx)
        selected_features.append(candidate_features[selected_actual_idx])
        candidates.pop(best_cand_idx)

    return selected


class BaseRecommender:
    """Base class for all TED Talk recommenders."""

    def __init__(self, df: pd.DataFrame, config: Optional[RecommenderConfig] = None):
        self.df = df.reset_index(drop=True)
        self.config = config or RecommenderConfig()

        # Fast lookup indices
        self.id_to_idx: Dict[int, int] = {
            int(row["talk_id"]): idx for idx, row in self.df.iterrows()
        }
        self.idx_to_id: Dict[int, int] = {
            idx: int(row["talk_id"]) for idx, row in self.df.iterrows()
        }

    def _get_row(self, talk_id: int) -> pd.Series:
        if talk_id not in self.id_to_idx:
            raise KeyError(f"Talk ID {talk_id} not found in dataset.")
        return self.df.iloc[self.id_to_idx[talk_id]]

    def _build_result_list(
        self,
        ranked_indices: List[int],
        scores: Dict[int, float],
        explanations: Dict[int, RecommendationExplanation],
    ) -> List[RecommendationResult]:
        results = []
        for idx in ranked_indices:
            row = self.df.iloc[idx]
            t_id = int(row["talk_id"])
            raw_tags = row.get("tags_processed", row.get("tags", "[]"))
            parsed_tag_list = parse_tags(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
            duration_val = float(row.get("duration_minutes", row.get("duration", 0) / 60))

            results.append(
                RecommendationResult(
                    talk_id=t_id,
                    title=str(row["title"]),
                    main_speaker=str(row.get("main_speaker", row.get("speaker", "Unknown"))),
                    description=str(row.get("description", "")),
                    tags=parsed_tag_list,
                    url=str(row.get("url", "")),
                    views=int(row.get("views", 0)),
                    duration_minutes=duration_val,
                    score=scores.get(idx, 0.0),
                    explanation=explanations[t_id],
                )
            )
        return results


class ContentBasedRecommender(BaseRecommender):
    """
    TF-IDF content-based recommender using cosine similarity over weighted text features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline: FeaturePipeline,
        config: Optional[RecommenderConfig] = None,
    ):
        super().__init__(df, config)
        self.pipeline = pipeline
        if self.pipeline.combined_tfidf_matrix is None:
            raise ValueError("FeaturePipeline must have a fitted combined_tfidf_matrix.")
        self.tfidf_matrix = self.pipeline.combined_tfidf_matrix

    def recommend_for_talk(
        self,
        talk_id: int,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        exclude_talk_ids: Optional[Set[int]] = None,
        apply_mmr: bool = False,
    ) -> List[RecommendationResult]:
        """Generate recommendations for a given seed talk."""
        k = top_k or self.config.top_k
        if talk_id not in self.id_to_idx:
            raise KeyError(f"Seed talk_id {talk_id} not found in dataset.")

        seed_idx = self.id_to_idx[talk_id]
        seed_row = self.df.iloc[seed_idx]
        seed_vec = self.tfidf_matrix.getrow(seed_idx)

        # Compute cosine similarity (dot product of L2 normalized vectors)
        # Sparse dot product: (1, Dim) x (N, Dim).T -> (1, N)
        sim_scores = (self.tfidf_matrix.dot(seed_vec.T)).toarray().ravel()

        # Exclusions: exclude self and any specified talks
        exclude = set(exclude_talk_ids or set())
        exclude.add(talk_id)

        # Tag filters
        required_tags = set(t.lower().strip() for t in (filter_tags or []))

        candidate_indices = []
        candidate_scores = []
        seed_tags = set(parse_tags(seed_row.get("tags_processed", seed_row.get("tags", "[]"))))

        for idx, score in enumerate(sim_scores):
            t_id = self.idx_to_id[idx]
            if t_id in exclude:
                continue

            if required_tags:
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                if not required_tags.intersection(row_tags):
                    continue

            candidate_indices.append(idx)
            candidate_scores.append(float(score))

        if not candidate_indices:
            return []

        cand_indices_arr = np.array(candidate_indices)
        cand_scores_arr = np.array(candidate_scores)

        if apply_mmr and len(cand_indices_arr) > k:
            # Dense slice of candidate vectors for MMR
            cand_features_dense = self.tfidf_matrix[cand_indices_arr].toarray()
            seed_vec_dense = seed_vec.toarray().ravel()
            ranked_indices = maximal_marginal_relevance(
                query_vector=seed_vec_dense,
                candidate_indices=cand_indices_arr,
                candidate_features=self.tfidf_matrix.toarray(),
                relevance_scores=cand_scores_arr,
                top_k=k,
                mmr_lambda=self.config.mmr_lambda,
            )
        else:
            top_ranked_pos = np.argsort(-cand_scores_arr)[:k]
            ranked_indices = cand_indices_arr[top_ranked_pos].tolist()

        score_map = {idx: float(sim_scores[idx]) for idx in ranked_indices}
        explanations = {}
        for idx in ranked_indices:
            t_id = self.idx_to_id[idx]
            row = self.df.iloc[idx]
            target_tags = set(parse_tags(row.get("tags_processed", row.get("tags", "[]"))))
            shared = sorted(list(seed_tags.intersection(target_tags)))
            spk_match = str(seed_row.get("main_speaker", "")) == str(row.get("main_speaker", ""))
            
            reasons = []
            if shared:
                reasons.append(f"Shared themes: {', '.join(shared[:4])}")
            if spk_match:
                reasons.append(f"Same speaker ({row.get('main_speaker')})")
            reasons.append(f"Lexical content similarity: {score_map[idx]:.3f}")

            explanations[t_id] = RecommendationExplanation(
                talk_id=t_id,
                title=str(row["title"]),
                combined_score=score_map[idx],
                tfidf_score=score_map[idx],
                shared_tags=shared,
                speaker_match=spk_match,
                rationale="; ".join(reasons),
            )

        return self._build_result_list(ranked_indices, score_map, explanations)

    def recommend_for_query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
    ) -> List[RecommendationResult]:
        """Recommend talks matching raw user search text."""
        k = top_k or self.config.top_k
        query_clean = query_text.strip()
        if not query_clean:
            return []

        q_vec = self.pipeline.transform_query_tfidf(query_clean)
        sim_scores = (self.tfidf_matrix.dot(q_vec.T)).toarray().ravel()

        required_tags = set(t.lower().strip() for t in (filter_tags or []))
        candidate_indices = []
        candidate_scores = []

        for idx, score in enumerate(sim_scores):
            if required_tags:
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                if not required_tags.intersection(row_tags):
                    continue
            candidate_indices.append(idx)
            candidate_scores.append(float(score))

        if not candidate_indices:
            return []

        cand_indices_arr = np.array(candidate_indices)
        cand_scores_arr = np.array(candidate_scores)
        top_pos = np.argsort(-cand_scores_arr)[:k]
        ranked_indices = cand_indices_arr[top_pos].tolist()

        score_map = {idx: float(sim_scores[idx]) for idx in ranked_indices}
        explanations = {
            self.idx_to_id[idx]: RecommendationExplanation(
                talk_id=self.idx_to_id[idx],
                title=str(self.df.iloc[idx]["title"]),
                combined_score=score_map[idx],
                tfidf_score=score_map[idx],
                rationale=f"Lexical match with query: {score_map[idx]:.3f}",
            )
            for idx in ranked_indices
        }
        return self._build_result_list(ranked_indices, score_map, explanations)


class SemanticRecommender(BaseRecommender):
    """
    Semantic recommender using pretrained SentenceTransformer embeddings.
    Captures conceptual and contextual similarity even when vocabulary diverges.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline: FeaturePipeline,
        config: Optional[RecommenderConfig] = None,
    ):
        super().__init__(df, config)
        self.pipeline = pipeline
        if self.pipeline.embedding_matrix is None:
            raise ValueError("FeaturePipeline must have a generated embedding_matrix.")
        self.embedding_matrix = self.pipeline.embedding_matrix

    def recommend_for_talk(
        self,
        talk_id: int,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        exclude_talk_ids: Optional[Set[int]] = None,
        apply_mmr: bool = False,
    ) -> List[RecommendationResult]:
        k = top_k or self.config.top_k
        if talk_id not in self.id_to_idx:
            raise KeyError(f"Seed talk_id {talk_id} not found in dataset.")

        seed_idx = self.id_to_idx[talk_id]
        seed_row = self.df.iloc[seed_idx]
        seed_vec = self.embedding_matrix[seed_idx]

        # Unit-normalized embeddings dot product equals cosine similarity
        sim_scores = np.dot(self.embedding_matrix, seed_vec)

        exclude = set(exclude_talk_ids or set())
        exclude.add(talk_id)
        required_tags = set(t.lower().strip() for t in (filter_tags or []))

        candidate_indices = []
        candidate_scores = []
        seed_tags = set(parse_tags(seed_row.get("tags_processed", seed_row.get("tags", "[]"))))

        for idx, score in enumerate(sim_scores):
            t_id = self.idx_to_id[idx]
            if t_id in exclude:
                continue
            if required_tags:
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                if not required_tags.intersection(row_tags):
                    continue
            candidate_indices.append(idx)
            candidate_scores.append(float(score))

        if not candidate_indices:
            return []

        cand_indices_arr = np.array(candidate_indices)
        cand_scores_arr = np.array(candidate_scores)

        if apply_mmr and len(cand_indices_arr) > k:
            ranked_indices = maximal_marginal_relevance(
                query_vector=seed_vec,
                candidate_indices=cand_indices_arr,
                candidate_features=self.embedding_matrix,
                relevance_scores=cand_scores_arr,
                top_k=k,
                mmr_lambda=self.config.mmr_lambda,
            )
        else:
            top_ranked_pos = np.argsort(-cand_scores_arr)[:k]
            ranked_indices = cand_indices_arr[top_ranked_pos].tolist()

        score_map = {idx: float(sim_scores[idx]) for idx in ranked_indices}
        explanations = {}
        for idx in ranked_indices:
            t_id = self.idx_to_id[idx]
            row = self.df.iloc[idx]
            target_tags = set(parse_tags(row.get("tags_processed", row.get("tags", "[]"))))
            shared = sorted(list(seed_tags.intersection(target_tags)))
            spk_match = str(seed_row.get("main_speaker", "")) == str(row.get("main_speaker", ""))

            reasons = [f"Semantic embedding similarity: {score_map[idx]:.3f}"]
            if shared:
                reasons.append(f"Shared themes: {', '.join(shared[:4])}")

            explanations[t_id] = RecommendationExplanation(
                talk_id=t_id,
                title=str(row["title"]),
                combined_score=score_map[idx],
                semantic_score=score_map[idx],
                shared_tags=shared,
                speaker_match=spk_match,
                rationale="; ".join(reasons),
            )

        return self._build_result_list(ranked_indices, score_map, explanations)

    def recommend_for_query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
    ) -> List[RecommendationResult]:
        k = top_k or self.config.top_k
        query_clean = query_text.strip()
        if not query_clean:
            return []

        q_vec = self.pipeline.transform_query_embedding(query_clean)
        sim_scores = np.dot(self.embedding_matrix, q_vec)

        required_tags = set(t.lower().strip() for t in (filter_tags or []))
        candidate_indices = []
        candidate_scores = []

        for idx, score in enumerate(sim_scores):
            if required_tags:
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                if not required_tags.intersection(row_tags):
                    continue
            candidate_indices.append(idx)
            candidate_scores.append(float(score))

        if not candidate_indices:
            return []

        cand_indices_arr = np.array(candidate_indices)
        cand_scores_arr = np.array(candidate_scores)
        top_pos = np.argsort(-cand_scores_arr)[:k]
        ranked_indices = cand_indices_arr[top_pos].tolist()

        score_map = {idx: float(sim_scores[idx]) for idx in ranked_indices}
        explanations = {
            self.idx_to_id[idx]: RecommendationExplanation(
                talk_id=self.idx_to_id[idx],
                title=str(self.df.iloc[idx]["title"]),
                combined_score=score_map[idx],
                semantic_score=score_map[idx],
                rationale=f"Semantic conceptual match: {score_map[idx]:.3f}",
            )
            for idx in ranked_indices
        }
        return self._build_result_list(ranked_indices, score_map, explanations)


class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommender fusing:
    1. Lexical TF-IDF similarity (exact term, keyword, and tag matches)
    2. Semantic Sentence-BERT embedding similarity (latent intent and conceptual overlap)
    3. Editorial metadata weighting (shared tags, speaker affinity)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline: FeaturePipeline,
        config: Optional[RecommenderConfig] = None,
    ):
        super().__init__(df, config)
        self.pipeline = pipeline
        self.tfidf_rec = ContentBasedRecommender(df, pipeline, config)
        self.has_semantic = pipeline.embedding_matrix is not None
        if self.has_semantic:
            self.semantic_rec = SemanticRecommender(df, pipeline, config)
        else:
            self.semantic_rec = None
            logger.warning("Semantic embeddings not available; running in TF-IDF fallback mode.")

    def recommend(
        self,
        seed_talk_id: Optional[int] = None,
        liked_talk_ids: Optional[List[int]] = None,
        disliked_talk_ids: Optional[List[int]] = None,
        user_interests: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_tags: Optional[List[str]] = None,
        apply_mmr: bool = True,
        tfidf_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """
        Produce personalized, diversified hybrid recommendations.

        Supports:
        - Single seed talk
        - Session profile from multiple liked talks
        - Negative feedback / exclusions (disliked talks)
        - Topical interests / tags filtering and boosting
        - Raw text search query
        - Configurable hybrid scoring and MMR diversification
        """
        k = top_k or self.config.top_k
        w_tfidf = tfidf_weight if tfidf_weight is not None else self.config.tfidf_weight
        w_semantic = semantic_weight if semantic_weight is not None else self.config.semantic_weight

        # Normalize weights
        total_w = w_tfidf + (w_semantic if self.has_semantic else 0.0)
        if total_w > 0:
            w_tfidf /= total_w
            w_semantic /= total_w
        else:
            w_tfidf = 1.0
            w_semantic = 0.0

        # Construct aggregate user preference profile
        positive_ids: List[int] = []
        if seed_talk_id is not None:
            positive_ids.append(seed_talk_id)
        if liked_talk_ids:
            positive_ids.extend(liked_talk_ids)

        exclude_ids: Set[int] = set(disliked_talk_ids or [])
        exclude_ids.update(positive_ids)  # Do not recommend seed / already liked talks

        # Cold-start handling: If no input given, fallback to highest viewed talks
        if not positive_ids and not user_interests and not query_text:
            return self._popular_fallback(top_k=k, exclude_ids=exclude_ids)

        n_items = len(self.df)
        tfidf_scores = np.zeros(n_items, dtype=np.float32)
        semantic_scores = np.zeros(n_items, dtype=np.float32)

        # 1. Compute similarity from positive talk profile
        if positive_ids:
            valid_pos_indices = [
                self.id_to_idx[t_id] for t_id in positive_ids if t_id in self.id_to_idx
            ]
            if valid_pos_indices:
                # Average TF-IDF profile vector
                pos_tfidf_vecs = self.pipeline.combined_tfidf_matrix[valid_pos_indices]
                avg_tfidf_vec = sparse.csr_matrix(pos_tfidf_vecs.mean(axis=0))
                avg_tfidf_vec = normalize(avg_tfidf_vec, norm="l2")
                tfidf_scores += (self.pipeline.combined_tfidf_matrix.dot(avg_tfidf_vec.T)).toarray().ravel()

                # Average Semantic embedding vector
                if self.has_semantic:
                    pos_emb_vecs = self.pipeline.embedding_matrix[valid_pos_indices]
                    avg_emb_vec = pos_emb_vecs.mean(axis=0)
                    norm = np.linalg.norm(avg_emb_vec)
                    if norm > 0:
                        avg_emb_vec /= norm
                    semantic_scores += np.dot(self.pipeline.embedding_matrix, avg_emb_vec)

        # 2. Add text query signal if provided
        if query_text and query_text.strip():
            q_tfidf = self.pipeline.transform_query_tfidf(query_text.strip())
            tfidf_scores += 0.5 * (self.pipeline.combined_tfidf_matrix.dot(q_tfidf.T)).toarray().ravel()

            if self.has_semantic:
                q_emb = self.pipeline.transform_query_embedding(query_text.strip())
                semantic_scores += 0.5 * np.dot(self.pipeline.embedding_matrix, q_emb)

        # 3. Add topical interest boost
        interest_boost = np.zeros(n_items, dtype=np.float32)
        interest_tags = set(t.lower().strip() for t in (user_interests or []))
        if interest_tags:
            for idx in range(n_items):
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                match_count = len(interest_tags.intersection(row_tags))
                if match_count > 0:
                    interest_boost[idx] = min(0.3, match_count * 0.1)

        # Combine scores
        if self.has_semantic:
            combined_scores = (w_tfidf * tfidf_scores) + (w_semantic * semantic_scores) + interest_boost
        else:
            combined_scores = tfidf_scores + interest_boost

        # Tag filters
        required_tags = set(t.lower().strip() for t in (filter_tags or []))
        candidate_indices = []
        candidate_scores = []

        for idx in range(n_items):
            t_id = self.idx_to_id[idx]
            if t_id in exclude_ids:
                continue
            if required_tags:
                row_tags = set(parse_tags(self.df.iloc[idx].get("tags_processed", self.df.iloc[idx].get("tags", "[]"))))
                if not required_tags.intersection(row_tags):
                    continue
            candidate_indices.append(idx)
            candidate_scores.append(float(combined_scores[idx]))

        if not candidate_indices:
            return []

        cand_indices_arr = np.array(candidate_indices)
        cand_scores_arr = np.array(candidate_scores)

        # Ranking: MMR diversification or direct top-K
        if apply_mmr and len(cand_indices_arr) > k and self.has_semantic:
            # Use semantic embeddings as feature space for MMR diversity
            ranked_indices = maximal_marginal_relevance(
                query_vector=self.pipeline.embedding_matrix[valid_pos_indices[0]] if positive_ids else self.pipeline.embedding_matrix[0],
                candidate_indices=cand_indices_arr,
                candidate_features=self.pipeline.embedding_matrix,
                relevance_scores=cand_scores_arr,
                top_k=k,
                mmr_lambda=self.config.mmr_lambda,
            )
        else:
            top_pos = np.argsort(-cand_scores_arr)[:k]
            ranked_indices = cand_indices_arr[top_pos].tolist()

        score_map = {idx: float(combined_scores[idx]) for idx in ranked_indices}

        # Build explanations
        seed_tags = set()
        for p_id in positive_ids:
            if p_id in self.id_to_idx:
                r = self.df.iloc[self.id_to_idx[p_id]]
                seed_tags.update(parse_tags(r.get("tags_processed", r.get("tags", "[]"))))

        explanations = {}
        for idx in ranked_indices:
            t_id = self.idx_to_id[idx]
            row = self.df.iloc[idx]
            target_tags = set(parse_tags(row.get("tags_processed", row.get("tags", "[]"))))
            shared = sorted(list((seed_tags | interest_tags).intersection(target_tags)))

            reasons = []
            if shared:
                reasons.append(f"Matching topics: {', '.join(shared[:4])}")
            if self.has_semantic:
                reasons.append(f"Hybrid Score: {score_map[idx]:.3f} (TF-IDF: {tfidf_scores[idx]:.2f}, Semantic: {semantic_scores[idx]:.2f})")
            else:
                reasons.append(f"TF-IDF Score: {tfidf_scores[idx]:.3f}")

            explanations[t_id] = RecommendationExplanation(
                talk_id=t_id,
                title=str(row["title"]),
                combined_score=score_map[idx],
                tfidf_score=float(tfidf_scores[idx]),
                semantic_score=float(semantic_scores[idx]) if self.has_semantic else None,
                shared_tags=shared,
                rationale="; ".join(reasons),
            )

        return self._build_result_list(ranked_indices, score_map, explanations)

    def _popular_fallback(self, top_k: int, exclude_ids: Set[int]) -> List[RecommendationResult]:
        """Cold-start fallback returning top viewed talks."""
        candidates = []
        for idx, row in self.df.iterrows():
            t_id = int(row["talk_id"])
            if t_id in exclude_ids:
                continue
            candidates.append((idx, int(row.get("views", 0))))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]
        ranked_indices = [idx for idx, _ in top_candidates]

        score_map = {idx: 1.0 - (rank / len(top_candidates)) for rank, (idx, _) in enumerate(top_candidates)}
        explanations = {
            self.idx_to_id[idx]: RecommendationExplanation(
                talk_id=self.idx_to_id[idx],
                title=str(self.df.iloc[idx]["title"]),
                combined_score=score_map[idx],
                rationale="Cold-start fallback: Top trending TED talk by view count.",
            )
            for idx in ranked_indices
        }
        return self._build_result_list(ranked_indices, score_map, explanations)
