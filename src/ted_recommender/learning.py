"""
Pedagogical Complexity and Difficulty Estimation Module.
TEDMind Recommendation & Learning Platform | Infosys Springboard 7.0

Provides:
- PedagogicalComplexityEstimator: Modular heuristic estimator computing talk difficulty
  and mapping presentations into progressive learning stages:
  * Foundational (Beginner)
  * Intermediate (Intermediate)
  * Specialized (Advanced)
- ComplexityProfile: Structured dataclass exposing transparent component scores,
  raw linguistic metrics, evidence-quality confidence, and human-readable explanations.

Scientific & Methodological Notice:
-----------------------------------
This module implements a HEURISTIC ESTIMATE based on observable text and audio-duration
proxies (lexical sophistication, speaking pace, topic domain, and duration load).
It is NOT a direct psychometric or cognitive measurement of human learner comprehension.
Assumptions and limitations:
1. Mean Word Length is used as an indicator of academic and technical vocabulary;
   however, short words can express profound concepts (e.g., philosophical logic).
2. Lexical Diversity (MATTR) measures vocabulary richness within fixed text windows to
   avoid raw Type-Token Ratio deflation on long texts.
3. Words-Per-Minute (WPM) is a PACING PROXY, not a direct measurement of cognitive load.
4. Stage thresholds (0.40 and 0.70) are provisional heuristic cutoffs for empirical analysis.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Curated Domain Specialization Lexicon ─────────────────────────────────────────
HIGH_SPECIALIZATION_TAGS: Set[str] = {
    "neuroscience", "physics", "quantum", "genetics", "biotechnology",
    "nanotechnology", "microbiology", "astrophysics", "synthetic biology",
    "machine learning", "cryptography", "mathematics", "macroeconomics",
    "artificial intelligence", "algorithms", "biomimicry", "molecular biology",
}

MODERATE_SPECIALIZATION_TAGS: Set[str] = {
    "technology", "science", "medicine", "engineering", "philosophy",
    "psychology", "business", "economics", "data", "ecology", "environment",
    "cognitive science", "sociology", "foreign policy", "global issues",
    "biology", "brain", "evolution", "medical research", "chemistry",
}

ACCESSIBLE_INTRODUCTORY_TAGS: Set[str] = {
    "storytelling", "humor", "entertainment", "music", "culture",
    "children", "parenting", "comedy", "art", "performance",
    "personal growth", "motivation", "happiness", "creativity",
}


@dataclass
class ComplexityEstimatorConfig:
    """
    Configurable parameters and weights for PedagogicalComplexityEstimator.
    All weights are normalized such that they sum to 1.0.
    """
    # Feature weights (Heuristic baseline)
    duration_weight: float = 0.20       # Limited supporting feature (not primary determinant)
    lexical_weight: float = 0.40        # Vocabulary depth + length-aware diversity
    pacing_weight: float = 0.20         # Speaking pace / information delivery rate proxy
    domain_weight: float = 0.20         # Domain specialization bias

    # Provisional stage boundaries
    foundational_threshold: float = 0.40
    specialized_threshold: float = 0.70

    # Normalization bounds
    min_duration_minutes: float = 3.0
    max_duration_minutes: float = 30.0   # Caps duration impact beyond 30 min
    expected_wpm_min: float = 100.0      # Lower reference speaking rate
    expected_wpm_max: float = 190.0      # Upper reference speaking rate
    min_avg_word_length: float = 4.0     # Conversational English baseline
    max_avg_word_length: float = 6.2     # Technical/academic English baseline
    mattr_window_size: int = 200         # Window size for Moving-Average TTR
    min_tokens_for_ttr: int = 100        # Minimum tokens required for reliable lexical diversity
    corpus_mean_word_length: float = 4.75 # Empirical mean word length in TED Talks corpus

    def __post_init__(self):
        total_w = self.duration_weight + self.lexical_weight + self.pacing_weight + self.domain_weight
        if not math.isclose(total_w, 1.0, rel_tol=1e-3):
            raise ValueError(f"Feature weights must sum to 1.0; got {total_w:.4f}")
        if not (0.0 < self.foundational_threshold < self.specialized_threshold < 1.0):
            raise ValueError("Stage thresholds must satisfy: 0 < foundational < specialized < 1")


@dataclass
class ComplexityProfile:
    """
    Full diagnostic evaluation of a talk's estimated pedagogical complexity.
    """
    talk_id: int
    complexity_score: float             # Bounded in [0.0, 1.0]
    learning_stage: str                 # 'foundational', 'intermediate', or 'specialized'
    feature_scores: Dict[str, float]    # Component scores: duration, lexical, pacing, domain
    raw_metrics: Dict[str, Any]         # Unnormalized observable metrics
    source: str                         # 'transcript' or 'metadata_fallback'
    is_fallback: bool
    evidence_quality: str               # 'high', 'medium', or 'low'
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "talk_id": self.talk_id,
            "complexity_score": round(self.complexity_score, 4),
            "learning_stage": self.learning_stage,
            "feature_scores": {k: round(v, 4) for k, v in self.feature_scores.items()},
            "raw_metrics": self.raw_metrics,
            "source": self.source,
            "is_fallback": self.is_fallback,
            "evidence_quality": self.evidence_quality,
            "explanation": self.explanation,
        }


class PedagogicalComplexityEstimator:
    """
    Modular, explainable estimator for evaluating TED Talk pedagogical complexity.
    Operates without side-effects on existing recommendation models or databases.
    """

    def __init__(self, config: Optional[ComplexityEstimatorConfig] = None):
        self.config = config or ComplexityEstimatorConfig()

    @staticmethod
    def _clean_tokens(text: str) -> List[str]:
        """Extract lowercase alphanumeric word tokens of length >= 2."""
        if not text or not isinstance(text, str):
            return []
        # Remove punctuation, split into alphanumeric tokens
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return tokens

    def _compute_duration_score(self, duration_minutes: float) -> Tuple[float, float]:
        """
        Duration is a supporting feature: short talks (<8m) have lower load;
        standard talks (12-18m) are medium; long lectures (>30m) saturate near 1.0.
        """
        safe_duration = max(0.5, float(duration_minutes))
        clamped = min(self.config.max_duration_minutes, max(self.config.min_duration_minutes, safe_duration))
        # Linear normalization within bounds
        span = self.config.max_duration_minutes - self.config.min_duration_minutes
        norm = (clamped - self.config.min_duration_minutes) / span
        return float(np.clip(norm, 0.0, 1.0)), safe_duration

    def _compute_lexical_metrics(
        self,
        tokens: Sequence[str],
        is_fallback: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute mean word length and Moving-Average Type-Token Ratio (MATTR).
        Controls for transcript length and protects against artificial score inflation
        on short fallback descriptions.
        """
        n_tokens = len(tokens)
        if n_tokens == 0:
            default_score = 0.50 if is_fallback else 0.0
            return default_score, {
                "avg_word_length": 0.0,
                "mattr": 0.50,
                "unique_words": 0,
                "ttr_reliable": False,
            }

        # 1. Mean word length with Bayesian shrinkage for short texts (< 50 tokens)
        char_counts = [len(w) for w in tokens]
        raw_avg_len = sum(char_counts) / n_tokens

        # If text is very short, shrink observed average towards corpus baseline (4.75)
        # to prevent a single long word (e.g. 'primatologist') from skewing the score.
        shrinkage = min(1.0, n_tokens / 50.0)
        smoothed_avg_len = shrinkage * raw_avg_len + (1.0 - shrinkage) * self.config.corpus_mean_word_length

        len_span = self.config.max_avg_word_length - self.config.min_avg_word_length
        len_norm = np.clip((smoothed_avg_len - self.config.min_avg_word_length) / len_span, 0.0, 1.0)

        # 2. Length-aware MATTR with strict minimum token requirement
        min_ttr_tokens = self.config.min_tokens_for_ttr
        ttr_reliable = n_tokens >= min_ttr_tokens

        if n_tokens >= self.config.mattr_window_size:
            # Average TTR across non-overlapping windows of 200 tokens
            w_size = self.config.mattr_window_size
            window_ttrs = []
            for i in range(0, n_tokens - w_size + 1, w_size):
                window = tokens[i:i + w_size]
                window_ttrs.append(len(set(window)) / w_size)
            mattr = sum(window_ttrs) / len(window_ttrs) if window_ttrs else 0.55
            ttr_norm = np.clip((mattr - 0.45) / 0.40, 0.0, 1.0)
        elif ttr_reliable:
            # Between 100 and 199 tokens: length-controlled root TTR
            raw_mattr = len(set(tokens)) / math.sqrt(2 * n_tokens)
            mattr = min(1.0, raw_mattr)
            ttr_norm = np.clip((mattr - 0.45) / 0.40, 0.0, 1.0)
        else:
            # Insufficient text (< 100 tokens, e.g. metadata description snippets):
            # Type-Token Ratio cannot be reliably computed because repetitions rarely occur in short text.
            # Do NOT extrapolate high TTR. Impute conservative population median (0.50).
            mattr = 0.55  # Population median reference
            ttr_norm = 0.50

        # 3. Blended lexical score
        if is_fallback or not ttr_reliable:
            # Conservative fallback: weight towards neutral baseline (0.50)
            lexical_score = float(0.50 * len_norm + 0.50 * 0.50)
        else:
            lexical_score = float(0.60 * len_norm + 0.40 * ttr_norm)

        raw = {
            "avg_word_length": round(float(raw_avg_len), 2),
            "smoothed_avg_length": round(float(smoothed_avg_len), 2),
            "mattr": round(float(mattr), 4),
            "unique_words": len(set(tokens)),
            "ttr_reliable": ttr_reliable,
        }
        return float(lexical_score), raw

    def _compute_pacing_score(self, n_words: int, duration_minutes: float) -> Tuple[float, float]:
        """
        Words-per-minute (WPM) as an information delivery and speech pace proxy.
        """
        if duration_minutes <= 0.0 or n_words <= 0:
            return 0.5, 0.0  # Median default

        wpm = n_words / duration_minutes
        span = self.config.expected_wpm_max - self.config.expected_wpm_min
        pacing_norm = np.clip((wpm - self.config.expected_wpm_min) / span, 0.0, 1.0)
        return float(pacing_norm), round(float(wpm), 1)

    @staticmethod
    def _compute_domain_score(tags: Union[str, Sequence[str]]) -> Tuple[float, List[str]]:
        """
        Domain specialization score based on curated taxonomy tags.
        """
        if isinstance(tags, str):
            from .data import parse_tags
            cleaned_tags = parse_tags(tags)
        elif isinstance(tags, (list, tuple, set)):
            from .data import parse_tags
            cleaned_tags = []
            for t in tags:
                if isinstance(t, str):
                    if t.startswith("[") and t.endswith("]"):
                        cleaned_tags.extend(parse_tags(t))
                    else:
                        cleaned_tags.append(t.strip().lower())
        else:
            cleaned_tags = []
        matched_high = [t for t in cleaned_tags if t in HIGH_SPECIALIZATION_TAGS]
        matched_mod = [t for t in cleaned_tags if t in MODERATE_SPECIALIZATION_TAGS]
        matched_acc = [t for t in cleaned_tags if t in ACCESSIBLE_INTRODUCTORY_TAGS]

        # Base neutral score is 0.50
        domain_val = 0.50
        if matched_high:
            domain_val += 0.15 * min(3, len(matched_high))
        elif matched_mod:
            domain_val += 0.08 * min(3, len(matched_mod))

        if matched_acc and not matched_high:
            domain_val -= 0.10 * min(2, len(matched_acc))

        domain_score = float(np.clip(domain_val, 0.10, 0.95))
        specialized_tags = matched_high if matched_high else matched_mod
        return domain_score, specialized_tags

    def estimate_complexity(
        self,
        talk_id: int,
        duration_minutes: float,
        transcript: Optional[str] = None,
        description: str = "",
        tags: Optional[Sequence[str]] = None,
    ) -> ComplexityProfile:
        """
        Estimates pedagogical complexity for a talk from transcript and metadata.
        Resilient against missing transcripts, short talks, and empty inputs.
        """
        tags = tags or []
        tokens = self._clean_tokens(transcript) if transcript else []

        # Determine evidence quality and fallback status
        if tokens and len(tokens) >= 50:
            source = "transcript"
            is_fallback = False
            evidence_quality = "high" if len(tokens) >= 200 else "medium"
            text_tokens = tokens
        else:
            # Fallback path: use description and tags
            source = "metadata_fallback"
            is_fallback = True
            evidence_quality = "low"
            fallback_text = f"{description} {' '.join(tags)}"
            text_tokens = self._clean_tokens(fallback_text)

        # 1. Duration score
        duration_score, raw_duration = self._compute_duration_score(duration_minutes)

        # 2. Lexical score (with short-text shrinkage and fallback protection)
        lexical_score, raw_lexical = self._compute_lexical_metrics(text_tokens, is_fallback=is_fallback)

        # 3. Pacing score
        if not is_fallback:
            pacing_score, raw_wpm = self._compute_pacing_score(len(tokens), raw_duration)
        else:
            pacing_score, raw_wpm = 0.50, 0.0  # Safe median default for fallback

        # 4. Domain specialization score
        domain_score, specialized_tags = self._compute_domain_score(tags)

        # 5. Composite weighted complexity score
        cfg = self.config
        composite = (
            cfg.duration_weight * duration_score +
            cfg.lexical_weight * lexical_score +
            cfg.pacing_weight * pacing_score +
            cfg.domain_weight * domain_score
        )
        final_score = float(np.clip(composite, 0.0, 1.0))

        # 6. Learning stage mapping with conservative fallback capping
        # When verbatim transcript speech is absent (is_fallback=True), evidence is insufficient
        # to justify an unverified 'specialized' stage. It is conservatively capped at 'intermediate'.
        if final_score < cfg.foundational_threshold:
            learning_stage = "foundational"
            stage_desc = "Foundational (Beginner: accessible themes and intuitive pacing)"
        elif final_score < cfg.specialized_threshold or is_fallback:
            learning_stage = "intermediate"
            if is_fallback and final_score >= cfg.specialized_threshold:
                stage_desc = "Intermediate [Provisional] (Capped: metadata indicates technical topic but transcript is missing)"
            else:
                stage_desc = "Intermediate (In-depth discussion with standard technical vocabulary)"
        else:
            learning_stage = "specialized"
            stage_desc = "Specialized (Advanced: dense domain terminology, deep conceptual complexity)"

        # 7. Generate transparent explanation
        factors = []
        if lexical_score >= 0.65 and not is_fallback:
            factors.append(f"high lexical sophistication (avg word len {raw_lexical.get('avg_word_length', 0)} chars)")
        elif lexical_score <= 0.35:
            factors.append("accessible, conversational vocabulary")

        if specialized_tags:
            factors.append(f"specialized domain themes ({', '.join(specialized_tags[:2])})")

        if duration_score >= 0.70:
            factors.append(f"extended lecture duration ({raw_duration:.1f}m)")
        elif duration_score <= 0.30:
            factors.append(f"concise duration ({raw_duration:.1f}m)")

        if is_fallback:
            factors.append("estimated from metadata fallback (transcript unavailable; lexical diversity unmeasurable on short text)")

        factor_str = "; ".join(factors) if factors else "balanced across duration, pacing, and vocabulary"
        explanation = f"Assigned to {stage_desc}. Key drivers: {factor_str}."

        feature_scores = {
            "duration_score": duration_score,
            "lexical_score": lexical_score,
            "pacing_score": pacing_score,
            "domain_score": domain_score,
        }

        raw_metrics = {
            "duration_minutes": raw_duration,
            "word_count": len(tokens) if not is_fallback else len(text_tokens),
            "words_per_minute": raw_wpm,
            "specialized_tags": specialized_tags,
            **raw_lexical,
        }

        return ComplexityProfile(
            talk_id=int(talk_id),
            complexity_score=final_score,
            learning_stage=learning_stage,
            feature_scores=feature_scores,
            raw_metrics=raw_metrics,
            source=source,
            is_fallback=is_fallback,
            evidence_quality=evidence_quality,
            explanation=explanation,
        )

    def estimate_dataframe(
        self,
        df: pd.DataFrame,
        talk_id_col: str = "talk_id",
        duration_col: str = "duration_minutes",
        transcript_col: str = "transcript_clean",
        description_col: str = "description",
        tags_col: str = "tags_list",
    ) -> List[ComplexityProfile]:
        """
        Batch estimate pedagogical complexity for all talks in a DataFrame.
        """
        profiles = []
        for _, row in df.iterrows():
            tid = int(row.get(talk_id_col, 0))
            dur = float(row.get(duration_col, 15.0))
            transcript = row.get(transcript_col, None)
            if transcript is None or pd.isna(transcript) or str(transcript).strip() == "":
                transcript = row.get("transcript", "")
            transcript_str = str(transcript) if pd.notna(transcript) else ""

            desc = str(row.get(description_col, "")) if pd.notna(row.get(description_col)) else ""
            raw_tags = row.get(tags_col, None)
            if raw_tags is None or (isinstance(raw_tags, float) and pd.isna(raw_tags)):
                raw_tags = row.get("tags", [])

            if isinstance(raw_tags, str):
                import ast
                try:
                    tag_list = ast.literal_eval(raw_tags) if raw_tags.startswith("[") else [t.strip() for t in raw_tags.split(",")]
                except Exception:
                    tag_list = [raw_tags]
            elif isinstance(raw_tags, (list, set, tuple)):
                tag_list = list(raw_tags)
            else:
                tag_list = []

            p = self.estimate_complexity(
                talk_id=tid,
                duration_minutes=dur,
                transcript=transcript_str,
                description=desc,
                tags=tag_list,
            )
            profiles.append(p)

        return profiles
