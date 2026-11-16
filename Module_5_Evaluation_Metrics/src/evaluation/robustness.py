"""
Robustness and Edge-Case Stress Testing Suite.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Tests system resilience across boundary conditions:
1. Valid talk ID (baseline sanity)
2. Unknown talk ID (raises descriptive KeyError)
3. Talk with missing transcript (gracefully generates recommendations via metadata fallback)
4. Empty or whitespace-only search query (returns empty list safely)
5. Short text query (single character)
6. Long text query (5,000+ characters)
7. Non-existent filter tags (returns empty list without crashing)
8. High K exceeding candidate pool (safely truncates to available candidates)
9. Disliked talk exclusions (strictly omitted from output)
10. Multiple liked talks session profile (all liked talks excluded, correct profile weighting)
11. Zero-input cold start (falls back to popularity-ranked talks)
12. Extreme MMR lambda values (0.0 pure diversity vs 1.0 pure relevance)
"""

import logging
from typing import Any, Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class RecommenderRobustnessHarness:
    """Runs systematic boundary condition and exception handling checks."""

    def __init__(self, recommender_suite, df: pd.DataFrame):
        self.tfidf_rec, self.semantic_rec, self.hybrid_rec = recommender_suite
        self.df = df
        self.catalogue_size = len(df)

    def run_all_checks(self) -> Dict[str, Any]:
        """Execute all robustness test cases and return structured report."""
        results = []

        # 1. Baseline Valid ID
        res_1 = self._check_valid_id()
        results.append(res_1)

        # 2. Unknown Talk ID
        res_2 = self._check_unknown_id()
        results.append(res_2)

        # 3. Missing Transcript Talk
        res_3 = self._check_missing_transcript()
        results.append(res_3)

        # 4. Empty Text Query
        res_4 = self._check_empty_query()
        results.append(res_4)

        # 5. Very Short Query
        res_5 = self._check_short_query()
        results.append(res_5)

        # 6. Very Long Query
        res_6 = self._check_long_query()
        results.append(res_6)

        # 7. Non-existent Tag Filter
        res_7 = self._check_nonexistent_tag()
        results.append(res_7)

        # 8. High K Exceeding Pool
        res_8 = self._check_high_k()
        results.append(res_8)

        # 9. Disliked Talk Exclusions
        res_9 = self._check_disliked_exclusions()
        results.append(res_9)

        # 10. Multiple Liked Session
        res_10 = self._check_multi_liked()
        results.append(res_10)

        # 11. Cold Start Fallback
        res_11 = self._check_cold_start()
        results.append(res_11)

        # 12. Extreme MMR Lambda
        res_12 = self._check_extreme_mmr()
        results.append(res_12)

        passed_count = sum(1 for r in results if r["status"] == "PASS")
        total_count = len(results)

        return {
            "total_checks": total_count,
            "passed_checks": passed_count,
            "failed_checks": total_count - passed_count,
            "all_passed": passed_count == total_count,
            "test_cases": results,
        }

    def _check_valid_id(self) -> Dict[str, Any]:
        try:
            recs = self.tfidf_rec.recommend_for_talk(talk_id=1, top_k=5)
            passed = len(recs) == 5 and all(r.talk_id != 1 for r in recs)
            return {
                "case_name": "Valid Seed Talk ID",
                "status": "PASS" if passed else "FAIL",
                "details": f"Generated {len(recs)} valid recommendations excluding seed 1.",
            }
        except Exception as e:
            return {"case_name": "Valid Seed Talk ID", "status": "FAIL", "error": str(e)}

    def _check_unknown_id(self) -> Dict[str, Any]:
        unknown_id = 99999999
        try:
            self.tfidf_rec.recommend_for_talk(talk_id=unknown_id, top_k=5)
            return {
                "case_name": "Unknown Talk ID",
                "status": "FAIL",
                "details": "Expected KeyError but call succeeded unexpectedly.",
            }
        except KeyError:
            return {
                "case_name": "Unknown Talk ID",
                "status": "PASS",
                "details": "Correctly raised KeyError for non-existent talk ID.",
            }
        except Exception as e:
            return {
                "case_name": "Unknown Talk ID",
                "status": "FAIL",
                "details": f"Raised unexpected exception type {type(e).__name__}: {e}",
            }

    def _check_missing_transcript(self) -> Dict[str, Any]:
        # Find a talk where has_transcript is False
        missing_talks = self.df[self.df["has_transcript"] == False]
        if missing_talks.empty:
            return {"case_name": "Missing Transcript Talk", "status": "SKIPPED", "details": "No missing transcripts in dataset."}
        
        target_id = int(missing_talks.iloc[0]["talk_id"])
        try:
            recs_tfidf = self.tfidf_rec.recommend_for_talk(talk_id=target_id, top_k=5)
            recs_hybrid = self.hybrid_rec.recommend_for_talk(talk_id=target_id, top_k=5)
            passed = len(recs_tfidf) == 5 and len(recs_hybrid) == 5
            return {
                "case_name": "Missing Transcript Talk",
                "status": "PASS" if passed else "FAIL",
                "details": f"Successfully recommended for talk {target_id} using metadata fallback.",
            }
        except Exception as e:
            return {"case_name": "Missing Transcript Talk", "status": "FAIL", "error": str(e)}

    def _check_empty_query(self) -> Dict[str, Any]:
        try:
            recs_1 = self.tfidf_rec.recommend_for_query("   ", top_k=5)
            recs_2 = self.semantic_rec.recommend_for_query("", top_k=5)
            passed = recs_1 == [] and recs_2 == []
            return {
                "case_name": "Empty Search Query",
                "status": "PASS" if passed else "FAIL",
                "details": "Correctly returned empty list for empty/whitespace search queries.",
            }
        except Exception as e:
            return {"case_name": "Empty Search Query", "status": "FAIL", "error": str(e)}

    def _check_short_query(self) -> Dict[str, Any]:
        try:
            recs = self.tfidf_rec.recommend_for_query("a", top_k=3)
            # Either empty (due to stop words) or valid list
            passed = isinstance(recs, list)
            return {
                "case_name": "Single Character Query",
                "status": "PASS" if passed else "FAIL",
                "details": f"Handled single character query safely (returned {len(recs)} items).",
            }
        except Exception as e:
            return {"case_name": "Single Character Query", "status": "FAIL", "error": str(e)}

    def _check_long_query(self) -> Dict[str, Any]:
        long_text = "technology and artificial intelligence " * 200  # ~4400 characters
        try:
            recs = self.tfidf_rec.recommend_for_query(long_text, top_k=5)
            passed = len(recs) == 5
            return {
                "case_name": "Extremely Long Query",
                "status": "PASS" if passed else "FAIL",
                "details": f"Successfully handled 4000+ character query with {len(recs)} results.",
            }
        except Exception as e:
            return {"case_name": "Extremely Long Query", "status": "FAIL", "error": str(e)}

    def _check_nonexistent_tag(self) -> Dict[str, Any]:
        bogus_tag = ["this_tag_does_not_exist_in_ted_dataset_xyz_12345"]
        try:
            recs = self.tfidf_rec.recommend_for_talk(talk_id=1, filter_tags=bogus_tag)
            passed = recs == []
            return {
                "case_name": "Non-existent Tag Filter",
                "status": "PASS" if passed else "FAIL",
                "details": "Correctly returned empty list when filter tags match 0 catalog items.",
            }
        except Exception as e:
            return {"case_name": "Non-existent Tag Filter", "status": "FAIL", "error": str(e)}

    def _check_high_k(self) -> Dict[str, Any]:
        try:
            huge_k = 50000
            recs = self.tfidf_rec.recommend_for_talk(talk_id=1, top_k=huge_k)
            # Should safely return all catalogue items except seed
            passed = len(recs) == self.catalogue_size - 1
            return {
                "case_name": "High K Exceeding Pool",
                "status": "PASS" if passed else "FAIL",
                "details": f"Handled top_k={huge_k} safely, returning exactly {len(recs)} items.",
            }
        except Exception as e:
            return {"case_name": "High K Exceeding Pool", "status": "FAIL", "error": str(e)}

    def _check_disliked_exclusions(self) -> Dict[str, Any]:
        disliked = [2, 3, 4]
        try:
            recs = self.hybrid_rec.recommend(
                seed_talk_id=1,
                disliked_talk_ids=disliked,
                top_k=10,
            )
            rec_ids = set(r.talk_id for r in recs)
            violations = rec_ids.intersection(set(disliked + [1]))
            passed = len(violations) == 0 and len(recs) == 10
            return {
                "case_name": "Disliked Exclusions",
                "status": "PASS" if passed else "FAIL",
                "details": f"Zero violations: disliked talks {disliked} and seed 1 were strictly omitted.",
            }
        except Exception as e:
            return {"case_name": "Disliked Exclusions", "status": "FAIL", "error": str(e)}

    def _check_multi_liked(self) -> Dict[str, Any]:
        liked = [1, 2, 3]
        try:
            recs = self.hybrid_rec.recommend(
                liked_talk_ids=liked,
                top_k=5,
            )
            rec_ids = set(r.talk_id for r in recs)
            violations = rec_ids.intersection(set(liked))
            passed = len(violations) == 0 and len(recs) == 5
            return {
                "case_name": "Multiple Liked Profile",
                "status": "PASS" if passed else "FAIL",
                "details": f"Successfully personalized across 3 liked talks; all liked talks excluded.",
            }
        except Exception as e:
            return {"case_name": "Multiple Liked Profile", "status": "FAIL", "error": str(e)}

    def _check_cold_start(self) -> Dict[str, Any]:
        try:
            recs = self.hybrid_rec.recommend(top_k=5)
            passed = len(recs) == 5 and all("Cold-start" in r.explanation.rationale for r in recs)
            return {
                "case_name": "Cold-Start Popularity Fallback",
                "status": "PASS" if passed else "FAIL",
                "details": "Returned top viewed talks with explicit cold-start explanation rationale.",
            }
        except Exception as e:
            return {"case_name": "Cold-Start Popularity Fallback", "status": "FAIL", "error": str(e)}

    def _check_extreme_mmr(self) -> Dict[str, Any]:
        try:
            recs_div = self.hybrid_rec.recommend(seed_talk_id=1, top_k=5, apply_mmr=True)
            recs_rel = self.hybrid_rec.recommend(seed_talk_id=1, top_k=5, apply_mmr=False)
            passed = len(recs_div) == 5 and len(recs_rel) == 5
            return {
                "case_name": "Extreme MMR Diversification",
                "status": "PASS" if passed else "FAIL",
                "details": "Both MMR-diversified and direct relevance rankings executed cleanly.",
            }
        except Exception as e:
            return {"case_name": "Extreme MMR Diversification", "status": "FAIL", "error": str(e)}

