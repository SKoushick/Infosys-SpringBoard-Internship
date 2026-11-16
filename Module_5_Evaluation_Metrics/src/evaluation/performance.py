"""
Performance, Latency, and Throughput Benchmarking Engine.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Profiles:
- Offline artifact loading time (vectorizers, embeddings, matrix deserialization).
- Online per-query recommendation inference latency (mean, median, p90, p95, p99).
- Query throughput (QPS).
- Process RAM consumption (RSS before and after loading).
"""

import gc
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


def get_process_memory_mb() -> float:
    """Return resident memory (RSS) in MB for current process."""
    process = psutil.Process()
    return round(process.memory_info().rss / (1024 * 1024), 2)


class RecommenderPerformanceProfiler:
    """Conducts rigorous performance profiling with warmups and repeated trials."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.catalogue_size = len(df)

    def benchmark_offline_loading(self, feature_pipeline_cls, config) -> Dict[str, Any]:
        """Measure cold artifact loading time and memory increase."""
        gc.collect()
        mem_before = get_process_memory_mb()
        t0 = time.perf_counter()

        pipeline = feature_pipeline_cls(
            tfidf_config=config.tfidf_config, field_weights=config.field_weights
        )
        loaded = pipeline.load_artifacts()
        load_time_sec = time.perf_counter() - t0
        mem_after = get_process_memory_mb()

        return {
            "load_successful": loaded,
            "offline_loading_seconds": round(load_time_sec, 3),
            "memory_before_load_mb": mem_before,
            "memory_after_load_mb": mem_after,
            "memory_increase_mb": round(max(0.0, mem_after - mem_before), 2),
        }

    def benchmark_query_latency(
        self,
        recommender,
        model_name: str,
        test_talk_ids: List[int],
        num_warmup: int = 5,
        num_iterations: int = 20,
        top_k: int = 10,
        apply_mmr: bool = False,
    ) -> Dict[str, Any]:
        """
        Benchmark per-query inference latency over repeated trials.
        Strictly isolates online inference from offline loading.
        """
        # 1. Warm-up runs to stabilize JIT/cache and load models into memory
        warmup_candidates = test_talk_ids[:num_warmup] if len(test_talk_ids) >= num_warmup else test_talk_ids
        for tid in warmup_candidates:
            try:
                if hasattr(recommender, "recommend_for_talk"):
                    try:
                        _ = recommender.recommend_for_talk(talk_id=tid, top_k=top_k, apply_mmr=apply_mmr)
                    except TypeError:
                        _ = recommender.recommend_for_talk(talk_id=tid, top_k=top_k)
                else:
                    _ = recommender.recommend(seed_talk_id=tid, top_k=top_k, apply_mmr=apply_mmr)
            except Exception:
                pass

        # 2. Timed inference trials
        latencies_ms: List[float] = []

        for _ in range(num_iterations):
            for tid in test_talk_ids:
                t0 = time.perf_counter()
                if hasattr(recommender, "recommend_for_talk"):
                    try:
                        _ = recommender.recommend_for_talk(talk_id=tid, top_k=top_k, apply_mmr=apply_mmr)
                    except TypeError:
                        _ = recommender.recommend_for_talk(talk_id=tid, top_k=top_k)
                else:
                    _ = recommender.recommend(seed_talk_id=tid, top_k=top_k, apply_mmr=apply_mmr)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(lat_ms)

        total_queries = len(latencies_ms)
        total_time_sec = sum(latencies_ms) / 1000.0
        qps = (total_queries / total_time_sec) if total_time_sec > 0 else 0.0

        return {
            "model_name": model_name,
            "total_queries_profiled": total_queries,
            "top_k": top_k,
            "apply_mmr": apply_mmr,
            "latency_ms": {
                "mean": round(float(np.mean(latencies_ms)), 2),
                "std": round(float(np.std(latencies_ms)), 2),
                "median": round(float(np.median(latencies_ms)), 2),
                "min": round(float(np.min(latencies_ms)), 2),
                "max": round(float(np.max(latencies_ms)), 2),
                "p90": round(float(np.percentile(latencies_ms, 90)), 2),
                "p95": round(float(np.percentile(latencies_ms, 95)), 2),
                "p99": round(float(np.percentile(latencies_ms, 99)), 2),
            },
            "throughput_queries_per_sec": round(qps, 1),
        }

