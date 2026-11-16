"""
Performance, Latency, and Throughput Benchmarking Engine.
Module 5: Evaluation Metrics | Infosys Springboard 7.0

Profiles:
- Offline artifact loading time (vectorizers, embeddings, matrix deserialization).
- Online per-query recommendation inference latency (mean, median, std, p90, p95, p99).
- Query throughput (QPS).
- Process RAM consumption (RSS before and after loading).
- Reports warmup count and timed iteration count transparently.
"""

import gc
import inspect
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
        recommender: Any,
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
        target_fn = recommender.recommend_for_talk if hasattr(recommender, "recommend_for_talk") else recommender.recommend
        sig = inspect.signature(target_fn)
        supports_apply_mmr = "apply_mmr" in sig.parameters

        def dispatch_call(talk_id: int):
            kwargs: Dict[str, Any] = {"top_k": top_k}
            if "talk_id" in sig.parameters:
                kwargs["talk_id"] = talk_id
            elif "seed_talk_id" in sig.parameters:
                kwargs["seed_talk_id"] = talk_id
            if supports_apply_mmr:
                kwargs["apply_mmr"] = apply_mmr
            return target_fn(**kwargs)

        # 1. Warm-up runs to stabilize JIT/cache and load models into memory
        warmup_candidates = test_talk_ids[:num_warmup] if len(test_talk_ids) >= num_warmup else test_talk_ids
        for tid in warmup_candidates:
            try:
                _ = dispatch_call(tid)
            except Exception as e:
                logger.debug(f"Warmup notice for talk {tid}: {e}")

        # 2. Timed inference trials
        latencies_ms: List[float] = []
        failed_count = 0

        for _ in range(num_iterations):
            for tid in test_talk_ids:
                t0 = time.perf_counter()
                try:
                    _ = dispatch_call(tid)
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(lat_ms)
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Latency benchmark error for query {tid} on {model_name}: {e}")

        total_queries = len(latencies_ms)
        total_time_sec = sum(latencies_ms) / 1000.0
        qps = (total_queries / total_time_sec) if total_time_sec > 0 else 0.0

        return {
            "model_name": model_name,
            "warmup_queries_executed": len(warmup_candidates),
            "timed_iterations": num_iterations,
            "total_queries_profiled": total_queries,
            "failed_queries_count": failed_count,
            "top_k": top_k,
            "apply_mmr": apply_mmr,
            "latency_ms": {
                "mean": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
                "std": round(float(np.std(latencies_ms)), 2) if latencies_ms else 0.0,
                "median": round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
                "min": round(float(np.min(latencies_ms)), 2) if latencies_ms else 0.0,
                "max": round(float(np.max(latencies_ms)), 2) if latencies_ms else 0.0,
                "p90": round(float(np.percentile(latencies_ms, 90)), 2) if latencies_ms else 0.0,
                "p95": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
                "p99": round(float(np.percentile(latencies_ms, 99)), 2) if latencies_ms else 0.0,
            },
            "throughput_queries_per_sec": round(qps, 1),
        }
