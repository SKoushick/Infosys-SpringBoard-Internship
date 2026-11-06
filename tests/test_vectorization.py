"""Unit tests for feature engineering, TF-IDF vectorizers, and semantic embeddings."""

from pathlib import Path
import numpy as np
import pytest
from scipy import sparse

from ted_recommender.config import RecommenderConfig
from ted_recommender.data import load_dataset
from ted_recommender.features import FeaturePipeline


@pytest.fixture(scope="module")
def pipeline_and_data():
    df = load_dataset(stage="features")
    config = RecommenderConfig()
    pipeline = FeaturePipeline(tfidf_config=config.tfidf_config, field_weights=config.field_weights)
    pipeline.load_artifacts()
    return pipeline, df


def test_tfidf_artifacts_loaded(pipeline_and_data):
    """Ensure loaded TF-IDF artifacts match the 2550 dataset rows."""
    pipeline, df = pipeline_and_data
    assert pipeline.combined_tfidf_matrix is not None
    assert pipeline.combined_tfidf_matrix.shape[0] == 2550
    assert len(pipeline.vectorizers) == 4
    for field in ["title", "description", "tags", "transcript"]:
        assert field in pipeline.vectorizers
        assert hasattr(pipeline.vectorizers[field], "vocabulary_")


def test_semantic_embeddings_shape(pipeline_and_data):
    """Ensure semantic embeddings have shape (2550, 384) and are unit-normalized."""
    pipeline, df = pipeline_and_data
    assert pipeline.embedding_matrix is not None
    assert pipeline.embedding_matrix.shape == (2550, 384)

    # Unit normalization test (norms should be approx 1.0)
    norms = np.linalg.norm(pipeline.embedding_matrix[:10], axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-3)


def test_query_tfidf_transformation(pipeline_and_data):
    """Verify transforming an arbitrary text query into the combined TF-IDF space."""
    pipeline, _ = pipeline_and_data
    q_vec = pipeline.transform_query_tfidf("artificial intelligence and machine learning")
    assert isinstance(q_vec, sparse.csr_matrix)
    assert q_vec.shape[1] == pipeline.combined_tfidf_matrix.shape[1]
    # Unit norm
    norm = sparse.linalg.norm(q_vec)
    assert pytest.approx(norm, 0.01) == 1.0

