"""
Tests for the SimilarityCalculator module.
"""

import pytest
import numpy as np
from unittest.mock import patch

from semantic_code_analyzer.similarity_calculator import (
    SimilarityCalculator,
    DistanceMetric,
    SimilarityResult,
    FileSimilarity
)


class TestSimilarityCalculator:
    """Test cases for SimilarityCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a SimilarityCalculator instance."""
        return SimilarityCalculator()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)  # For reproducible tests
        return {
            'target': np.random.rand(768).astype(np.float32),
            'reference1': np.random.rand(768).astype(np.float32),
            'reference2': np.random.rand(768).astype(np.float32),
            'similar': np.random.rand(768).astype(np.float32),
        }

    @pytest.fixture
    def file_embeddings(self, sample_embeddings):
        """Create file embeddings dictionary."""
        return {
            'file1.py': sample_embeddings['reference1'],
            'file2.py': sample_embeddings['reference2'],
            'file3.py': sample_embeddings['similar']
        }

    def test_init_default(self):
        """Test default initialization."""
        calc = SimilarityCalculator()
        assert calc.distance_metric == DistanceMetric.EUCLIDEAN
        assert calc.normalize_scores is True
        assert calc.enable_caching is True

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        calc = SimilarityCalculator(
            distance_metric=DistanceMetric.COSINE,
            normalize_scores=False,
            enable_caching=False
        )
        assert calc.distance_metric == DistanceMetric.COSINE
        assert calc.normalize_scores is False
        assert calc.enable_caching is False

    def test_init_with_string_metric(self):
        """Test initialization with string distance metric."""
        calc = SimilarityCalculator(distance_metric="cosine")
        assert calc.distance_metric == DistanceMetric.COSINE

    def test_calculate_similarity_score_detailed(self, calculator, sample_embeddings):
        """Test detailed similarity score calculation."""
        target = sample_embeddings['target']
        references = [sample_embeddings['reference1'], sample_embeddings['reference2']]

        result = calculator.calculate_similarity_score(target, references, return_details=True)

        assert isinstance(result, SimilarityResult)
        assert 0 <= result.max_similarity <= 1
        assert 0 <= result.mean_similarity <= 1
        assert 0 <= result.median_similarity <= 1
        assert result.std_similarity >= 0
        assert 0 <= result.min_similarity <= 1
        assert len(result.similarity_scores) == 2
        assert result.distance_metric == "euclidean"

    def test_calculate_similarity_score_simple(self, calculator, sample_embeddings):
        """Test simple similarity score calculation."""
        target = sample_embeddings['target']
        references = [sample_embeddings['reference1'], sample_embeddings['reference2']]

        result = calculator.calculate_similarity_score(target, references, return_details=False)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_calculate_similarity_empty_references(self, calculator, sample_embeddings):
        """Test similarity calculation with empty references."""
        target = sample_embeddings['target']

        # Detailed result
        result_detailed = calculator.calculate_similarity_score(target, [], return_details=True)
        assert isinstance(result_detailed, SimilarityResult)
        assert result_detailed.max_similarity == 0.0
        assert result_detailed.similarity_scores == []

        # Simple result
        result_simple = calculator.calculate_similarity_score(target, [], return_details=False)
        assert result_simple == 0.0

    def test_euclidean_distance_function(self, calculator):
        """Test euclidean distance calculation."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.0, 2.0, 3.0])  # Identical

        similarity = calculator._euclidean_distance(emb1, emb2)
        assert similarity == 1.0  # Should be 1.0 for identical vectors

        emb3 = np.array([4.0, 5.0, 6.0])  # Different
        similarity2 = calculator._euclidean_distance(emb1, emb3)
        assert 0 < similarity2 < 1  # Should be between 0 and 1

    def test_cosine_distance_function(self, calculator):
        """Test cosine similarity calculation."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical

        similarity = calculator._cosine_distance(emb1, emb2)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical unit vectors

        emb3 = np.array([0.0, 1.0, 0.0])  # Orthogonal
        similarity2 = calculator._cosine_distance(emb1, emb3)
        assert abs(similarity2) < 1e-6  # Should be 0.0 for orthogonal vectors

    def test_cosine_distance_zero_vectors(self, calculator):
        """Test cosine similarity with zero vectors."""
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])

        similarity = calculator._cosine_distance(emb1, emb2)
        assert similarity == 0.0  # Should handle zero vectors gracefully

    def test_find_most_similar_files(self, calculator, sample_embeddings, file_embeddings):
        """Test finding most similar files."""
        target = sample_embeddings['target']

        similar_files = calculator.find_most_similar_files(
            target, file_embeddings, top_k=2
        )

        assert len(similar_files) == 2
        assert all(isinstance(sim, FileSimilarity) for sim in similar_files)

        # Check that results are sorted by similarity (highest first)
        assert similar_files[0].similarity_score >= similar_files[1].similarity_score

        # Check ranks are assigned correctly
        assert similar_files[0].rank == 1
        assert similar_files[1].rank == 2

    def test_find_most_similar_files_empty(self, calculator, sample_embeddings):
        """Test finding similar files with empty file embeddings."""
        target = sample_embeddings['target']

        similar_files = calculator.find_most_similar_files(target, {})
        assert similar_files == []

    def test_calculate_cross_similarity_matrix(self, calculator, sample_embeddings):
        """Test cross-similarity matrix calculation."""
        embeddings_a = [sample_embeddings['target'], sample_embeddings['reference1']]
        embeddings_b = [sample_embeddings['reference2'], sample_embeddings['similar']]

        result = calculator.calculate_cross_similarity_matrix(
            embeddings_a, embeddings_b
        )

        assert 'similarity_matrix' in result
        assert result['shape'] == (2, 2)
        assert 'max_similarity' in result
        assert 'mean_similarity' in result
        assert 'min_similarity' in result

        matrix = result['similarity_matrix']
        assert matrix.shape == (2, 2)
        assert np.all(matrix >= 0)  # All similarities should be non-negative

    def test_calculate_cross_similarity_matrix_self(self, calculator, sample_embeddings):
        """Test cross-similarity matrix with same embeddings."""
        embeddings = [sample_embeddings['target'], sample_embeddings['reference1']]

        result = calculator.calculate_cross_similarity_matrix(embeddings)

        matrix = result['similarity_matrix']
        assert matrix.shape == (2, 2)
        # Diagonal should have high similarity (self-similarity)
        assert matrix[0, 0] >= matrix[0, 1]
        assert matrix[1, 1] >= matrix[1, 0]

    def test_calculate_aggregate_similarity(self, calculator, sample_embeddings, file_embeddings):
        """Test aggregate similarity calculation."""
        commit_embeddings = {
            'commit_file1.py': sample_embeddings['target'],
            'commit_file2.py': sample_embeddings['reference1']
        }

        result = calculator.calculate_aggregate_similarity(
            commit_embeddings, file_embeddings, aggregation_method="max"
        )

        assert 'aggregate_similarity' in result
        assert 'method' in result
        assert 'file_similarities' in result
        assert result['method'] == "max"
        assert 0 <= result['aggregate_similarity'] <= 1

    def test_different_distance_metrics(self, sample_embeddings):
        """Test different distance metrics produce different results."""
        target = sample_embeddings['target']
        references = [sample_embeddings['reference1']]

        results = {}
        for metric in DistanceMetric:
            calc = SimilarityCalculator(distance_metric=metric)
            result = calc.calculate_similarity_score(target, references, return_details=False)
            results[metric] = result

        # Different metrics should potentially produce different results
        # (though they might be similar for some data)
        assert len(results) == len(DistanceMetric)
        assert all(0 <= score <= 1 or metric == DistanceMetric.DOT_PRODUCT
                  for metric, score in results.items())

    def test_normalize_similarities(self, calculator):
        """Test similarity normalization."""
        similarities = np.array([0.1, 0.5, 0.9, 0.3])
        normalized = calculator._normalize_similarities(similarities)

        assert len(normalized) == len(similarities)
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0

    def test_normalize_similarities_equal_values(self, calculator):
        """Test normalization with equal values."""
        similarities = np.array([0.5, 0.5, 0.5])
        normalized = calculator._normalize_similarities(similarities)

        # All values should be 0.5 when all inputs are equal
        assert np.allclose(normalized, 0.5)

    def test_normalize_similarities_empty(self, calculator):
        """Test normalization with empty array."""
        similarities = np.array([])
        normalized = calculator._normalize_similarities(similarities)

        assert len(normalized) == 0

    def test_get_similarity_statistics(self, calculator):
        """Test similarity statistics calculation."""
        similarities = [0.1, 0.3, 0.5, 0.7, 0.9]
        stats = calculator.get_similarity_statistics(similarities)

        assert stats['count'] == 5
        assert stats['mean'] == 0.5
        assert stats['min'] == 0.1
        assert stats['max'] == 0.9
        assert stats['median'] == 0.5
        assert 'std' in stats
        assert 'q1' in stats
        assert 'q3' in stats

    def test_get_similarity_statistics_empty(self, calculator):
        """Test statistics with empty list."""
        stats = calculator.get_similarity_statistics([])

        assert stats['count'] == 0
        assert stats['mean'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0

    def test_compare_distance_metrics(self, calculator):
        """Test comparison of distance metrics."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([2.0, 3.0, 4.0])

        results = calculator.compare_distance_metrics(emb1, emb2)

        assert len(results) == len(DistanceMetric)
        for metric in DistanceMetric:
            assert metric.value in results
            assert isinstance(results[metric.value], float)

    def test_clear_cache(self, calculator):
        """Test cache clearing functionality."""
        # Calculator with caching enabled
        calc = SimilarityCalculator(enable_caching=True)

        # Add something to cache (simulate)
        calc._similarity_cache = {"test": "value"}

        calc.clear_cache()
        assert len(calc._similarity_cache) == 0

    def test_get_cache_info(self, calculator):
        """Test cache information retrieval."""
        # Calculator with caching enabled
        calc = SimilarityCalculator(enable_caching=True)
        info = calc.get_cache_info()

        assert info['enabled'] is True
        assert 'size' in info

        # Calculator with caching disabled
        calc_no_cache = SimilarityCalculator(enable_caching=False)
        info_no_cache = calc_no_cache.get_cache_info()

        assert info_no_cache['enabled'] is False
        assert info_no_cache['size'] == 0


class TestDistanceMetric:
    """Test cases for DistanceMetric enum."""

    def test_distance_metric_values(self):
        """Test that all expected distance metrics are available."""
        expected_metrics = [
            "euclidean", "cosine", "manhattan", "chebyshev", "dot_product"
        ]

        actual_metrics = [metric.value for metric in DistanceMetric]

        for expected in expected_metrics:
            assert expected in actual_metrics

    def test_distance_metric_from_string(self):
        """Test creating DistanceMetric from string."""
        metric = DistanceMetric("euclidean")
        assert metric == DistanceMetric.EUCLIDEAN

        metric2 = DistanceMetric("cosine")
        assert metric2 == DistanceMetric.COSINE


class TestSimilarityResult:
    """Test cases for SimilarityResult dataclass."""

    def test_similarity_result_creation(self):
        """Test creating SimilarityResult instance."""
        result = SimilarityResult(
            max_similarity=0.8,
            mean_similarity=0.6,
            median_similarity=0.7,
            std_similarity=0.1,
            min_similarity=0.4,
            similarity_scores=[0.4, 0.6, 0.8],
            distance_metric="euclidean"
        )

        assert result.max_similarity == 0.8
        assert result.mean_similarity == 0.6
        assert result.similarity_scores == [0.4, 0.6, 0.8]
        assert result.distance_metric == "euclidean"


class TestFileSimilarity:
    """Test cases for FileSimilarity dataclass."""

    def test_file_similarity_creation(self):
        """Test creating FileSimilarity instance."""
        file_sim = FileSimilarity(
            file_path="test.py",
            similarity_score=0.75,
            distance=0.25,
            rank=1
        )

        assert file_sim.file_path == "test.py"
        assert file_sim.similarity_score == 0.75
        assert file_sim.distance == 0.25
        assert file_sim.rank == 1


# Edge cases and performance tests
class TestSimilarityCalculatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_similar_embeddings(self):
        """Test with nearly identical embeddings."""
        calc = SimilarityCalculator()

        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.001, 2.001, 3.001])  # Very similar

        similarity = calc._euclidean_distance(emb1, emb2)
        assert similarity > 0.9  # Should be very high similarity

    def test_very_different_embeddings(self):
        """Test with very different embeddings."""
        calc = SimilarityCalculator()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 0.0, 1.0])  # Very different

        similarity = calc._euclidean_distance(emb1, emb2)
        assert similarity < 0.5  # Should be low similarity

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings."""
        calc = SimilarityCalculator()

        np.random.seed(42)
        emb1 = np.random.rand(1000)
        emb2 = np.random.rand(1000)

        similarity = calc._euclidean_distance(emb1, emb2)
        assert 0 <= similarity <= 1

    def test_single_element_embeddings(self):
        """Test with single-element embeddings."""
        calc = SimilarityCalculator()

        emb1 = np.array([1.0])
        emb2 = np.array([2.0])

        similarity = calc._euclidean_distance(emb1, emb2)
        assert 0 <= similarity <= 1

    @patch('semantic_code_analyzer.similarity_calculator.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        calc = SimilarityCalculator()

        # Should log initialization
        mock_logger.info.assert_called()

    def test_manhattan_distance_calculation(self):
        """Test Manhattan distance calculation."""
        calc = SimilarityCalculator(distance_metric=DistanceMetric.MANHATTAN)

        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.0, 2.0, 3.0])  # Identical

        similarity = calc._manhattan_distance(emb1, emb2)
        assert similarity == 1.0  # Should be 1.0 for identical vectors

    def test_chebyshev_distance_calculation(self):
        """Test Chebyshev distance calculation."""
        calc = SimilarityCalculator(distance_metric=DistanceMetric.CHEBYSHEV)

        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([1.0, 2.0, 3.0])  # Identical

        similarity = calc._chebyshev_distance(emb1, emb2)
        assert similarity == 1.0  # Should be 1.0 for identical vectors

    def test_dot_product_similarity_calculation(self):
        """Test dot product similarity calculation."""
        calc = SimilarityCalculator(distance_metric=DistanceMetric.DOT_PRODUCT)

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical unit vectors

        similarity = calc._dot_product_similarity(emb1, emb2)
        assert similarity == 1.0  # Dot product of identical unit vectors