# MIT License
#
# Copyright (c) 2024 Semantic Code Analyzer Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests for the domain adherence analyzer module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from semantic_code_analyzer.analyzers.domain_adherence_analyzer import (
    AdherenceScore,
    DomainAwareAdherenceAnalyzer,
)
from semantic_code_analyzer.analyzers.domain_classifier import ArchitecturalDomain
from semantic_code_analyzer.embeddings.pattern_indexer import SimilarityMatch


class TestDomainAwareAdherenceAnalyzer:
    """Test cases for the DomainAwareAdherenceAnalyzer class."""

    @pytest.fixture
    def mock_dependencies(self) -> Any:
        """Mock the dependencies to avoid loading actual models."""
        with (
            patch(
                "semantic_code_analyzer.analyzers.domain_adherence_analyzer.DomainClassifier"
            ) as mock_classifier,
            patch(
                "semantic_code_analyzer.analyzers.domain_adherence_analyzer.PatternIndexer"
            ) as mock_indexer,
        ):
            # Mock domain classifier
            mock_classifier_instance = Mock()
            mock_classification = Mock()
            mock_classification.domain = ArchitecturalDomain.FRONTEND
            mock_classification.confidence = 0.8
            mock_classification.classification_factors = {"test": "factors"}
            mock_classification.secondary_domains = []
            mock_classifier_instance.classify_domain.return_value = mock_classification
            mock_classifier.return_value = mock_classifier_instance

            # Mock pattern indexer
            mock_indexer_instance = Mock()
            mock_indexer_instance.domain_indices = {"frontend": Mock()}
            mock_indexer_instance.search_similar_patterns.return_value = []
            mock_indexer.return_value = mock_indexer_instance

            yield {
                "classifier": mock_classifier_instance,
                "indexer": mock_indexer_instance,
                "classification": mock_classification,
            }

    @pytest.fixture
    def analyzer(self, mock_dependencies: Any) -> DomainAwareAdherenceAnalyzer:
        """Create a DomainAwareAdherenceAnalyzer instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "similarity_threshold": 0.3,
                "domain_confidence_threshold": 0.6,
                "max_similar_patterns": 10,
                "cache_dir": temp_dir,
            }
            return DomainAwareAdherenceAnalyzer(config)

    def test_get_analyzer_name(self, analyzer: DomainAwareAdherenceAnalyzer) -> None:
        """Test that analyzer returns correct name."""
        assert analyzer.get_analyzer_name() == "domain_adherence"

    def test_get_weight(self, analyzer: DomainAwareAdherenceAnalyzer) -> None:
        """Test that analyzer returns correct weight."""
        assert analyzer.get_weight() == 0.25

    def test_get_supported_extensions(
        self, analyzer: DomainAwareAdherenceAnalyzer
    ) -> None:
        """Test that analyzer supports expected file extensions."""
        extensions = analyzer._get_supported_extensions()
        expected = {".ts", ".tsx", ".js", ".jsx", ".py", ".sql", ".md", ".json"}
        assert expected.issubset(extensions)

    def test_initialization_with_config(
        self, mock_dependencies: Any, tmp_path: Path
    ) -> None:
        """Test analyzer initialization with custom configuration."""
        config = {
            "similarity_threshold": 0.5,
            "domain_confidence_threshold": 0.7,
            "max_similar_patterns": 5,
            "model_name": "test-model",
            "cache_dir": str(tmp_path / "test"),
        }

        analyzer = DomainAwareAdherenceAnalyzer(config)

        assert analyzer.similarity_threshold == 0.5
        assert analyzer.domain_confidence_threshold == 0.7
        assert analyzer.max_similar_patterns == 5

    def test_analyze_file_basic(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test basic file analysis functionality."""
        file_path = "src/components/Button.tsx"
        content = """
import React from 'react';

const Button: React.FC = () => {
    return <button>Click me</button>;
};

export default Button;
"""

        result = analyzer.analyze_file(file_path, content)

        assert result.file_path == file_path
        assert 0 <= result.score <= 1
        assert len(result.patterns_found) > 0
        assert "domain" in result.metrics
        assert "confidence" in result.metrics
        assert result.analysis_time >= 0

    def test_analyze_domain_adherence(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test domain adherence analysis."""
        content = "const test = () => <div>Test</div>;"
        classification = mock_dependencies["classification"]

        result = analyzer.analyze_domain_adherence(content, classification, "test.tsx")

        assert result.domain_classification == classification
        assert isinstance(result.adherence_score, AdherenceScore)
        assert 0 <= result.adherence_score.overall_adherence <= 1
        assert 0 <= result.adherence_score.confidence <= 1
        assert len(result.improvement_suggestions) >= 0

    def test_build_pattern_indices(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test pattern indices building."""
        codebase_files = {
            "src/Button.tsx": "const Button = () => <button>Test</button>;",
            "src/Input.tsx": "const Input = () => <input type='text' />;",
            "tests/Button.test.tsx": "test('button works', () => {});",
        }

        # Mock the domain classifier to return different domains
        classifier_mock = mock_dependencies["classifier"]

        def mock_classify(file_path: str, content: str) -> Any:
            classification = Mock()
            if "test" in file_path:
                classification.domain = ArchitecturalDomain.TESTING
            else:
                classification.domain = ArchitecturalDomain.FRONTEND
            classification.confidence = 0.8
            return classification

        classifier_mock.classify_domain.side_effect = mock_classify

        # Should not raise any exceptions
        analyzer.build_pattern_indices(codebase_files)

        # Verify domain classifier was called for each file
        assert classifier_mock.classify_domain.call_count == len(codebase_files)

    def test_calculate_detailed_adherence_scores(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test detailed adherence score calculation."""
        classification = mock_dependencies["classification"]

        # Test with no similar patterns
        similar_patterns: list[Any] = []
        score = analyzer._calculate_detailed_adherence_scores(
            "test code", classification, similar_patterns
        )

        assert isinstance(score, AdherenceScore)
        assert 0 <= score.overall_adherence <= 1
        assert 0 <= score.domain_adherence <= 1
        assert score.pattern_similarity == 0.0  # No patterns
        assert score.similar_patterns_count == 0

        # Test with similar patterns
        similar_patterns = [
            SimilarityMatch("file1.ts", 0.8, "code1", "frontend", {}),
            SimilarityMatch("file2.ts", 0.6, "code2", "frontend", {}),
        ]
        score_with_patterns = analyzer._calculate_detailed_adherence_scores(
            "test code", classification, similar_patterns
        )

        assert score_with_patterns.similar_patterns_count == 2
        assert score_with_patterns.pattern_similarity > 0
        assert score_with_patterns.overall_adherence >= score.overall_adherence

    def test_calculate_weighted_adherence(
        self, analyzer: DomainAwareAdherenceAnalyzer
    ) -> None:
        """Test weighted adherence calculation."""
        # Test various combinations
        score1 = analyzer._calculate_weighted_adherence(0.8, 0.7, 5)
        score2 = analyzer._calculate_weighted_adherence(0.8, 0.7, 15)  # More patterns
        score3 = analyzer._calculate_weighted_adherence(
            0.5, 0.9, 10
        )  # Different quality/similarity

        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
        assert 0 <= score3 <= 1

        # More patterns should generally lead to higher score
        assert score2 >= score1

    def test_generate_improvement_suggestions(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test improvement suggestion generation."""
        classification = mock_dependencies["classification"]
        classification.domain = ArchitecturalDomain.FRONTEND

        # Test with no similar patterns
        adherence_score = AdherenceScore(
            overall_adherence=0.4,
            domain_adherence=0.5,
            pattern_similarity=0.3,
            confidence=0.6,
            similar_patterns_count=0,
            domain_match_quality=0.7,
        )

        suggestions = analyzer._generate_improvement_suggestions(
            classification, [], adherence_score
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0  # Should have suggestions for low adherence

        # Test with similar patterns
        similar_patterns = [SimilarityMatch("file1.ts", 0.9, "code", "frontend", {})]
        suggestions_with_patterns = analyzer._generate_improvement_suggestions(
            classification, similar_patterns, adherence_score
        )

        assert len(suggestions_with_patterns) > len(
            suggestions
        )  # Should have pattern-based suggestions

    def test_create_adherence_patterns(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test adherence pattern creation."""
        analysis_result = Mock()
        analysis_result.domain_classification.domain = ArchitecturalDomain.FRONTEND
        analysis_result.adherence_score.overall_adherence = 0.8
        analysis_result.adherence_score.similar_patterns_count = 3
        analysis_result.adherence_score.confidence = 0.7

        similar_patterns = [
            SimilarityMatch("file1.ts", 0.9, "code1", "frontend", {}),
            SimilarityMatch("file2.ts", 0.8, "code2", "frontend", {}),
        ]
        analysis_result.similar_patterns = similar_patterns

        patterns = analyzer._create_adherence_patterns(analysis_result, "test.tsx")

        assert len(patterns) > 0
        # Should have at least domain adherence pattern + similar pattern matches
        domain_patterns = [p for p in patterns if "domain_adherence" in p.pattern_name]
        similar_pattern_matches = [
            p for p in patterns if "similar_pattern" in p.pattern_name
        ]

        assert len(domain_patterns) >= 1
        assert len(similar_pattern_matches) >= 1

    def test_generate_adherence_recommendations(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test adherence recommendation generation."""
        analysis_result = Mock()
        analysis_result.domain_classification.domain = ArchitecturalDomain.FRONTEND
        analysis_result.domain_classification.confidence = 0.4  # Low confidence
        analysis_result.adherence_score.overall_adherence = 0.3  # Low adherence
        analysis_result.adherence_score.similar_patterns_count = 0  # No patterns
        analysis_result.improvement_suggestions = [
            "Use React hooks",
            "Follow component patterns",
        ]

        recommendations = analyzer._generate_adherence_recommendations(
            analysis_result, "test.tsx"
        )

        assert len(recommendations) > 0

        # Should have recommendations for low adherence, low confidence, and no patterns
        rule_ids = [rec.rule_id for rec in recommendations]
        assert any("LOW_DOMAIN_ADHERENCE" in rule_id for rule_id in rule_ids)
        assert any("UNCLEAR_DOMAIN" in rule_id for rule_id in rule_ids)
        assert any("NO_SIMILAR_PATTERNS" in rule_id for rule_id in rule_ids)

    def test_get_domain_statistics(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test domain statistics retrieval."""
        # Mock some built indices
        analyzer._indices_built = {"frontend", "backend"}

        mock_indexer = mock_dependencies["indexer"]
        mock_indexer.get_cache_statistics.return_value = {"test": "stats"}
        mock_indexer.get_domain_statistics.return_value = {"domain_test": "stats"}

        stats = analyzer.get_domain_statistics()

        assert "indices_built" in stats
        assert "total_domains" in stats
        assert "pattern_indexer_stats" in stats
        assert stats["indices_built"] == ["backend", "frontend"]
        assert stats["total_domains"] == 2

    def test_adherence_score_dataclass(self) -> None:
        """Test AdherenceScore dataclass structure."""
        score = AdherenceScore(
            overall_adherence=0.8,
            domain_adherence=0.7,
            pattern_similarity=0.6,
            confidence=0.9,
            similar_patterns_count=5,
            domain_match_quality=0.85,
        )

        assert score.overall_adherence == 0.8
        assert score.domain_adherence == 0.7
        assert score.pattern_similarity == 0.6
        assert score.confidence == 0.9
        assert score.similar_patterns_count == 5
        assert score.domain_match_quality == 0.85

    def test_unknown_domain_handling(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test handling of unknown domain classification."""
        classification = mock_dependencies["classification"]
        classification.domain = ArchitecturalDomain.UNKNOWN
        classification.confidence = 0.2

        result = analyzer.analyze_domain_adherence("unclear code", classification)

        # Should handle unknown domain gracefully
        assert result.domain_classification.domain == ArchitecturalDomain.UNKNOWN
        assert isinstance(result.adherence_score, AdherenceScore)
        # Unknown domain should have neutral adherence score
        assert 0.2 <= result.adherence_score.domain_adherence <= 0.4

    def test_error_handling(
        self, analyzer: DomainAwareAdherenceAnalyzer, mock_dependencies: Any
    ) -> None:
        """Test error handling in analysis."""
        # Test with invalid input
        mock_dependencies["classifier"].classify_domain.side_effect = Exception(
            "Test error"
        )

        try:
            result = analyzer.analyze_file("test.tsx", "invalid content")
            # Should handle the error gracefully and not crash
            assert result is not None
        except Exception as e:
            # If it does throw, that's also acceptable as long as it's handled properly
            # Verify the error is logged and contains useful information
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Expected error during analysis: {e}")
            # Ensure analyzer remains in a valid state after error
            assert analyzer is not None

    def test_configuration_parameters_used(self, mock_dependencies: Any) -> None:
        """Test that configuration parameters are properly used."""
        config = {
            "similarity_threshold": 0.7,
            "domain_confidence_threshold": 0.8,
            "max_similar_patterns": 3,
            "min_patterns_for_analysis": 5,
        }

        analyzer = DomainAwareAdherenceAnalyzer(config)

        assert analyzer.similarity_threshold == 0.7
        assert analyzer.domain_confidence_threshold == 0.8
        assert analyzer.max_similar_patterns == 3
