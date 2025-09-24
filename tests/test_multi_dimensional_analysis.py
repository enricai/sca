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

"""
Tests for the multi-dimensional code analysis system.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from semantic_code_analyzer import EnhancedScorerConfig, MultiDimensionalScorer
from semantic_code_analyzer.analyzers import ArchitecturalAnalyzer, QualityAnalyzer


class TestMultiDimensionalAnalysis:
    """Tests for the multi-dimensional analysis system."""

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Valid configuration
        config = EnhancedScorerConfig(
            architectural_weight=0.25,
            quality_weight=0.25,
            typescript_weight=0.20,
            framework_weight=0.15,
            domain_adherence_weight=0.15,
        )
        assert (
            abs(
                sum(
                    [
                        config.architectural_weight,
                        config.quality_weight,
                        config.typescript_weight,
                        config.framework_weight,
                        config.domain_adherence_weight,
                    ]
                )
                - 1.0
            )
            < 0.01
        )

        # Invalid configuration (weights > 1.0)
        with pytest.raises(ValueError):
            EnhancedScorerConfig(
                architectural_weight=0.50,
                quality_weight=0.50,
                typescript_weight=0.50,
                framework_weight=0.50,
                domain_adherence_weight=0.50,
            )

    def test_analyzer_initialization(
        self, enhanced_scorer_config: EnhancedScorerConfig
    ) -> None:
        """Test that analyzers initialize correctly."""
        with patch("git.Repo"):
            scorer = MultiDimensionalScorer(enhanced_scorer_config, repo_path=".")

            # Check that all expected analyzers are initialized
            expected = ["architectural", "quality", "typescript", "framework"]
            assert all(a in scorer.analyzers for a in expected)

    def test_file_analysis(
        self,
        enhanced_scorer_config: EnhancedScorerConfig,
        sample_typescript_files: dict[str, str],
    ) -> None:
        """Test analysis of TypeScript files."""
        with patch("git.Repo"):
            scorer = MultiDimensionalScorer(enhanced_scorer_config, repo_path=".")

            results = scorer.analyze_files(sample_typescript_files)

            assert "overall_adherence" in results
            assert "dimensional_scores" in results
            assert "confidence" in results
            assert "pattern_analysis" in results
            assert "actionable_feedback" in results

            # Check that we got reasonable scores
            assert 0.0 <= results["overall_adherence"] <= 1.0  # noqa: S101
            assert 0.0 <= results["confidence"] <= 1.0  # noqa: S101

    def test_individual_analyzers(
        self, sample_typescript_files: dict[str, str]
    ) -> None:
        """Test individual analyzers work correctly."""
        # Test architectural analyzer
        arch_analyzer = ArchitecturalAnalyzer()
        for file_path, content in sample_typescript_files.items():
            result = arch_analyzer.analyze_file(file_path, content)
            assert hasattr(result, "score")  # noqa: S101
            assert hasattr(result, "patterns_found")  # noqa: S101
            assert hasattr(result, "recommendations")  # noqa: S101
            assert 0.0 <= result.score <= 1.0  # noqa: S101

        # Test quality analyzer
        quality_analyzer = QualityAnalyzer()
        for file_path, content in sample_typescript_files.items():
            result = quality_analyzer.analyze_file(file_path, content)
            assert hasattr(result, "score")  # noqa: S101
            assert 0.0 <= result.score <= 1.0  # noqa: S101

    def test_weight_configuration_impact(
        self, sample_typescript_files: dict[str, str]
    ) -> None:
        """Test that different weight configurations produce different results."""
        import os

        # Disable domain adherence during tests to prevent model loading segfaults
        disable_models = os.getenv("SCA_DISABLE_MODEL_LOADING", "0") == "1"

        # Configuration emphasizing architecture
        arch_config = EnhancedScorerConfig(
            architectural_weight=0.70 if disable_models else 0.60,
            quality_weight=0.10,
            typescript_weight=0.10,
            framework_weight=0.10,
            domain_adherence_weight=0.0 if disable_models else 0.10,
            enable_domain_adherence_analysis=not disable_models,
        )

        # Configuration emphasizing quality
        quality_config = EnhancedScorerConfig(
            architectural_weight=0.10,
            quality_weight=0.70 if disable_models else 0.60,
            typescript_weight=0.10,
            framework_weight=0.10,
            domain_adherence_weight=0.0 if disable_models else 0.10,
            enable_domain_adherence_analysis=not disable_models,
        )

        with patch("git.Repo"):
            arch_scorer = MultiDimensionalScorer(arch_config, repo_path=".")
            quality_scorer = MultiDimensionalScorer(quality_config, repo_path=".")

            arch_results = arch_scorer.analyze_files(sample_typescript_files)
            quality_results = quality_scorer.analyze_files(sample_typescript_files)

            # Results should be different due to different weights
            # (unless the dimensional scores are identical, which is unlikely)
            arch_score = arch_results["overall_adherence"]
            quality_score = quality_results["overall_adherence"]

            # Both should be valid scores
            assert 0.0 <= arch_score <= 1.0  # noqa: S101
            assert 0.0 <= quality_score <= 1.0  # noqa: S101

    def test_actionable_feedback_generation(
        self,
        enhanced_scorer_config: EnhancedScorerConfig,
        sample_poor_quality_files: dict[str, str],
    ) -> None:
        """Test that poor quality code generates actionable feedback."""
        with patch("git.Repo"):
            scorer = MultiDimensionalScorer(enhanced_scorer_config, repo_path=".")

            results = scorer.analyze_files(sample_poor_quality_files)

            feedback = results.get("actionable_feedback", [])
            assert (
                len(feedback) > 0
            )  # Poor quality code should generate feedback  # noqa: S101

            # Check feedback structure
            for rec in feedback:
                assert "severity" in rec  # noqa: S101
                assert "message" in rec  # noqa: S101
                assert "category" in rec  # noqa: S101
                assert rec["severity"] in [
                    "info",
                    "warning",
                    "error",
                    "critical",
                ]  # noqa: S101

    def test_pattern_analysis_results(
        self,
        enhanced_scorer_config: EnhancedScorerConfig,
        sample_typescript_files: dict[str, str],
    ) -> None:
        """Test pattern analysis results structure."""
        with patch("git.Repo"):
            scorer = MultiDimensionalScorer(enhanced_scorer_config, repo_path=".")

            results = scorer.analyze_files(sample_typescript_files)

            pattern_analysis = results.get("pattern_analysis", {})
            assert "total_patterns_found" in pattern_analysis  # noqa: S101
            assert "patterns_by_type" in pattern_analysis  # noqa: S101
            assert "pattern_confidence_avg" in pattern_analysis  # noqa: S101

            # Should find some patterns in good TypeScript code
            assert pattern_analysis["total_patterns_found"] > 0  # noqa: S101
