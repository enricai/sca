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

"""Integration tests for domain-aware adherence measurement system."""

from __future__ import annotations

import tempfile
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from semantic_code_analyzer import EnhancedScorerConfig, MultiDimensionalScorer
from semantic_code_analyzer.analyzers.domain_adherence_analyzer import (
    DomainAwareAdherenceAnalyzer,
)
from semantic_code_analyzer.analyzers.domain_classifier import ArchitecturalDomain


class TestDomainAwareIntegration:
    """Integration tests for the complete domain-aware adherence measurement system."""

    @pytest.fixture
    def mock_git_repo(self) -> Any:
        """Mock git repository for testing."""
        with patch("git.Repo") as mock_repo_class:
            mock_repo = Mock()
            mock_commit = Mock()
            mock_tree = Mock()

            # Mock file tree structure
            mock_files = [
                Mock(type="blob", path="src/components/Button.tsx", data_stream=Mock()),
                Mock(
                    type="blob", path="src/app/api/users/route.ts", data_stream=Mock()
                ),
                Mock(type="blob", path="tests/Button.test.tsx", data_stream=Mock()),
                Mock(
                    type="blob",
                    path="database/migrations/001_users.sql",
                    data_stream=Mock(),
                ),
            ]

            # Mock file contents
            file_contents = {
                "src/components/Button.tsx": """
import React from 'react';

interface ButtonProps {
    title: string;
    onClick: () => void;
}

const Button: React.FC<ButtonProps> = ({ title, onClick }) => {
    return <button onClick={onClick}>{title}</button>;
};

export default Button;
""",
                "src/app/api/users/route.ts": """
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    return NextResponse.json({ users: [] });
}
""",
                "tests/Button.test.tsx": """
import { render, screen } from '@testing-library/react';
import Button from '../src/components/Button';

test('renders button', () => {
    render(<Button title="Test" onClick={() => {}} />);
    expect(screen.getByText('Test')).toBeInTheDocument();
});
""",
                "database/migrations/001_users.sql": """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL
);
""",
            }

            for mock_file in mock_files:
                content = file_contents.get(mock_file.path, "")
                mock_file.data_stream.read.return_value.decode.return_value = content

            mock_tree.traverse.return_value = mock_files
            mock_commit.tree = mock_tree

            # Mock diff for commit analysis
            mock_diff_item = Mock()
            mock_diff_item.change_type = "A"  # Added file
            mock_diff_item.b_path = "src/components/NewButton.tsx"

            # Mock parent commit with hexsha
            mock_parent = Mock()
            mock_parent.hexsha = "parent123abc"
            mock_parent.tree = mock_tree  # Parent has same tree for pattern building
            mock_parent.diff.return_value = [mock_diff_item]

            mock_commit.parents = [mock_parent]  # Has parent commit
            mock_commit.hexsha = "abc123def"  # pragma: allowlist secret

            # Mock commit lookup to return appropriate commits
            def mock_commit_lookup(ref: str) -> Mock:
                if ref == "parent123abc":
                    return mock_parent
                else:
                    return mock_commit

            mock_repo.commit.side_effect = mock_commit_lookup

            # Mock commit tree access for file content
            def mock_tree_access(self: Any, path: str) -> Any:
                mock_blob = Mock()
                content = """
import React from 'react';

const NewButton: React.FC = () => {
    return <button>New Button</button>;
};

export default NewButton;
"""
                mock_blob.data_stream.read.return_value.decode.return_value = content
                return mock_blob

            mock_commit.tree.__truediv__ = mock_tree_access

            mock_repo_class.return_value = mock_repo
            yield mock_repo

    @pytest.fixture
    def mock_models(self) -> Any:
        """Mock the ML models to avoid loading actual models."""
        with (
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer"),
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel"),
            patch(
                "semantic_code_analyzer.embeddings.pattern_indexer.torch"
            ) as mock_torch,
        ):
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__.return_value = None
            mock_torch.no_grad.return_value.__exit__.return_value = None

            yield

    @pytest.fixture
    def domain_aware_config(self) -> EnhancedScorerConfig:
        """Create configuration with domain-aware analysis enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return EnhancedScorerConfig(
                architectural_weight=0.20,
                quality_weight=0.20,
                typescript_weight=0.20,
                framework_weight=0.20,
                domain_adherence_weight=0.20,
                enable_domain_adherence_analysis=True,
                similarity_threshold=0.3,
                max_similar_patterns=5,
                build_pattern_indices=True,
                cache_dir=temp_dir,
                include_actionable_feedback=True,
                max_recommendations_per_file=5,
            )

    def test_domain_aware_scorer_initialization(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test that MultiDimensionalScorer initializes with domain-aware analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Should have domain adherence analyzer
            assert "domain_adherence" in scorer.analyzers
            assert (
                len(scorer.analyzers) == 5
            )  # All analyzers including domain adherence

            # Configuration should be properly set
            assert scorer.config.enable_domain_adherence_analysis is True
            assert scorer.config.domain_adherence_weight == 0.20

    def test_full_analysis_workflow(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test complete analysis workflow with domain-aware analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Analyze a commit
            results = scorer.analyze_commit("abc123")

            # Verify results structure
            assert "overall_adherence" in results
            assert "dimensional_scores" in results
            assert "domain_adherence" in results["dimensional_scores"]
            assert "config" in results
            assert "domain_adherence" in results["config"]["weights"]

            # Should have analysis metadata
            assert "analysis_metadata" in results
            assert "domain_adherence" in results["analysis_metadata"]["analyzers_used"]

    def test_pattern_index_building(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test that pattern indices are built from codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Build pattern indices from parent commit
            scorer.build_pattern_indices_from_codebase("parent123abc")

            # Should have built indices for the domain adherence analyzer
            domain_analyzer = cast(
                DomainAwareAdherenceAnalyzer, scorer.analyzers["domain_adherence"]
            )
            assert hasattr(domain_analyzer, "_indices_built")

    def test_domain_classification_integration(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test domain classification integration in full workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Get domain adherence analyzer
            domain_analyzer = cast(
                DomainAwareAdherenceAnalyzer, scorer.analyzers["domain_adherence"]
            )

            # Test domain classification for different file types
            frontend_code = "import React from 'react'; const Button = () => <button>Click</button>;"
            backend_code = "export async function GET() { return Response.json({}); }"
            test_code = "test('works', () => { expect(true).toBe(true); });"

            frontend_result = domain_analyzer.domain_classifier.classify_domain(
                "src/Button.tsx", frontend_code
            )
            backend_result = domain_analyzer.domain_classifier.classify_domain(
                "src/api/route.ts", backend_code
            )
            test_result = domain_analyzer.domain_classifier.classify_domain(
                "tests/test.tsx", test_code
            )

            # Should classify into appropriate domains
            assert frontend_result.domain == ArchitecturalDomain.FRONTEND
            assert backend_result.domain == ArchitecturalDomain.BACKEND
            assert test_result.domain == ArchitecturalDomain.TESTING

    def test_similarity_search_integration(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test similarity search integration in analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)
            domain_analyzer = cast(
                DomainAwareAdherenceAnalyzer, scorer.analyzers["domain_adherence"]
            )

            # Mock some built indices
            domain_analyzer._indices_built.add("frontend")

            # Test similarity search
            query_code = "const TestButton = () => <button>Test</button>;"
            matches = domain_analyzer.pattern_indexer.search_similar_patterns(
                query_code, "frontend", top_k=3
            )

            # Should return a list (may be empty with mocked components)
            assert isinstance(matches, list)

    def test_end_to_end_recommendations(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test end-to-end recommendation generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Analyze a commit
            results = scorer.analyze_commit("abc123")

            # Should have actionable feedback
            assert "actionable_feedback" in results
            feedback = results["actionable_feedback"]
            assert isinstance(feedback, list)

            # Should have file-level analysis
            assert "file_level_analysis" in results

    def test_configuration_validation(
        self, mock_git_repo: Any, mock_models: Any
    ) -> None:
        """Test configuration validation for domain-aware analysis."""
        # Test invalid weights (sum > 1.0)
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EnhancedScorerConfig(
                architectural_weight=0.3,
                quality_weight=0.3,
                typescript_weight=0.3,
                framework_weight=0.3,
                domain_adherence_weight=0.3,  # Total = 1.5, should fail
            )

        # Test valid configuration
        config = EnhancedScorerConfig(
            architectural_weight=0.2,
            quality_weight=0.2,
            typescript_weight=0.2,
            framework_weight=0.2,
            domain_adherence_weight=0.2,  # Total = 1.0, should pass
        )
        assert config.domain_adherence_weight == 0.2

    def test_analyzer_interaction(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test interaction between different analyzers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # All analyzers should be enabled
            expected_analyzers = [
                "architectural",
                "quality",
                "typescript",
                "framework",
                "domain_adherence",
            ]
            assert set(scorer.analyzers.keys()) == set(expected_analyzers)

            # Each analyzer should have proper configuration
            for _analyzer_name, analyzer in scorer.analyzers.items():
                assert hasattr(analyzer, "get_analyzer_name")
                assert hasattr(analyzer, "get_weight")
                assert hasattr(analyzer, "analyze_file")

    def test_results_aggregation_with_domain_adherence(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test that results properly aggregate domain adherence scores."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Mock analysis results with proper AnalysisResult mock
            from semantic_code_analyzer.analyzers.base_analyzer import AnalysisResult

            def create_mock_result(
                score: float, domain: str = "frontend"
            ) -> AnalysisResult:
                mock_result = Mock(spec=AnalysisResult)
                mock_result.score = score
                mock_result.metrics = {"domain": domain}
                return mock_result

            mock_results = {
                "architectural": {"file1.tsx": create_mock_result(0.8)},
                "quality": {"file1.tsx": create_mock_result(0.7)},
                "typescript": {"file1.tsx": create_mock_result(0.9)},
                "framework": {"file1.tsx": create_mock_result(0.6)},
                "domain_adherence": {"file1.tsx": create_mock_result(0.75)},
            }

            # Test aggregation
            aggregated = scorer._aggregate_results(mock_results)

            # Should include domain adherence in dimensional scores
            assert "domain_adherence" in aggregated.dimensional_scores
            assert aggregated.dimensional_scores["domain_adherence"] == 0.75

            # Overall score should incorporate all dimensions
            assert 0 <= aggregated.overall_score <= 1

    def test_disabled_domain_adherence(
        self, mock_git_repo: Any, mock_models: Any
    ) -> None:
        """Test behavior when domain adherence analysis is disabled."""
        config = EnhancedScorerConfig(
            enable_domain_adherence_analysis=False,
            architectural_weight=0.25,
            quality_weight=0.25,
            typescript_weight=0.25,
            framework_weight=0.25,
            domain_adherence_weight=0.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(config, temp_dir)

            # Should not have domain adherence analyzer
            assert "domain_adherence" not in scorer.analyzers
            assert len(scorer.analyzers) == 4  # Only the original 4 analyzers

    def test_error_resilience(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test that system is resilient to errors in domain-aware analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(domain_aware_config, temp_dir)

            # Mock domain analyzer to raise exception
            domain_analyzer = cast(
                DomainAwareAdherenceAnalyzer, scorer.analyzers["domain_adherence"]
            )
            with patch.object(
                domain_analyzer, "analyze_file", side_effect=Exception("Test error")
            ):
                # Analysis should continue even if domain analyzer fails
                results = scorer.analyze_commit("abc123")

                # Should still have results from other analyzers
                assert "overall_adherence" in results
                assert "dimensional_scores" in results

    def test_performance_with_large_codebase(
        self,
        domain_aware_config: EnhancedScorerConfig,
        mock_git_repo: Any,
        mock_models: Any,
    ) -> None:
        """Test performance considerations with larger codebases."""
        # Create config with limits for performance
        config = EnhancedScorerConfig(
            enable_domain_adherence_analysis=True,
            domain_adherence_weight=0.2,
            framework_weight=0.1,  # Reduced to maintain weight sum of 1.0
            max_similar_patterns=3,  # Limit patterns for performance
            build_pattern_indices=False,  # Disable for performance test
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scorer = MultiDimensionalScorer(config, temp_dir)

            # Should initialize without issues
            assert "domain_adherence" in scorer.analyzers

            # Configuration should respect performance settings
            domain_analyzer = cast(
                DomainAwareAdherenceAnalyzer, scorer.analyzers["domain_adherence"]
            )
            assert domain_analyzer.max_similar_patterns == 3
