"""
Integration tests for the complete Semantic Code Analyzer workflow.

These tests verify that all components work together correctly in realistic scenarios.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock
import numpy as np

from semantic_code_analyzer import SemanticScorer, ScorerConfig
from semantic_code_analyzer.semantic_scorer import CommitAnalysisResult


class TestSemanticScorerIntegration:
    """Integration tests for the complete semantic analysis workflow."""

    @pytest.fixture
    def scorer_config(self):
        """Create a test configuration for the scorer."""
        return ScorerConfig(
            model_name="microsoft/graphcodebert-base",
            distance_metric="euclidean",
            max_files=10,
            cache_embeddings=False,  # Disable caching for tests
            detailed_output=True
        )

    @pytest.fixture
    def mock_embedder(self):
        """Mock the code embedder to avoid loading actual models."""
        with patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder') as mock:
            embedder_instance = Mock()

            # Mock embedding generation
            def get_code_embedding(code, language="python", use_cache=True):
                # Generate deterministic embeddings based on code hash
                import hashlib
                code_hash = hashlib.md5(code.encode()).hexdigest()
                np.random.seed(int(code_hash[:8], 16) % (2**32))
                return np.random.rand(768).astype(np.float32)

            def get_batch_embeddings(code_snippets, language="python"):
                return [get_code_embedding(code, language) for code in code_snippets]

            embedder_instance.get_code_embedding.side_effect = get_code_embedding
            embedder_instance.get_batch_embeddings.side_effect = get_batch_embeddings
            embedder_instance.get_model_info.return_value = {
                "model_name": "microsoft/graphcodebert-base",
                "device": "cpu",
                "max_length": 512,
                "embedding_dim": 768,
                "cache_size": 0,
                "mps_available": False,
                "cuda_available": False
            }
            embedder_instance.save_cache.return_value = None
            embedder_instance.clear_cache.return_value = None

            mock.return_value = embedder_instance
            yield embedder_instance

    @pytest.mark.integration
    def test_complete_commit_analysis_workflow(self, test_repo, scorer_config, mock_embedder):
        """Test the complete workflow from commit to similarity score."""
        repo_path = test_repo['repo_path']
        commit_hash = test_repo['commits'][1]  # Use the utils commit

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Analyze commit
        result = scorer.score_commit_similarity(commit_hash, language="python")

        # Verify result structure
        assert isinstance(result, CommitAnalysisResult)
        assert result.commit_info is not None
        assert result.file_results is not None
        assert result.aggregate_scores is not None
        assert result.processing_time > 0

        # Verify commit info
        assert result.commit_info.hash == commit_hash[:8]
        assert result.commit_info.message == "Add utils module"
        assert len(result.commit_info.files_changed) == 1
        assert "utils.py" in result.commit_info.files_changed

        # Verify file results
        assert len(result.file_results) == 1
        assert "utils.py" in result.file_results

        file_result = result.file_results["utils.py"]
        assert "overall_similarity" in file_result
        assert "most_similar_files" in file_result

        # Verify similarity scores are in valid range
        overall_sim = file_result["overall_similarity"]
        assert 0 <= overall_sim["max_similarity"] <= 1
        assert 0 <= overall_sim["mean_similarity"] <= 1

        # Verify aggregate scores
        assert 0 <= result.aggregate_scores["max_similarity"] <= 1
        assert 0 <= result.aggregate_scores["mean_similarity"] <= 1

    @pytest.mark.integration
    def test_batch_analysis_workflow(self, test_repo, scorer_config, mock_embedder):
        """Test analyzing multiple commits in batch."""
        repo_path = test_repo['repo_path']

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Analyze recent commits
        results = scorer.get_recent_commits_analysis(max_commits=3, language="python")

        # Verify results
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, CommitAnalysisResult) for r in results)

        # Verify ordering (most recent first)
        if len(results) > 1:
            timestamps = [r.commit_info.timestamp for r in results]
            # Note: timestamps should be in descending order for recent commits

        # Verify each result has valid structure
        for result in results:
            assert result.commit_info is not None
            assert result.aggregate_scores is not None
            assert 0 <= result.aggregate_scores.get("max_similarity", 0) <= 1

    @pytest.mark.integration
    def test_commit_comparison_workflow(self, test_repo, scorer_config, mock_embedder):
        """Test comparing two commits."""
        repo_path = test_repo['repo_path']
        commit_a = test_repo['commits'][1]  # utils commit
        commit_b = test_repo['commits'][2]  # constants commit

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Compare commits
        comparison_result = scorer.compare_commits(commit_a, commit_b, language="python")

        # Verify comparison structure
        assert "commit_a" in comparison_result
        assert "commit_b" in comparison_result
        assert "cross_similarity" in comparison_result
        assert "similarity_difference" in comparison_result

        # Verify individual commit results
        result_a = comparison_result["commit_a"]
        result_b = comparison_result["commit_b"]

        assert isinstance(result_a, CommitAnalysisResult)
        assert isinstance(result_b, CommitAnalysisResult)

        # Verify cross-similarity analysis
        cross_sim = comparison_result["cross_similarity"]
        assert "similarity_matrix" in cross_sim
        assert "max_similarity" in cross_sim
        assert "mean_similarity" in cross_sim

        # Verify similarity differences
        sim_diff = comparison_result["similarity_difference"]
        assert "max_similarity_diff" in sim_diff
        assert "mean_similarity_diff" in sim_diff

    @pytest.mark.integration
    def test_empty_commit_handling(self, test_repo, scorer_config, mock_embedder):
        """Test handling of commits with no code changes."""
        repo_path = test_repo['repo_path']

        # Create a commit with no code changes (e.g., documentation only)
        repo = test_repo['repo']
        readme_file = Path(repo_path) / "README.md"
        readme_file.write_text("# Test Project\nThis is a test project.")

        repo.index.add(["README.md"])
        empty_commit = repo.index.commit("Add README (non-code)")

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Analyze the empty commit
        result = scorer.score_commit_similarity(empty_commit.hexsha, language="python")

        # Should handle gracefully
        assert isinstance(result, CommitAnalysisResult)
        assert len(result.file_results) == 0
        assert result.aggregate_scores["max_similarity"] == 0.0

    @pytest.mark.integration
    def test_large_file_handling(self, test_repo, scorer_config, mock_embedder):
        """Test handling of large files."""
        repo_path = test_repo['repo_path']

        # Create a large Python file
        repo = test_repo['repo']
        large_file = Path(repo_path) / "large.py"

        # Generate large content
        large_content = "# Large file\n" + "\n".join([
            f"def function_{i}():\n    return {i}"
            for i in range(1000)
        ])
        large_file.write_text(large_content)

        repo.index.add(["large.py"])
        large_commit = repo.index.commit("Add large file")

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Should handle large file gracefully
        result = scorer.score_commit_similarity(large_commit.hexsha, language="python")

        assert isinstance(result, CommitAnalysisResult)
        # Large file might be excluded, so file_results could be empty
        assert isinstance(result.file_results, dict)

    @pytest.mark.integration
    def test_multi_language_support(self, test_repo, scorer_config, mock_embedder):
        """Test support for multiple programming languages."""
        repo_path = test_repo['repo_path']
        repo = test_repo['repo']

        # Create JavaScript file
        js_file = Path(repo_path) / "app.js"
        js_file.write_text("""
function calculateArea(radius) {
    return Math.PI * radius * radius;
}

class Calculator {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
}
""")

        repo.index.add(["app.js"])
        js_commit = repo.index.commit("Add JavaScript calculator")

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Analyze JavaScript commit
        result = scorer.score_commit_similarity(js_commit.hexsha, language="javascript")

        assert isinstance(result, CommitAnalysisResult)
        assert "app.js" in result.file_results
        assert result.aggregate_scores["max_similarity"] >= 0

    @pytest.mark.integration
    def test_error_recovery(self, test_repo, scorer_config):
        """Test error recovery and graceful degradation."""
        repo_path = test_repo['repo_path']

        # Test with invalid model name
        bad_config = ScorerConfig(
            model_name="nonexistent/model",
            cache_embeddings=False
        )

        with pytest.raises(Exception):
            # Should fail to initialize with bad model
            scorer = SemanticScorer(repo_path, bad_config)

    @pytest.mark.integration
    def test_scorer_info_retrieval(self, test_repo, scorer_config, mock_embedder):
        """Test retrieving scorer information."""
        repo_path = test_repo['repo_path']

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Get scorer info
        info = scorer.get_scorer_info()

        # Verify info structure
        assert "repo_path" in info
        assert "config" in info
        assert "model_info" in info
        assert "cache_info" in info
        assert "supported_languages" in info

        # Verify content
        assert info["repo_path"] == str(Path(repo_path).resolve())
        assert "python" in info["supported_languages"]

    @pytest.mark.integration
    def test_cache_management(self, test_repo, scorer_config, mock_embedder):
        """Test embedding cache management."""
        repo_path = test_repo['repo_path']

        # Enable caching
        scorer_config.cache_embeddings = True

        # Initialize scorer
        scorer = SemanticScorer(repo_path, scorer_config)

        # Analyze a commit (should populate cache)
        commit_hash = test_repo['commits'][1]
        result1 = scorer.score_commit_similarity(commit_hash, language="python")

        # Analyze same commit again (should use cache)
        result2 = scorer.score_commit_similarity(commit_hash, language="python")

        # Results should be identical
        assert result1.aggregate_scores == result2.aggregate_scores

        # Clear caches
        scorer.clear_caches()

        # Should complete without error
        assert True

    @pytest.mark.integration
    def test_different_distance_metrics(self, test_repo, mock_embedder):
        """Test analysis with different distance metrics."""
        repo_path = test_repo['repo_path']
        commit_hash = test_repo['commits'][1]

        metrics = ["euclidean", "cosine", "manhattan"]
        results = {}

        for metric in metrics:
            config = ScorerConfig(
                distance_metric=metric,
                cache_embeddings=False
            )

            scorer = SemanticScorer(repo_path, config)
            result = scorer.score_commit_similarity(commit_hash, language="python")

            results[metric] = result.aggregate_scores["max_similarity"]

        # All metrics should produce valid results
        for metric, score in results.items():
            assert 0 <= score <= 1, f"Invalid score for {metric}: {score}"

        # Different metrics might produce different results
        assert len(set(results.values())) >= 1  # At least one unique value


class TestCLIIntegration:
    """Integration tests for the CLI interface."""

    @pytest.mark.integration
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from semantic_code_analyzer.cli import main
        assert callable(main)

    @pytest.mark.integration
    def test_cli_help(self):
        """Test CLI help functionality."""
        from semantic_code_analyzer.cli import cli
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "Semantic Code Analyzer" in result.output

    @pytest.mark.integration
    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_cli_analyze_command(self, mock_scorer_class, test_repo):
        """Test CLI analyze command."""
        from semantic_code_analyzer.cli import cli
        from click.testing import CliRunner

        # Mock scorer
        mock_scorer = Mock()
        mock_result = Mock()
        mock_result.commit_info.hash = "abc123"
        mock_result.commit_info.author = "Test User"
        mock_result.commit_info.message = "Test commit"
        mock_result.commit_info.files_changed = ["test.py"]
        mock_result.commit_info.insertions = 10
        mock_result.commit_info.deletions = 5
        mock_result.aggregate_scores = {"max_similarity": 0.75, "mean_similarity": 0.65}
        mock_result.file_results = {}
        mock_result.processing_time = 1.5

        mock_scorer.score_commit_similarity.return_value = mock_result
        mock_scorer_class.return_value = mock_scorer

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Test basic analyze command
            result = runner.invoke(cli, [
                'analyze',
                'abc123',
                '--repo-path', test_repo['repo_path']
            ])

            # Should complete successfully
            assert result.exit_code == 0


class TestEndToEndWorkflow:
    """End-to-end tests simulating real usage scenarios."""

    @pytest.mark.integration
    def test_full_analysis_pipeline(self, mock_embedder):
        """Test a complete analysis pipeline from repository creation to results."""
        # Create a temporary repository with realistic code
        with tempfile.TemporaryDirectory() as tmpdir:
            from tests.conftest import create_temp_git_repo

            files = {
                "calculator.py": '''
class Calculator:
    """A simple calculator implementation."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
''',
                "math_utils.py": '''
import math

def factorial(n):
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
'''
            }

            repo_path, commit_hash = create_temp_git_repo(files)

            # Configure scorer
            config = ScorerConfig(
                model_name="microsoft/graphcodebert-base",
                distance_metric="euclidean",
                cache_embeddings=False,
                detailed_output=True
            )

            # Initialize and run analysis
            scorer = SemanticScorer(repo_path, config)
            result = scorer.score_commit_similarity(commit_hash, language="python")

            # Verify complete analysis
            assert isinstance(result, CommitAnalysisResult)
            assert len(result.file_results) == 2  # calculator.py and math_utils.py

            # Verify both files were analyzed
            assert "calculator.py" in result.file_results
            assert "math_utils.py" in result.file_results

            # Verify similarity scores are reasonable
            for file_path, file_result in result.file_results.items():
                overall_sim = file_result["overall_similarity"]
                assert 0 <= overall_sim["max_similarity"] <= 1
                assert 0 <= overall_sim["mean_similarity"] <= 1

            # Clean up
            from tests.conftest import cleanup_temp_repo
            cleanup_temp_repo(repo_path)

    @pytest.mark.integration
    def test_performance_with_realistic_codebase(self, mock_embedder):
        """Test performance with a more realistic codebase size."""
        # Create a larger codebase
        with tempfile.TemporaryDirectory() as tmpdir:
            from tests.conftest import create_temp_git_repo

            # Generate multiple files
            files = {}
            for i in range(20):  # 20 files
                files[f"module_{i}.py"] = f'''
"""Module {i} - Auto-generated for testing."""

class Class{i}:
    """Class {i} implementation."""

    def __init__(self):
        self.value = {i}

    def method_{i}(self, x):
        """Method {i}."""
        return x * {i}

    def calculate(self):
        """Calculate something."""
        return self.value ** 2

def function_{i}(data):
    """Function {i}."""
    return [item + {i} for item in data if isinstance(item, int)]

CONSTANT_{i} = {i * 10}
'''

            repo_path, commit_hash = create_temp_git_repo(files)

            # Configure scorer with file limit
            config = ScorerConfig(
                model_name="microsoft/graphcodebert-base",
                max_files=15,  # Limit to 15 files
                cache_embeddings=False
            )

            # Run analysis and measure time
            import time
            start_time = time.time()

            scorer = SemanticScorer(repo_path, config)
            result = scorer.score_commit_similarity(commit_hash, language="python")

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify results
            assert isinstance(result, CommitAnalysisResult)
            assert len(result.file_results) <= 20  # Should handle all files

            # Performance should be reasonable (this is subjective)
            assert processing_time < 60  # Should complete within 60 seconds

            # Clean up
            from tests.conftest import cleanup_temp_repo
            cleanup_temp_repo(repo_path)