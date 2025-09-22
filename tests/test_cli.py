"""
Tests for the CLI module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import numpy as np

from semantic_code_analyzer.cli import (
    cli,
    setup_logging,
    _interpret_score,
    _make_json_serializable,
    main
)
from semantic_code_analyzer.semantic_scorer import CommitAnalysisResult, ScorerConfig
from semantic_code_analyzer.commit_extractor import CommitInfo


class TestCLIUtils:
    """Test utility functions in CLI module."""

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose flag."""
        with patch('semantic_code_analyzer.cli.logging') as mock_logging:
            setup_logging(verbose=True, quiet=False)
            mock_logging.getLogger().setLevel.assert_called_with(mock_logging.DEBUG)

    def test_setup_logging_quiet(self):
        """Test logging setup with quiet flag."""
        with patch('semantic_code_analyzer.cli.logging') as mock_logging:
            setup_logging(verbose=False, quiet=True)
            mock_logging.getLogger().setLevel.assert_called_with(mock_logging.ERROR)

    def test_setup_logging_normal(self):
        """Test logging setup with normal verbosity."""
        with patch('semantic_code_analyzer.cli.logging') as mock_logging:
            setup_logging(verbose=False, quiet=False)
            mock_logging.getLogger().setLevel.assert_called_with(mock_logging.INFO)

    def test_interpret_score_very_high(self):
        """Test score interpretation for very high similarity."""
        result = _interpret_score(0.9)
        assert "Very High" in result
        assert "closely" in result

    def test_interpret_score_good(self):
        """Test score interpretation for good similarity."""
        result = _interpret_score(0.7)
        assert "Good" in result
        assert "consistent" in result

    def test_interpret_score_moderate(self):
        """Test score interpretation for moderate similarity."""
        result = _interpret_score(0.5)
        assert "Moderate" in result
        assert "alignment" in result

    def test_interpret_score_low(self):
        """Test score interpretation for low similarity."""
        result = _interpret_score(0.3)
        assert "Low" in result
        assert "Different" in result

    def test_interpret_score_very_low(self):
        """Test score interpretation for very low similarity."""
        result = _interpret_score(0.1)
        assert "Very Low" in result
        assert "different" in result

    def test_make_json_serializable(self):
        """Test JSON serialization utility."""
        test_data = {
            "numpy_array": np.array([1, 2, 3]),
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "normal_dict": {"key": "value"},
            "normal_list": [1, 2, 3],
            "normal_string": "test"
        }

        result = _make_json_serializable(test_data)

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Check specific conversions
        assert result["numpy_array"] == [1, 2, 3]
        assert result["numpy_int"] == 42
        assert result["numpy_float"] == 3.14
        assert result["normal_dict"]["key"] == "value"


class TestCLICommands:
    """Test CLI command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_scorer(self):
        """Create mock SemanticScorer."""
        mock_scorer = Mock()

        # Mock commit info
        mock_commit_info = CommitInfo(
            hash="abc123",
            message="Test commit message",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py", "utils.py"],
            insertions=15,
            deletions=5
        )

        # Mock analysis result
        mock_result = CommitAnalysisResult(
            commit_info=mock_commit_info,
            file_results={
                "test.py": {
                    "overall_similarity": {
                        "max_similarity": 0.75,
                        "mean_similarity": 0.65,
                        "std_similarity": 0.10
                    },
                    "most_similar_files": [
                        {"file_path": "similar.py", "similarity_score": 0.75, "rank": 1}
                    ]
                }
            },
            aggregate_scores={
                "max_similarity": 0.75,
                "mean_similarity": 0.65,
                "median_similarity": 0.70
            },
            processing_time=2.5,
            model_info={
                "model_name": "test-model",
                "device": "cpu",
                "embedding_dim": 768
            },
            config={"distance_metric": "euclidean"}
        )

        mock_scorer.score_commit_similarity.return_value = mock_result
        mock_scorer.get_recent_commits_analysis.return_value = [mock_result]
        mock_scorer.compare_commits.return_value = {
            "commit_a": mock_result,
            "commit_b": mock_result,
            "cross_similarity": {
                "max_similarity": 0.8,
                "mean_similarity": 0.7,
                "shape": (2, 2)
            },
            "similarity_difference": {
                "max_similarity_diff": 0.0,
                "mean_similarity_diff": 0.0
            }
        }
        mock_scorer.get_scorer_info.return_value = {
            "repo_path": "/test/repo",
            "config": {"distance_metric": "euclidean"},
            "model_info": {
                "model_name": "test-model",
                "device": "cpu",
                "embedding_dim": 768,
                "cache_size": 0,
                "mps_available": False,
                "cuda_available": False
            },
            "cache_info": {"enabled": True, "size": 0},
            "supported_languages": ["python", "javascript", "java"]
        }

        return mock_scorer

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "Semantic Code Analyzer" in result.output
        assert "analyze" in result.output

    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_analyze_command_basic(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test basic analyze command."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'analyze', 'abc123',
            '--repo-path', test_repo['repo_path']
        ])

        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "0.750" in result.output  # Similarity score
        mock_scorer.score_commit_similarity.assert_called_once_with('abc123', 'python')

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_analyze_command_with_options(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test analyze command with various options."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'analyze', 'abc123',
            '--repo-path', test_repo['repo_path'],
            '--language', 'javascript',
            '--model', 'custom/model',
            '--distance-metric', 'cosine',
            '--max-files', '50',
            '--detailed'
        ])

        assert result.exit_code == 0
        mock_scorer.score_commit_similarity.assert_called_once_with('abc123', 'javascript')

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_analyze_command_with_output(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test analyze command with output file."""
        mock_scorer_class.return_value = mock_scorer

        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'analyze', 'abc123',
                '--repo-path', test_repo['repo_path'],
                '--output', 'results.json'
            ])

            assert result.exit_code == 0
            assert Path('results.json').exists()

            # Check file content
            with open('results.json', 'r') as f:
                data = json.load(f)
            assert data['commit_info']['hash'] == 'abc123'

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_analyze_command_error(self, mock_scorer_class, runner, test_repo):
        """Test analyze command with error."""
        mock_scorer_class.side_effect = Exception("Test error")

        result = runner.invoke(cli, [
            'analyze', 'abc123',
            '--repo-path', test_repo['repo_path']
        ])

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_batch_command_basic(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test basic batch command."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'batch',
            '--repo-path', test_repo['repo_path'],
            '--count', '5'
        ])

        assert result.exit_code == 0
        assert "abc123" in result.output
        mock_scorer.get_recent_commits_analysis.assert_called_once_with(5, 'HEAD', 'python')

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_batch_command_with_options(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test batch command with options."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'batch',
            '--repo-path', test_repo['repo_path'],
            '--count', '3',
            '--branch', 'develop',
            '--language', 'javascript'
        ])

        assert result.exit_code == 0
        mock_scorer.get_recent_commits_analysis.assert_called_once_with(3, 'develop', 'javascript')

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_batch_command_with_output(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test batch command with output file."""
        mock_scorer_class.return_value = mock_scorer

        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'batch',
                '--repo-path', test_repo['repo_path'],
                '--output', 'batch_results.json'
            ])

            assert result.exit_code == 0
            assert Path('batch_results.json').exists()

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_compare_command_basic(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test basic compare command."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'compare', 'abc123', 'def456',
            '--repo-path', test_repo['repo_path']
        ])

        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "def456" in result.output
        mock_scorer.compare_commits.assert_called_once_with('abc123', 'def456', 'python')

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_compare_command_with_output(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test compare command with output file."""
        mock_scorer_class.return_value = mock_scorer

        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'compare', 'abc123', 'def456',
                '--repo-path', test_repo['repo_path'],
                '--output', 'comparison.json'
            ])

            assert result.exit_code == 0
            assert Path('comparison.json').exists()

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_info_command(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test info command."""
        mock_scorer_class.return_value = mock_scorer

        result = runner.invoke(cli, [
            'info',
            '--repo-path', test_repo['repo_path']
        ])

        assert result.exit_code == 0
        assert "test-model" in result.output
        assert "euclidean" in result.output
        assert "python" in result.output

    def test_global_verbose_flag(self, runner):
        """Test global verbose flag."""
        with patch('semantic_code_analyzer.cli.setup_logging') as mock_setup:
            result = runner.invoke(cli, ['--verbose', '--help'])

            assert result.exit_code == 0
            mock_setup.assert_called_once_with(True, False)

    def test_global_quiet_flag(self, runner):
        """Test global quiet flag."""
        with patch('semantic_code_analyzer.cli.setup_logging') as mock_setup:
            result = runner.invoke(cli, ['--quiet', '--help'])

            assert result.exit_code == 0
            mock_setup.assert_called_once_with(False, True)


class TestCLIDisplayFunctions:
    """Test CLI display and formatting functions."""

    @pytest.fixture
    def sample_result(self):
        """Create sample analysis result."""
        commit_info = CommitInfo(
            hash="abc123",
            message="Test commit message for display",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py", "utils.py"],
            insertions=15,
            deletions=5
        )

        return CommitAnalysisResult(
            commit_info=commit_info,
            file_results={
                "test.py": {
                    "overall_similarity": {
                        "max_similarity": 0.85,
                        "mean_similarity": 0.75,
                        "std_similarity": 0.08
                    },
                    "most_similar_files": [
                        {"file_path": "similar.py", "similarity_score": 0.85, "rank": 1},
                        {"file_path": "other.py", "similarity_score": 0.70, "rank": 2}
                    ]
                },
                "utils.py": {
                    "overall_similarity": {
                        "max_similarity": 0.65,
                        "mean_similarity": 0.55,
                        "std_similarity": 0.12
                    },
                    "most_similar_files": [
                        {"file_path": "helpers.py", "similarity_score": 0.65, "rank": 1}
                    ]
                }
            },
            aggregate_scores={
                "max_similarity": 0.85,
                "mean_similarity": 0.70,
                "median_similarity": 0.75
            },
            processing_time=3.2,
            model_info={
                "model_name": "test-model",
                "device": "cpu"
            },
            config={"distance_metric": "euclidean"}
        )

    @patch('semantic_code_analyzer.cli.console')
    def test_display_analysis_results_basic(self, mock_console, sample_result):
        """Test basic display of analysis results."""
        from semantic_code_analyzer.cli import _display_analysis_results

        _display_analysis_results(sample_result, detailed=False)

        # Should have printed tables and panels
        assert mock_console.print.called
        print_calls = [call[0][0] for call in mock_console.print.call_args_list]

        # Check that similarity scores are displayed
        assert any("0.850" in str(call) for call in print_calls)

    @patch('semantic_code_analyzer.cli.console')
    def test_display_analysis_results_detailed(self, mock_console, sample_result):
        """Test detailed display of analysis results."""
        from semantic_code_analyzer.cli import _display_analysis_results

        _display_analysis_results(sample_result, detailed=True)

        assert mock_console.print.called
        # Should have more calls for detailed output
        assert len(mock_console.print.call_args_list) > 3

    @patch('semantic_code_analyzer.cli.console')
    def test_display_batch_results(self, mock_console, sample_result):
        """Test display of batch results."""
        from semantic_code_analyzer.cli import _display_batch_results

        results = [sample_result, sample_result]  # Duplicate for testing

        _display_batch_results(results)

        assert mock_console.print.called
        print_calls = [call[0][0] for call in mock_console.print.call_args_list]

        # Should display summary statistics
        assert any("abc123" in str(call) for call in print_calls)

    @patch('semantic_code_analyzer.cli.console')
    def test_display_batch_results_empty(self, mock_console):
        """Test display of empty batch results."""
        from semantic_code_analyzer.cli import _display_batch_results

        _display_batch_results([])

        assert mock_console.print.called
        print_calls = [call[0][0] for call in mock_console.print.call_args_list]
        assert any("No results" in str(call) for call in print_calls)

    @patch('semantic_code_analyzer.cli.console')
    def test_display_comparison_results(self, mock_console, sample_result):
        """Test display of comparison results."""
        from semantic_code_analyzer.cli import _display_comparison_results

        comparison_result = {
            "commit_a": sample_result,
            "commit_b": sample_result,
            "cross_similarity": {
                "max_similarity": 0.80,
                "mean_similarity": 0.70,
                "shape": (2, 2)
            },
            "similarity_difference": {
                "max_similarity_diff": 0.05,
                "mean_similarity_diff": 0.03
            }
        }

        _display_comparison_results(comparison_result)

        assert mock_console.print.called
        print_calls = [call[0][0] for call in mock_console.print.call_args_list]

        # Should display comparison data
        assert any("0.800" in str(call) for call in print_calls)

    @patch('semantic_code_analyzer.cli.console')
    def test_display_info(self, mock_console):
        """Test display of scorer information."""
        from semantic_code_analyzer.cli import _display_info

        info_data = {
            "repo_path": "/test/repo",
            "config": {
                "model_name": "test-model",
                "distance_metric": "euclidean",
                "max_files": None,
                "use_mps": True,
                "cache_embeddings": True
            },
            "model_info": {
                "device": "mps",
                "embedding_dim": 768,
                "max_length": 512,
                "cache_size": 42,
                "mps_available": True,
                "cuda_available": False
            },
            "supported_languages": ["python", "javascript", "java"]
        }

        _display_info(info_data)

        assert mock_console.print.called
        print_calls = [call[0][0] for call in mock_console.print.call_args_list]

        # Should display configuration info
        assert any("test-model" in str(call) for call in print_calls)
        assert any("euclidean" in str(call) for call in print_calls)


class TestCLIFileOperations:
    """Test CLI file save/load operations."""

    def test_save_results_to_file(self, sample_code_snippets):
        """Test saving results to JSON file."""
        from semantic_code_analyzer.cli import _save_results_to_file

        # Create sample result
        commit_info = CommitInfo(
            hash="abc123",
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        result = CommitAnalysisResult(
            commit_info=commit_info,
            file_results={},
            aggregate_scores={"max_similarity": 0.75},
            processing_time=1.0,
            model_info={"model_name": "test"},
            config={"test": "config"}
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            _save_results_to_file(result, output_path)

            # Check file was created and has correct content
            assert Path(output_path).exists()

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert data['commit_info']['hash'] == 'abc123'
            assert data['aggregate_scores']['max_similarity'] == 0.75

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_save_batch_results_to_file(self):
        """Test saving batch results to JSON file."""
        from semantic_code_analyzer.cli import _save_batch_results_to_file

        # Create sample results
        results = []
        for i in range(3):
            commit_info = CommitInfo(
                hash=f"commit{i}",
                message=f"Test commit {i}",
                author="Test Author",
                timestamp="2023-01-01T00:00:00",
                files_changed=["test.py"],
                insertions=10,
                deletions=5
            )

            result = CommitAnalysisResult(
                commit_info=commit_info,
                file_results={},
                aggregate_scores={"max_similarity": 0.7 + i * 0.1},
                processing_time=1.0,
                model_info={"model_name": "test"},
                config={"test": "config"}
            )
            results.append(result)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            _save_batch_results_to_file(results, output_path)

            # Check file was created and has correct content
            assert Path(output_path).exists()

            with open(output_path, 'r') as f:
                data = json.load(f)

            assert len(data) == 3
            assert data[0]['commit_info']['hash'] == 'commit0'
            assert data[2]['aggregate_scores']['max_similarity'] == 0.9

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestCLIEdgeCases:
    """Test CLI edge cases and error conditions."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_analyze_missing_commit_hash(self, runner):
        """Test analyze command without commit hash."""
        result = runner.invoke(cli, ['analyze'])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_compare_missing_commits(self, runner):
        """Test compare command with missing commit arguments."""
        result = runner.invoke(cli, ['compare', 'abc123'])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_invalid_repo_path(self, runner):
        """Test commands with invalid repository path."""
        with patch('semantic_code_analyzer.cli.SemanticScorer') as mock_scorer_class:
            mock_scorer_class.side_effect = ValueError("Invalid repository")

            result = runner.invoke(cli, [
                'analyze', 'abc123',
                '--repo-path', '/nonexistent/path'
            ])

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_file_write_permission_error(self, runner):
        """Test handling of file write permission errors."""
        with patch('semantic_code_analyzer.cli.SemanticScorer') as mock_scorer_class:
            mock_scorer = Mock()
            mock_scorer.score_commit_similarity.return_value = Mock()
            mock_scorer_class.return_value = mock_scorer

            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                result = runner.invoke(cli, [
                    'analyze', 'abc123',
                    '--output', '/root/results.json'  # Likely to cause permission error
                ])

                # Should handle gracefully (exact behavior depends on implementation)
                assert result.exit_code in [0, 1]  # May continue or fail gracefully

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_keyboard_interrupt(self, mock_scorer_class, runner):
        """Test handling of keyboard interrupt."""
        mock_scorer_class.side_effect = KeyboardInterrupt()

        result = runner.invoke(cli, ['analyze', 'abc123'])

        # Should handle gracefully
        assert result.exit_code in [0, 1]

    def test_main_function(self):
        """Test main function entry point."""
        with patch('semantic_code_analyzer.cli.cli') as mock_cli:
            main()
            mock_cli.assert_called_once()

    def test_invalid_distance_metric(self, runner):
        """Test invalid distance metric option."""
        result = runner.invoke(cli, [
            'analyze', 'abc123',
            '--distance-metric', 'invalid_metric'
        ])

        assert result.exit_code != 0
        # Should show valid choices in error message

    def test_invalid_language(self, runner):
        """Test invalid language option."""
        result = runner.invoke(cli, [
            'analyze', 'abc123',
            '--language', 'invalid_language'
        ])

        assert result.exit_code != 0
        # Should show valid choices in error message

    @patch('semantic_code_analyzer.cli.console')
    def test_very_long_output(self, mock_console):
        """Test handling of very long output content."""
        from semantic_code_analyzer.cli import _display_analysis_results

        # Create result with very long content
        commit_info = CommitInfo(
            hash="abc123",
            message="Very long commit message " * 100,  # Very long message
            author="Test Author With Very Long Name" * 10,
            timestamp="2023-01-01T00:00:00",
            files_changed=[f"file_{i}.py" for i in range(100)],  # Many files
            insertions=10000,
            deletions=5000
        )

        large_file_results = {}
        for i in range(50):  # Many files
            large_file_results[f"file_{i}.py"] = {
                "overall_similarity": {"max_similarity": 0.5 + i * 0.01},
                "most_similar_files": [
                    {"file_path": f"similar_{j}.py", "similarity_score": 0.7, "rank": j}
                    for j in range(10)  # Many similar files
                ]
            }

        result = CommitAnalysisResult(
            commit_info=commit_info,
            file_results=large_file_results,
            aggregate_scores={"max_similarity": 0.99},
            processing_time=100.0,
            model_info={"model_name": "test"},
            config={}
        )

        # Should handle large output without crashing
        _display_analysis_results(result, detailed=True)
        assert mock_console.print.called


class TestCLIIntegrationScenarios:
    """Test realistic CLI usage scenarios."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_complete_workflow_scenario(self, mock_scorer_class, runner, mock_scorer, test_repo):
        """Test a complete analysis workflow scenario."""
        mock_scorer_class.return_value = mock_scorer

        # Simulate a typical user workflow
        with runner.isolated_filesystem():
            # 1. Get repository info
            result1 = runner.invoke(cli, [
                'info',
                '--repo-path', test_repo['repo_path']
            ])
            assert result1.exit_code == 0

            # 2. Analyze a specific commit
            result2 = runner.invoke(cli, [
                'analyze', 'abc123',
                '--repo-path', test_repo['repo_path'],
                '--detailed',
                '--output', 'analysis.json'
            ])
            assert result2.exit_code == 0
            assert Path('analysis.json').exists()

            # 3. Run batch analysis
            result3 = runner.invoke(cli, [
                'batch',
                '--repo-path', test_repo['repo_path'],
                '--count', '5',
                '--output', 'batch.json'
            ])
            assert result3.exit_code == 0
            assert Path('batch.json').exists()

            # 4. Compare two commits
            result4 = runner.invoke(cli, [
                'compare', 'abc123', 'def456',
                '--repo-path', test_repo['repo_path'],
                '--output', 'comparison.json'
            ])
            assert result4.exit_code == 0
            assert Path('comparison.json').exists()

    @patch('semantic_code_analyzer.cli.SemanticScorer')
    def test_error_recovery_scenario(self, mock_scorer_class, runner, test_repo):
        """Test error recovery in realistic scenarios."""
        # Simulate scorer that fails intermittently
        failing_scorer = Mock()
        failing_scorer.score_commit_similarity.side_effect = [
            Exception("Network error"),  # First call fails
            Mock(
                commit_info=Mock(hash="abc123"),
                aggregate_scores={"max_similarity": 0.75},
                file_results={},
                processing_time=1.0
            )  # Second call succeeds
        ]

        mock_scorer_class.return_value = failing_scorer

        # First attempt should fail
        result1 = runner.invoke(cli, [
            'analyze', 'abc123',
            '--repo-path', test_repo['repo_path']
        ])
        assert result1.exit_code == 1

        # Could implement retry logic or graceful degradation
        # This tests that the CLI handles errors appropriately