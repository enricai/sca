"""
Tests for the SemanticScorer module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time

from semantic_code_analyzer.semantic_scorer import (
    SemanticScorer,
    ScorerConfig,
    CommitAnalysisResult
)
from semantic_code_analyzer.commit_extractor import CommitInfo


class TestScorerConfig:
    """Test cases for ScorerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScorerConfig()

        assert config.model_name == "microsoft/graphcodebert-base"
        assert config.max_length == 512
        assert config.use_mps is True
        assert config.normalize_embeddings is True
        assert config.distance_metric == "euclidean"
        assert config.normalize_scores is True
        assert config.max_files is None
        assert config.include_functions is True
        assert config.cache_embeddings is True
        assert config.detailed_output is True
        assert config.save_results is False
        assert config.results_dir == "similarity_results"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScorerConfig(
            model_name="custom/model",
            max_length=256,
            use_mps=False,
            distance_metric="cosine",
            max_files=50,
            cache_embeddings=False,
            detailed_output=False,
            save_results=True,
            results_dir="custom_results"
        )

        assert config.model_name == "custom/model"
        assert config.max_length == 256
        assert config.use_mps is False
        assert config.distance_metric == "cosine"
        assert config.max_files == 50
        assert config.cache_embeddings is False
        assert config.detailed_output is False
        assert config.save_results is True
        assert config.results_dir == "custom_results"

    def test_post_init_default_exclude_patterns(self):
        """Test that default exclude patterns are set in post_init."""
        config = ScorerConfig()

        assert config.exclude_patterns is not None
        assert "__pycache__" in config.exclude_patterns
        assert ".git" in config.exclude_patterns
        assert "test_" in config.exclude_patterns

    def test_post_init_custom_exclude_patterns(self):
        """Test custom exclude patterns."""
        custom_patterns = ["custom1", "custom2"]
        config = ScorerConfig(exclude_patterns=custom_patterns)

        assert config.exclude_patterns == custom_patterns


class TestCommitAnalysisResult:
    """Test cases for CommitAnalysisResult dataclass."""

    def test_commit_analysis_result_creation(self):
        """Test creating CommitAnalysisResult instance."""
        commit_info = CommitInfo(
            hash="abc123",
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        file_results = {
            "test.py": {
                "overall_similarity": {"max_similarity": 0.75},
                "most_similar_files": []
            }
        }

        aggregate_scores = {"max_similarity": 0.75, "mean_similarity": 0.65}

        result = CommitAnalysisResult(
            commit_info=commit_info,
            file_results=file_results,
            aggregate_scores=aggregate_scores,
            processing_time=1.5,
            model_info={"model_name": "test"},
            config={"test": "config"}
        )

        assert result.commit_info == commit_info
        assert result.file_results == file_results
        assert result.aggregate_scores == aggregate_scores
        assert result.processing_time == 1.5
        assert result.model_info == {"model_name": "test"}
        assert result.config == {"test": "config"}


class TestSemanticScorer:
    """Test cases for SemanticScorer class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for SemanticScorer."""
        # Mock CommitExtractor
        mock_commit_extractor = Mock()
        mock_commit_extractor.validate_repository.return_value = True

        # Mock CodeEmbedder
        mock_code_embedder = Mock()
        mock_code_embedder.get_model_info.return_value = {
            "model_name": "test-model",
            "device": "cpu",
            "embedding_dim": 768
        }

        def mock_get_embedding(code, language="python", use_cache=True):
            # Generate deterministic embeddings based on code hash
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()
            np.random.seed(int(code_hash[:8], 16) % (2**32))
            return np.random.rand(768).astype(np.float32)

        mock_code_embedder.get_code_embedding.side_effect = mock_get_embedding
        mock_code_embedder.get_batch_embeddings.side_effect = lambda codes, lang: [
            mock_get_embedding(code, lang) for code in codes
        ]
        mock_code_embedder.save_cache.return_value = None
        mock_code_embedder.clear_cache.return_value = None

        # Mock SimilarityCalculator
        mock_similarity_calculator = Mock()
        mock_similarity_calculator.calculate_similarity_score.return_value = Mock(
            max_similarity=0.75,
            mean_similarity=0.65,
            median_similarity=0.70,
            std_similarity=0.10,
            min_similarity=0.50,
            similarity_scores=[0.60, 0.70, 0.75],
            distance_metric="euclidean"
        )
        mock_similarity_calculator.find_most_similar_files.return_value = []
        mock_similarity_calculator.calculate_aggregate_similarity.return_value = {
            "aggregate_similarity": 0.75,
            "method": "max"
        }
        mock_similarity_calculator.get_cache_info.return_value = {"enabled": True, "size": 0}
        mock_similarity_calculator.clear_cache.return_value = None

        return {
            "commit_extractor": mock_commit_extractor,
            "code_embedder": mock_code_embedder,
            "similarity_calculator": mock_similarity_calculator
        }

    @pytest.fixture
    def scorer_with_mocks(self, test_repo, mock_components):
        """Create SemanticScorer with mocked components."""
        repo_path = test_repo['repo_path']
        config = ScorerConfig(cache_embeddings=False, save_results=False)

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor') as mock_ce_class, \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder') as mock_emb_class, \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator') as mock_sim_class:

            mock_ce_class.return_value = mock_components["commit_extractor"]
            mock_emb_class.return_value = mock_components["code_embedder"]
            mock_sim_class.return_value = mock_components["similarity_calculator"]

            scorer = SemanticScorer(repo_path, config)
            return scorer

    def test_init_valid_repo(self, test_repo, mock_components):
        """Test initialization with valid repository."""
        repo_path = test_repo['repo_path']
        config = ScorerConfig()

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor') as mock_ce_class, \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder') as mock_emb_class, \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator') as mock_sim_class:

            mock_ce_class.return_value = mock_components["commit_extractor"]
            mock_emb_class.return_value = mock_components["code_embedder"]
            mock_sim_class.return_value = mock_components["similarity_calculator"]

            scorer = SemanticScorer(repo_path, config)

            assert scorer.repo_path == Path(repo_path).resolve()
            assert scorer.config == config

    def test_init_invalid_repo_path(self, mock_components):
        """Test initialization with invalid repository path."""
        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor'), \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder'), \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator'):

            with pytest.raises(ValueError, match="Repository path does not exist"):
                SemanticScorer("/nonexistent/path")

    def test_init_invalid_repo_validation(self, test_repo, mock_components):
        """Test initialization with repository validation failure."""
        repo_path = test_repo['repo_path']
        mock_components["commit_extractor"].validate_repository.return_value = False

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor') as mock_ce_class, \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder'), \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator'):

            mock_ce_class.return_value = mock_components["commit_extractor"]

            with pytest.raises(ValueError, match="Repository validation failed"):
                SemanticScorer(repo_path)

    def test_score_commit_similarity_basic(self, scorer_with_mocks, test_repo):
        """Test basic commit similarity scoring."""
        commit_hash = test_repo['commits'][1]

        # Setup mocks
        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {
            "test.py": "def test(): pass"
        }

        scorer_with_mocks.commit_extractor.get_existing_codebase.return_value = {
            "existing.py": "def existing(): pass"
        }

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        assert result.commit_info.hash == commit_hash[:8]
        assert "test.py" in result.file_results
        assert "max_similarity" in result.aggregate_scores
        assert result.processing_time > 0

    def test_score_commit_similarity_no_changes(self, scorer_with_mocks, test_repo):
        """Test commit analysis with no code changes."""
        commit_hash = test_repo['commits'][1]

        # Setup mocks
        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=[],
            insertions=0,
            deletions=0
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {}

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        assert len(result.file_results) == 0
        assert result.aggregate_scores["max_similarity"] == 0.0

    def test_score_commit_similarity_no_existing_codebase(self, scorer_with_mocks, test_repo):
        """Test commit analysis with no existing codebase."""
        commit_hash = test_repo['commits'][1]

        # Setup mocks
        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {
            "test.py": "def test(): pass"
        }

        scorer_with_mocks.commit_extractor.get_existing_codebase.return_value = {}

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        assert len(result.file_results) == 0
        assert result.aggregate_scores["max_similarity"] == 0.0

    def test_score_multiple_commits(self, scorer_with_mocks, test_repo):
        """Test analyzing multiple commits."""
        commit_hashes = test_repo['commits'][:3]

        # Setup mocks for each commit
        def mock_score_commit(commit_hash, language):
            return CommitAnalysisResult(
                commit_info=CommitInfo(
                    hash=commit_hash[:8],
                    message=f"Commit {commit_hash[:8]}",
                    author="Test Author",
                    timestamp="2023-01-01T00:00:00",
                    files_changed=["test.py"],
                    insertions=10,
                    deletions=5
                ),
                file_results={"test.py": {"overall_similarity": {"max_similarity": 0.75}}},
                aggregate_scores={"max_similarity": 0.75, "mean_similarity": 0.65},
                processing_time=1.0,
                model_info={"model_name": "test"},
                config={}
            )

        scorer_with_mocks.score_commit_similarity = Mock(side_effect=mock_score_commit)

        results = scorer_with_mocks.score_multiple_commits(commit_hashes, "python")

        assert len(results) == 3
        assert all(isinstance(r, CommitAnalysisResult) for r in results)

    def test_score_multiple_commits_with_failures(self, scorer_with_mocks, test_repo):
        """Test batch analysis with some commits failing."""
        commit_hashes = test_repo['commits'][:3]

        def mock_score_commit(commit_hash, language):
            if commit_hash == commit_hashes[1]:  # Second commit fails
                raise Exception("Analysis failed")
            return CommitAnalysisResult(
                commit_info=CommitInfo(
                    hash=commit_hash[:8],
                    message=f"Commit {commit_hash[:8]}",
                    author="Test Author",
                    timestamp="2023-01-01T00:00:00",
                    files_changed=["test.py"],
                    insertions=10,
                    deletions=5
                ),
                file_results={},
                aggregate_scores={"max_similarity": 0.75},
                processing_time=1.0,
                model_info={},
                config={}
            )

        scorer_with_mocks.score_commit_similarity = Mock(side_effect=mock_score_commit)

        results = scorer_with_mocks.score_multiple_commits(commit_hashes, "python")

        # Should return 2 results (excluding the failed one)
        assert len(results) == 2

    def test_get_recent_commits_analysis(self, scorer_with_mocks):
        """Test analyzing recent commits."""
        # Mock commit list
        mock_commits = [
            CommitInfo(
                hash=f"commit{i}",
                message=f"Message {i}",
                author="Test Author",
                timestamp="2023-01-01T00:00:00",
                files_changed=["test.py"],
                insertions=10,
                deletions=5
            )
            for i in range(5)
        ]

        scorer_with_mocks.commit_extractor.get_commit_list.return_value = mock_commits

        def mock_score_multiple(commit_hashes, language):
            return [
                CommitAnalysisResult(
                    commit_info=mock_commits[i],
                    file_results={},
                    aggregate_scores={"max_similarity": 0.75},
                    processing_time=1.0,
                    model_info={},
                    config={}
                )
                for i in range(len(commit_hashes))
            ]

        scorer_with_mocks.score_multiple_commits = Mock(side_effect=mock_score_multiple)

        results = scorer_with_mocks.get_recent_commits_analysis(max_commits=3, language="python")

        assert len(results) == 3
        scorer_with_mocks.commit_extractor.get_commit_list.assert_called_with(
            branch="HEAD", max_count=3
        )

    def test_compare_commits(self, scorer_with_mocks, test_repo):
        """Test comparing two commits."""
        commit_a = test_repo['commits'][0]
        commit_b = test_repo['commits'][1]

        # Mock individual commit analyses
        def mock_score_commit(commit_hash, language):
            return CommitAnalysisResult(
                commit_info=CommitInfo(
                    hash=commit_hash[:8],
                    message=f"Commit {commit_hash[:8]}",
                    author="Test Author",
                    timestamp="2023-01-01T00:00:00",
                    files_changed=["test.py"],
                    insertions=10,
                    deletions=5
                ),
                file_results={},
                aggregate_scores={"max_similarity": 0.75, "mean_similarity": 0.65},
                processing_time=1.0,
                model_info={},
                config={}
            )

        scorer_with_mocks.score_commit_similarity = Mock(side_effect=mock_score_commit)

        # Mock commit extraction
        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {
            "test.py": "def test(): pass"
        }

        # Mock cross-similarity calculation
        scorer_with_mocks.similarity_calculator.calculate_cross_similarity_matrix.return_value = {
            "similarity_matrix": np.array([[1.0, 0.8], [0.8, 1.0]]),
            "max_similarity": 1.0,
            "mean_similarity": 0.9,
            "shape": (2, 2)
        }

        result = scorer_with_mocks.compare_commits(commit_a, commit_b, "python")

        assert "commit_a" in result
        assert "commit_b" in result
        assert "cross_similarity" in result
        assert "similarity_difference" in result

        assert result["similarity_difference"]["max_similarity_diff"] == 0.0  # Same scores

    def test_generate_commit_embeddings(self, scorer_with_mocks):
        """Test generating embeddings for commit changes."""
        commit_changes = {
            "file1.py": "def func1(): pass",
            "file2.py": "def func2(): pass"
        }

        embeddings = scorer_with_mocks._generate_commit_embeddings(commit_changes, "python")

        assert len(embeddings) == 2
        assert "file1.py" in embeddings
        assert "file2.py" in embeddings
        assert all(isinstance(emb, np.ndarray) for emb in embeddings.values())

    def test_generate_commit_embeddings_with_errors(self, scorer_with_mocks):
        """Test commit embeddings generation with some files failing."""
        commit_changes = {
            "file1.py": "def func1(): pass",
            "file2.py": "def func2(): pass"
        }

        # Mock embedder to fail on second file
        def mock_get_embedding(code, language):
            if "func2" in code:
                raise Exception("Embedding failed")
            return np.random.rand(768).astype(np.float32)

        scorer_with_mocks.code_embedder.get_code_embedding.side_effect = mock_get_embedding

        embeddings = scorer_with_mocks._generate_commit_embeddings(commit_changes, "python")

        # Should only have embedding for file1.py
        assert len(embeddings) == 1
        assert "file1.py" in embeddings

    def test_generate_codebase_embeddings_batch(self, scorer_with_mocks):
        """Test generating embeddings for codebase using batch processing."""
        codebase = {
            "file1.py": "def func1(): pass",
            "file2.py": "def func2(): pass",
            "file3.py": "def func3(): pass"
        }

        embeddings = scorer_with_mocks._generate_codebase_embeddings(codebase, "python")

        assert len(embeddings) == 3
        assert all(file_path in embeddings for file_path in codebase.keys())
        assert all(isinstance(emb, np.ndarray) for emb in embeddings.values())

    def test_generate_codebase_embeddings_fallback(self, scorer_with_mocks):
        """Test codebase embeddings with fallback to individual processing."""
        codebase = {
            "file1.py": "def func1(): pass",
            "file2.py": "def func2(): pass"
        }

        # Mock batch processing to fail
        scorer_with_mocks.code_embedder.get_batch_embeddings.side_effect = Exception("Batch failed")

        embeddings = scorer_with_mocks._generate_codebase_embeddings(codebase, "python")

        assert len(embeddings) == 2
        # Should have called individual embedding generation as fallback
        assert scorer_with_mocks.code_embedder.get_code_embedding.call_count >= 2

    def test_calculate_file_similarities(self, scorer_with_mocks):
        """Test calculating file-level similarities."""
        commit_embeddings = {
            "commit_file.py": np.random.rand(768).astype(np.float32)
        }
        codebase_embeddings = {
            "existing1.py": np.random.rand(768).astype(np.float32),
            "existing2.py": np.random.rand(768).astype(np.float32)
        }

        file_results = scorer_with_mocks._calculate_file_similarities(
            commit_embeddings, codebase_embeddings
        )

        assert "commit_file.py" in file_results
        result = file_results["commit_file.py"]
        assert "overall_similarity" in result
        assert "most_similar_files" in result
        assert "embedding_shape" in result

    def test_calculate_aggregate_scores(self, scorer_with_mocks):
        """Test calculating aggregate similarity scores."""
        commit_embeddings = {
            "file1.py": np.random.rand(768).astype(np.float32),
            "file2.py": np.random.rand(768).astype(np.float32)
        }
        codebase_embeddings = {
            "existing.py": np.random.rand(768).astype(np.float32)
        }

        file_results = {
            "file1.py": {
                "overall_similarity": {"max_similarity": 0.8, "mean_similarity": 0.7}
            },
            "file2.py": {
                "overall_similarity": {"max_similarity": 0.6, "mean_similarity": 0.5}
            }
        }

        aggregate_scores = scorer_with_mocks._calculate_aggregate_scores(
            commit_embeddings, codebase_embeddings, file_results
        )

        assert "max_similarity" in aggregate_scores
        assert "mean_similarity" in aggregate_scores
        assert "median_similarity" in aggregate_scores
        assert "weighted_similarity" in aggregate_scores

        # Should be maximum of file max similarities
        assert aggregate_scores["max_similarity"] == 0.8

    def test_save_results(self, scorer_with_mocks):
        """Test saving analysis results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer_with_mocks.config.results_dir = tmpdir
            scorer_with_mocks.config.save_results = True

            result = CommitAnalysisResult(
                commit_info=CommitInfo(
                    hash="abc123",
                    message="Test commit",
                    author="Test Author",
                    timestamp="2023-01-01T00:00:00",
                    files_changed=["test.py"],
                    insertions=10,
                    deletions=5
                ),
                file_results={},
                aggregate_scores={"max_similarity": 0.75},
                processing_time=1.0,
                model_info={"model_name": "test"},
                config={"test": "config"}
            )

            scorer_with_mocks._save_results(result, "abc123")

            # Check that file was created
            results_files = list(Path(tmpdir).glob("similarity_analysis_*.json"))
            assert len(results_files) == 1

            # Check file content
            with open(results_files[0], 'r') as f:
                saved_data = json.load(f)

            assert saved_data["commit_info"]["hash"] == "abc123"
            assert saved_data["aggregate_scores"]["max_similarity"] == 0.75

    def test_make_json_serializable(self, scorer_with_mocks):
        """Test making objects JSON serializable."""
        test_obj = {
            "numpy_array": np.array([1, 2, 3]),
            "numpy_int": np.int64(42),
            "numpy_float": np.float64(3.14),
            "regular_dict": {"nested": "value"},
            "regular_list": [1, 2, 3],
            "regular_value": "string"
        }

        serializable = scorer_with_mocks._make_json_serializable(test_obj)

        # Should be JSON serializable
        json_str = json.dumps(serializable)
        assert json_str is not None

        # Check conversions
        assert serializable["numpy_array"] == [1, 2, 3]
        assert serializable["numpy_int"] == 42
        assert serializable["numpy_float"] == 3.14
        assert serializable["regular_dict"]["nested"] == "value"

    def test_get_scorer_info(self, scorer_with_mocks):
        """Test retrieving scorer information."""
        info = scorer_with_mocks.get_scorer_info()

        assert "repo_path" in info
        assert "config" in info
        assert "model_info" in info
        assert "cache_info" in info
        assert "supported_languages" in info

        assert isinstance(info["supported_languages"], list)
        assert "python" in info["supported_languages"]

    def test_clear_caches(self, scorer_with_mocks):
        """Test clearing all caches."""
        scorer_with_mocks.clear_caches()

        scorer_with_mocks.code_embedder.clear_cache.assert_called_once()
        scorer_with_mocks.similarity_calculator.clear_cache.assert_called_once()

    def test_error_handling_general_exception(self, scorer_with_mocks, test_repo):
        """Test error handling for general exceptions."""
        commit_hash = test_repo['commits'][0]

        # Mock to raise exception
        scorer_with_mocks.commit_extractor.get_commit_info.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            scorer_with_mocks.score_commit_similarity(commit_hash, "python")

    def test_destructor_saves_cache(self, test_repo, mock_components):
        """Test that destructor calls save_cache."""
        repo_path = test_repo['repo_path']
        config = ScorerConfig()

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor') as mock_ce_class, \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder') as mock_emb_class, \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator') as mock_sim_class:

            mock_ce_class.return_value = mock_components["commit_extractor"]
            mock_emb_class.return_value = mock_components["code_embedder"]
            mock_sim_class.return_value = mock_components["similarity_calculator"]

            scorer = SemanticScorer(repo_path, config)

            # Manually call destructor
            scorer.__del__()

            mock_components["code_embedder"].save_cache.assert_called_once()


class TestSemanticScorerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def scorer_with_mocks(self, test_repo, mock_components):
        """Create SemanticScorer with mocked components."""
        repo_path = test_repo['repo_path']
        config = ScorerConfig(cache_embeddings=False, save_results=False)

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor') as mock_ce_class, \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder') as mock_emb_class, \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator') as mock_sim_class:

            mock_ce_class.return_value = mock_components["commit_extractor"]
            mock_emb_class.return_value = mock_components["code_embedder"]
            mock_sim_class.return_value = mock_components["similarity_calculator"]

            scorer = SemanticScorer(repo_path, config)
            return scorer

    def test_empty_commit_changes(self, scorer_with_mocks, test_repo):
        """Test handling empty commit changes."""
        commit_hash = test_repo['commits'][0]

        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Empty commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=[],
            insertions=0,
            deletions=0
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {}

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        assert len(result.file_results) == 0
        assert result.aggregate_scores["max_similarity"] == 0.0

    def test_large_number_of_files(self, scorer_with_mocks, test_repo):
        """Test handling large number of files."""
        commit_hash = test_repo['commits'][0]

        # Mock large commit
        large_commit_changes = {f"file_{i}.py": f"def func_{i}(): pass" for i in range(100)}
        large_codebase = {f"existing_{i}.py": f"def existing_{i}(): pass" for i in range(50)}

        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Large commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=list(large_commit_changes.keys()),
            insertions=1000,
            deletions=0
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = large_commit_changes
        scorer_with_mocks.commit_extractor.get_existing_codebase.return_value = large_codebase

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        assert len(result.file_results) == 100

    def test_very_large_files(self, scorer_with_mocks, test_repo):
        """Test handling very large files."""
        commit_hash = test_repo['commits'][0]

        # Create very large code content
        large_code = "def func():\n" + "    x = 1\n" * 10000

        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Large file commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["large_file.py"],
            insertions=20000,
            deletions=0
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {
            "large_file.py": large_code
        }
        scorer_with_mocks.commit_extractor.get_existing_codebase.return_value = {
            "existing.py": "def existing(): pass"
        }

        result = scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        assert isinstance(result, CommitAnalysisResult)
        # Should handle large files gracefully

    @patch('semantic_code_analyzer.semantic_scorer.logger')
    def test_logging_calls(self, mock_logger, scorer_with_mocks, test_repo):
        """Test that appropriate logging calls are made."""
        commit_hash = test_repo['commits'][0]

        scorer_with_mocks.commit_extractor.get_commit_info.return_value = CommitInfo(
            hash=commit_hash[:8],
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        scorer_with_mocks.commit_extractor.extract_commit_changes.return_value = {
            "test.py": "def test(): pass"
        }
        scorer_with_mocks.commit_extractor.get_existing_codebase.return_value = {
            "existing.py": "def existing(): pass"
        }

        scorer_with_mocks.score_commit_similarity(commit_hash, "python")

        # Should have made some log calls
        assert mock_logger.info.called

    def test_config_validation(self, test_repo):
        """Test configuration validation and edge cases."""
        repo_path = test_repo['repo_path']

        # Test with extreme values
        config = ScorerConfig(
            max_files=0,  # Edge case
            max_length=1,  # Very small
            batch_size=1000  # Very large
        )

        with patch('semantic_code_analyzer.semantic_scorer.CommitExtractor'), \
             patch('semantic_code_analyzer.semantic_scorer.CodeEmbedder'), \
             patch('semantic_code_analyzer.semantic_scorer.SimilarityCalculator'):

            # Should not raise exceptions
            scorer = SemanticScorer(repo_path, config)
            assert scorer.config.max_files == 0