"""
Main semantic similarity scorer that orchestrates all components.

This module provides the primary interface for analyzing semantic similarity
between Git commits and existing codebases using state-of-the-art embeddings
and optimized distance metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time
import json
from dataclasses import dataclass, asdict
import numpy as np

from .commit_extractor import CommitExtractor, CommitInfo
from .code_embedder import CodeEmbedder, EmbeddingConfig
from .similarity_calculator import SimilarityCalculator, DistanceMetric, SimilarityResult

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Configuration for the SemanticScorer."""
    # Embedding configuration
    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 512
    use_mps: bool = True
    normalize_embeddings: bool = True

    # Similarity configuration
    distance_metric: str = "euclidean"
    normalize_scores: bool = True

    # Processing configuration
    max_files: Optional[int] = None
    exclude_patterns: List[str] = None
    include_functions: bool = True
    cache_embeddings: bool = True
    compare_against_parent: bool = True  # Default: compare against parent commit state

    # Output configuration
    detailed_output: bool = True
    save_results: bool = False
    results_dir: str = "similarity_results"

    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__", ".git", "node_modules", ".venv",
                "test_", "_test.", ".test.", "spec_", "_spec."
            ]


@dataclass
class CommitAnalysisResult:
    """Complete analysis result for a commit."""
    commit_info: CommitInfo
    file_results: Dict[str, Dict[str, Any]]
    aggregate_scores: Dict[str, float]
    processing_time: float
    model_info: Dict[str, Any]
    config: Dict[str, Any]


class SemanticScorer:
    """
    Main orchestration class for semantic similarity analysis.

    Combines commit extraction, code embedding, and similarity calculation
    to provide comprehensive semantic analysis of Git commits against codebases.
    """

    def __init__(self,
                 repo_path: str,
                 config: Optional[ScorerConfig] = None):
        """
        Initialize the SemanticScorer.

        Args:
            repo_path: Path to the Git repository
            config: Configuration object for the scorer

        Raises:
            ValueError: If repository path is invalid
        """
        self.repo_path = Path(repo_path).resolve()
        self.config = config or ScorerConfig()

        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Initialize components
        logger.info(f"Initializing SemanticScorer for {self.repo_path}")

        try:
            # Initialize commit extractor
            self.commit_extractor = CommitExtractor(str(self.repo_path))

            # Validate repository
            if not self.commit_extractor.validate_repository():
                raise ValueError("Repository validation failed")

            # Initialize code embedder
            embedding_config = EmbeddingConfig(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                use_mps=self.config.use_mps,
                cache_embeddings=self.config.cache_embeddings,
                normalize_embeddings=self.config.normalize_embeddings
            )
            self.code_embedder = CodeEmbedder(embedding_config)

            # Initialize similarity calculator
            distance_metric = DistanceMetric(self.config.distance_metric)
            self.similarity_calculator = SimilarityCalculator(
                distance_metric=distance_metric,
                normalize_scores=self.config.normalize_scores
            )

            logger.info("SemanticScorer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SemanticScorer: {e}")
            raise

    def score_commit_similarity(self,
                               commit_hash: str,
                               language: str = "python") -> CommitAnalysisResult:
        """
        Analyze semantic similarity of a commit against the codebase.

        Args:
            commit_hash: Hash of the commit to analyze
            language: Programming language of the code

        Returns:
            CommitAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        logger.info(f"Starting similarity analysis for commit {commit_hash}")

        try:
            # Step 1: Extract commit information and changes
            commit_info = self.commit_extractor.get_commit_info(commit_hash)
            commit_changes = self.commit_extractor.extract_commit_changes(commit_hash)

            if not commit_changes:
                logger.warning(f"No code changes found in commit {commit_hash}")
                return self._create_empty_result(commit_info, start_time)

            logger.info(f"Found {len(commit_changes)} changed files in commit")

            # Step 2: Get existing codebase for comparison
            if self.config.compare_against_parent:
                # Default: Compare against parent commit state (before this commit)
                existing_codebase = self.commit_extractor.get_codebase_at_parent_commit(
                    commit_hash, max_files=self.config.max_files
                )
                logger.info("Comparing against parent commit state")
            else:
                # Alternative: Compare against current filesystem (excluding commit files)
                existing_codebase = self.commit_extractor.get_existing_codebase(
                    exclude_files=list(commit_changes.keys()),
                    max_files=self.config.max_files
                )
                logger.info("Comparing against current filesystem state")

            if not existing_codebase:
                logger.warning("No existing codebase files found for comparison")
                return self._create_empty_result(commit_info, start_time)

            logger.info(f"Loaded {len(existing_codebase)} existing codebase files")

            # Step 3: Generate embeddings for commit changes
            commit_embeddings = self._generate_commit_embeddings(
                commit_changes, language
            )

            # Step 4: Generate embeddings for existing codebase
            codebase_embeddings = self._generate_codebase_embeddings(
                existing_codebase, language
            )

            # Step 5: Calculate similarity scores
            file_results = self._calculate_file_similarities(
                commit_embeddings, codebase_embeddings
            )

            # Step 6: Calculate aggregate scores
            aggregate_scores = self._calculate_aggregate_scores(
                commit_embeddings, codebase_embeddings, file_results
            )

            # Step 7: Create result
            processing_time = time.time() - start_time

            result = CommitAnalysisResult(
                commit_info=commit_info,
                file_results=file_results,
                aggregate_scores=aggregate_scores,
                processing_time=processing_time,
                model_info=self.code_embedder.get_model_info(),
                config=asdict(self.config)
            )

            logger.info(f"Similarity analysis completed in {processing_time:.2f}s")
            logger.info(f"Aggregate similarity score: {aggregate_scores.get('max_similarity', 0):.3f}")

            # Save results if requested
            if self.config.save_results:
                self._save_results(result, commit_hash)

            return result

        except Exception as e:
            logger.error(f"Failed to analyze commit {commit_hash}: {e}")
            raise

    def score_multiple_commits(self,
                             commit_hashes: List[str],
                             language: str = "python") -> List[CommitAnalysisResult]:
        """
        Analyze multiple commits for semantic similarity.

        Args:
            commit_hashes: List of commit hashes to analyze
            language: Programming language of the code

        Returns:
            List of CommitAnalysisResult objects
        """
        logger.info(f"Analyzing {len(commit_hashes)} commits")

        results = []
        for i, commit_hash in enumerate(commit_hashes):
            logger.info(f"Processing commit {i+1}/{len(commit_hashes)}: {commit_hash}")

            try:
                result = self.score_commit_similarity(commit_hash, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process commit {commit_hash}: {e}")
                continue

        logger.info(f"Successfully analyzed {len(results)}/{len(commit_hashes)} commits")
        return results

    def get_recent_commits_analysis(self,
                                  max_commits: int = 10,
                                  branch: str = "HEAD",
                                  language: str = "python") -> List[CommitAnalysisResult]:
        """
        Analyze recent commits from the repository.

        Args:
            max_commits: Maximum number of recent commits to analyze
            branch: Branch to get commits from
            language: Programming language of the code

        Returns:
            List of CommitAnalysisResult objects
        """
        # Get recent commits
        recent_commits = self.commit_extractor.get_commit_list(
            branch=branch,
            max_count=max_commits
        )

        commit_hashes = [commit.hash for commit in recent_commits]
        return self.score_multiple_commits(commit_hashes, language)

    def compare_commits(self,
                       commit_hash_a: str,
                       commit_hash_b: str,
                       language: str = "python") -> Dict[str, Any]:
        """
        Compare similarity scores between two commits.

        Args:
            commit_hash_a: First commit hash
            commit_hash_b: Second commit hash
            language: Programming language of the code

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing commits {commit_hash_a} and {commit_hash_b}")

        # Analyze both commits
        result_a = self.score_commit_similarity(commit_hash_a, language)
        result_b = self.score_commit_similarity(commit_hash_b, language)

        # Extract embeddings for direct comparison
        changes_a = self.commit_extractor.extract_commit_changes(commit_hash_a)
        changes_b = self.commit_extractor.extract_commit_changes(commit_hash_b)

        embeddings_a = self._generate_commit_embeddings(changes_a, language)
        embeddings_b = self._generate_commit_embeddings(changes_b, language)

        # Calculate cross-similarity
        cross_similarity = self.similarity_calculator.calculate_cross_similarity_matrix(
            list(embeddings_a.values()),
            list(embeddings_b.values()),
            list(embeddings_a.keys()),
            list(embeddings_b.keys())
        )

        return {
            "commit_a": result_a,
            "commit_b": result_b,
            "cross_similarity": cross_similarity,
            "similarity_difference": {
                "max_similarity_diff": (
                    result_b.aggregate_scores["max_similarity"] -
                    result_a.aggregate_scores["max_similarity"]
                ),
                "mean_similarity_diff": (
                    result_b.aggregate_scores["mean_similarity"] -
                    result_a.aggregate_scores["mean_similarity"]
                )
            }
        }

    def _generate_commit_embeddings(self,
                                   commit_changes: Dict[str, str],
                                   language: str) -> Dict[str, np.ndarray]:
        """Generate embeddings for commit changes."""
        logger.info(f"Generating embeddings for {len(commit_changes)} commit files")

        embeddings = {}
        for file_path, code in commit_changes.items():
            try:
                embedding = self.code_embedder.get_code_embedding(code, language)
                embeddings[file_path] = embedding
                logger.debug(f"Generated embedding for {file_path}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {file_path}: {e}")
                continue

        logger.info(f"Generated {len(embeddings)} commit embeddings")
        return embeddings

    def _generate_codebase_embeddings(self,
                                    codebase: Dict[str, str],
                                    language: str) -> Dict[str, np.ndarray]:
        """Generate embeddings for codebase files."""
        logger.info(f"Generating embeddings for {len(codebase)} codebase files")

        embeddings = {}
        codes = list(codebase.values())
        file_paths = list(codebase.keys())

        # Use batch processing for efficiency
        try:
            batch_embeddings = self.code_embedder.get_batch_embeddings(codes, language)

            for file_path, embedding in zip(file_paths, batch_embeddings):
                embeddings[file_path] = embedding

        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to individual: {e}")

            # Fallback to individual processing
            for file_path, code in codebase.items():
                try:
                    embedding = self.code_embedder.get_code_embedding(code, language)
                    embeddings[file_path] = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {file_path}: {e}")
                    continue

        logger.info(f"Generated {len(embeddings)} codebase embeddings")
        return embeddings

    def _calculate_file_similarities(self,
                                   commit_embeddings: Dict[str, np.ndarray],
                                   codebase_embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Calculate similarity scores for each commit file."""
        logger.info("Calculating file-level similarities")

        file_results = {}
        reference_embeddings = list(codebase_embeddings.values())

        for commit_file, commit_embedding in commit_embeddings.items():
            # Overall similarity against entire codebase
            similarity_result = self.similarity_calculator.calculate_similarity_score(
                commit_embedding, reference_embeddings, return_details=True
            )

            # Find most similar files
            similar_files = self.similarity_calculator.find_most_similar_files(
                commit_embedding, codebase_embeddings, top_k=10
            )

            file_results[commit_file] = {
                "overall_similarity": asdict(similarity_result),
                "most_similar_files": [
                    {
                        "file_path": sim.file_path,
                        "similarity_score": sim.similarity_score,
                        "rank": sim.rank
                    }
                    for sim in similar_files
                ],
                "embedding_shape": commit_embedding.shape
            }

        return file_results

    def _calculate_aggregate_scores(self,
                                  commit_embeddings: Dict[str, np.ndarray],
                                  codebase_embeddings: Dict[str, np.ndarray],
                                  file_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate similarity scores."""
        logger.info("Calculating aggregate similarity scores")

        # Extract individual file scores
        max_similarities = [
            result["overall_similarity"]["max_similarity"]
            for result in file_results.values()
        ]
        mean_similarities = [
            result["overall_similarity"]["mean_similarity"]
            for result in file_results.values()
        ]

        # Calculate different aggregation methods
        aggregate_scores = {
            "max_similarity": max(max_similarities) if max_similarities else 0.0,
            "mean_similarity": np.mean(mean_similarities) if mean_similarities else 0.0,
            "median_similarity": np.median(max_similarities) if max_similarities else 0.0,
            "weighted_similarity": np.mean(max_similarities) if max_similarities else 0.0,
        }

        # Add advanced aggregate using similarity calculator
        advanced_aggregate = self.similarity_calculator.calculate_aggregate_similarity(
            commit_embeddings, codebase_embeddings, aggregation_method="max"
        )
        aggregate_scores.update(advanced_aggregate)

        return aggregate_scores

    def _create_empty_result(self,
                           commit_info: CommitInfo,
                           start_time: float) -> CommitAnalysisResult:
        """Create an empty result for commits with no changes."""
        processing_time = time.time() - start_time

        return CommitAnalysisResult(
            commit_info=commit_info,
            file_results={},
            aggregate_scores={
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "median_similarity": 0.0,
                "weighted_similarity": 0.0,
                "aggregate_similarity": 0.0
            },
            processing_time=processing_time,
            model_info=self.code_embedder.get_model_info(),
            config=asdict(self.config)
        )

    def _save_results(self,
                     result: CommitAnalysisResult,
                     commit_hash: str):
        """Save analysis results to file."""
        try:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)

            filename = f"similarity_analysis_{commit_hash}_{int(time.time())}.json"
            filepath = results_dir / filename

            # Convert result to JSON-serializable format
            result_dict = asdict(result)

            # Handle numpy arrays and other non-serializable objects
            result_dict = self._make_json_serializable(result_dict)

            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)

            logger.info(f"Results saved to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def get_scorer_info(self) -> Dict[str, Any]:
        """Get information about the scorer configuration and state."""
        return {
            "repo_path": str(self.repo_path),
            "config": asdict(self.config),
            "model_info": self.code_embedder.get_model_info(),
            "cache_info": self.similarity_calculator.get_cache_info(),
            "supported_languages": ["python", "javascript", "java", "cpp", "c", "go"]
        }

    def clear_caches(self):
        """Clear all internal caches."""
        self.code_embedder.clear_cache()
        self.similarity_calculator.clear_cache()
        logger.info("All caches cleared")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'code_embedder'):
            self.code_embedder.save_cache()