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

"""Multi-dimensional scorer for comprehensive code quality analysis.

This module provides the main orchestration for multi-dimensional code analysis,
focusing on architectural patterns, code quality, TypeScript usage, and
framework-specific conventions.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import git
from rich.console import Console

from ..analyzers import (
    AnalysisResult,
    ArchitecturalAnalyzer,
    BaseAnalyzer,
    DomainAwareAdherenceAnalyzer,
    FrameworkAnalyzer,
    PatternMatch,
    QualityAnalyzer,
    Recommendation,
    TypeScriptAnalyzer,
)
from ..hardware import DeviceManager
from .weighted_aggregator import AggregatedResult, WeightedAggregator

logger = logging.getLogger(__name__)

# Domain importance weights for code review relevance
# Higher weight = more important for evaluating code quality in a review context
DOMAIN_WEIGHTS = {
    "backend": 1.0,  # Critical - server-side logic and API implementation
    "frontend": 1.0,  # Critical - UI components and client-side logic
    "testing": 0.9,  # Important - quality assurance and test coverage
    "database": 0.8,  # Important - data layer and migrations
    "infrastructure": 0.7,  # Medium - deployment and DevOps configurations
    "unknown": 0.5,  # Medium - unclear classification
    "configuration": 0.3,  # Low - usually simple key-value pairs
    "documentation": 0.2,  # Low - markdown content (easy to match patterns)
}


@dataclass
class EnhancedScorerConfig:
    """Configuration for the enhanced multi-dimensional code analyzer."""

    # Scoring weights (must sum to 1.0)
    architectural_weight: float = 0.25
    quality_weight: float = 0.25
    typescript_weight: float = 0.20
    framework_weight: float = 0.15
    domain_adherence_weight: float = 0.15

    # Analysis settings
    enable_architectural_analysis: bool = True
    enable_quality_analysis: bool = True
    enable_typescript_analysis: bool = True
    enable_framework_analysis: bool = True
    enable_domain_adherence_analysis: bool = True

    # Domain-aware analysis settings
    similarity_threshold: float = 0.3
    domain_confidence_threshold: float = 0.6
    max_similar_patterns: int = 10
    build_pattern_indices: bool = True
    model_name: str = "microsoft/graphcodebert-base"
    cache_dir: str | None = None
    fine_tuned_model_commit: str | None = None  # Commit hash of fine-tuned model to use

    # Output settings
    include_actionable_feedback: bool = True
    include_pattern_details: bool = True
    max_recommendations_per_file: int = 10

    # File filtering
    exclude_patterns: list[str] | None = None
    include_test_files: bool = False
    include_generated_files: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate weights sum to 1.0
        total_weight = (
            self.architectural_weight
            + self.quality_weight
            + self.typescript_weight
            + self.framework_weight
            + self.domain_adherence_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Ensure all weights are positive
        weights = [
            self.architectural_weight,
            self.quality_weight,
            self.typescript_weight,
            self.framework_weight,
            self.domain_adherence_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        # Set default exclude patterns
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__",
                ".git",
                "node_modules",
                ".venv",
                "dist",
                "build",
                ".next",
            ]


class MultiDimensionalScorer:
    """Main scorer that orchestrates multi-dimensional code analysis.

    This class provides comprehensive code quality analysis through:
    - Architectural pattern analysis
    - Code quality and best practices analysis
    - TypeScript usage analysis
    - Framework-specific pattern analysis
    """

    def __init__(
        self,
        config: EnhancedScorerConfig,
        repo_path: str = ".",
        device_manager: DeviceManager | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the MultiDimensionalScorer with configuration.

        Args:
            config: Configuration for the multi-dimensional scorer.
            repo_path: Path to the git repository (default: current directory).
            device_manager: DeviceManager for hardware acceleration (auto-created if None).
            progress_callback: Optional callback to report initialization progress.
        """
        logger.info("=== MULTIDIMENSIONAL SCORER INIT ===")
        logger.info("Starting MultiDimensionalScorer initialization")

        self.config = config
        self.repo_path = Path(repo_path)
        self.console = Console()
        self.progress_callback = progress_callback
        logger.info(f"Repository path: {self.repo_path}")

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if self.progress_callback:
                self.progress_callback(message)

        report_progress("Setting up repository connection...")
        try:
            logger.info("Initializing git repository...")
            self.repo = git.Repo(self.repo_path)
            logger.info("Git repository initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")
            raise

        report_progress("Initializing scoring components...")
        logger.info("Creating weighted aggregator...")
        self.aggregator = WeightedAggregator()
        logger.info("Weighted aggregator created")

        if device_manager is None:
            report_progress("Setting up hardware acceleration...")
            logger.info("Creating new DeviceManager (none provided)...")
            self.device_manager = DeviceManager()
            logger.info("New DeviceManager created")
        else:
            logger.info("Using provided DeviceManager")
            self.device_manager = device_manager

        # Initialize analyzers based on configuration
        logger.info("=== ANALYZER INITIALIZATION ===")
        self.analyzers: dict[str, BaseAnalyzer] = {}

        if config.enable_architectural_analysis:
            report_progress("Initializing architectural analyzer...")
            logger.info("Initializing ArchitecturalAnalyzer...")
            try:
                self.analyzers["architectural"] = ArchitecturalAnalyzer()
                logger.info("ArchitecturalAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ArchitecturalAnalyzer: {e}")
                raise

        if config.enable_quality_analysis:
            report_progress("Initializing code quality analyzer...")
            logger.info("Initializing QualityAnalyzer...")
            try:
                self.analyzers["quality"] = QualityAnalyzer()
                logger.info("QualityAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize QualityAnalyzer: {e}")
                raise

        if config.enable_typescript_analysis:
            report_progress("Initializing TypeScript analyzer...")
            logger.info("Initializing TypeScriptAnalyzer...")
            try:
                self.analyzers["typescript"] = TypeScriptAnalyzer()
                logger.info("TypeScriptAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize TypeScriptAnalyzer: {e}")
                raise

        if config.enable_framework_analysis:
            report_progress("Initializing framework analyzer...")
            logger.info("Initializing FrameworkAnalyzer...")
            try:
                self.analyzers["framework"] = FrameworkAnalyzer()
                logger.info("FrameworkAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FrameworkAnalyzer: {e}")
                raise

        if config.enable_domain_adherence_analysis:
            report_progress(
                "Initializing domain adherence analyzer (loading ML models)..."
            )
            logger.info("Initializing DomainAwareAdherenceAnalyzer...")
            logger.info("This is the most complex analyzer - involves model loading")

            domain_config = {
                "similarity_threshold": config.similarity_threshold,
                "domain_confidence_threshold": config.domain_confidence_threshold,
                "max_similar_patterns": config.max_similar_patterns,
                "model_name": config.model_name,
                "cache_dir": config.cache_dir,
                "fine_tuned_model_commit": config.fine_tuned_model_commit,
            }
            logger.info(f"Domain config: {domain_config}")

            def domain_progress_callback(message: str) -> None:
                """Nested progress callback for domain analyzer."""
                report_progress(f"Domain analyzer: {message}")

            try:
                logger.info(
                    "Creating DomainAwareAdherenceAnalyzer with device_manager..."
                )
                self.analyzers["domain_adherence"] = DomainAwareAdherenceAnalyzer(
                    domain_config,
                    device_manager=self.device_manager,
                    progress_callback=domain_progress_callback,
                )
                logger.info("DomainAwareAdherenceAnalyzer initialized successfully")
            except Exception as e:
                logger.error(
                    f"CRITICAL: Failed to initialize DomainAwareAdherenceAnalyzer: {e}"
                )
                logger.exception(
                    "Full traceback for DomainAwareAdherenceAnalyzer failure:"
                )
                raise

        report_progress("Analyzers initialized successfully!")
        logger.info(
            f"=== INITIALIZATION COMPLETE === Initialized MultiDimensionalScorer with {len(self.analyzers)} analyzers"
        )
        logger.info(f"Enabled analyzers: {list(self.analyzers.keys())}")

    def build_pattern_indices_from_codebase(self, target_commit: str = "HEAD") -> None:
        """Build pattern indices for domain-aware analysis from the entire codebase.

        Args:
            target_commit: Git commit to use as the codebase snapshot (default: HEAD)
        """
        if (
            "domain_adherence" not in self.analyzers
            or not self.config.build_pattern_indices
        ):
            logger.info(
                "Pattern index building disabled or domain adherence analyzer not enabled"
            )
            return

        logger.info(f"Building pattern indices from codebase at commit {target_commit}")

        try:
            # Get all files from the target commit
            commit = self.repo.commit(target_commit)
            codebase_files = {}

            for item in commit.tree.traverse():
                if (
                    hasattr(item, "type")
                    and hasattr(item, "path")
                    and hasattr(item, "data_stream")
                    and item.type == "blob"  # type: ignore[union-attr]
                ):  # It's a file, not a directory
                    file_path = str(item.path)  # type: ignore[union-attr]

                    # Skip files based on exclude patterns
                    if self._should_exclude_file(file_path):
                        continue

                    try:
                        file_content = item.data_stream.read().decode("utf-8")  # type: ignore[union-attr]
                        codebase_files[str(file_path)] = file_content
                    except (UnicodeDecodeError, Exception) as e:
                        logger.debug(f"Skipping file {file_path}: {e}")
                        continue

            logger.info(f"Extracted {len(codebase_files)} files from codebase")

            # Build pattern indices using the domain adherence analyzer
            domain_analyzer = self.analyzers["domain_adherence"]
            if isinstance(domain_analyzer, DomainAwareAdherenceAnalyzer):
                domain_analyzer.build_pattern_indices(
                    codebase_files, source_commit=target_commit
                )

        except Exception as e:
            logger.error(f"Failed to build pattern indices: {e}")

    def analyze_commit(
        self,
        commit_hash: str,
        pattern_index_commit: str = "parent",
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive multi-dimensional analysis of a commit.

        Args:
            commit_hash: Git commit hash to analyze
            pattern_index_commit: Commit to use for building pattern indices.
                - "parent" (default): Use the parent commit of commit_hash
                - Any valid commit ref: "HEAD", "main", commit hash, etc.
            progress_callback: Optional callback to report analysis progress

        Returns:
            Comprehensive analysis results with dimensional scores and recommendations
        """
        start_time = time.time()
        logger.info(f"Starting multi-dimensional analysis for commit {commit_hash}")
        logger.info(f"Pattern index commit specification: {pattern_index_commit}")

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if progress_callback:
                progress_callback(message)

        # Resolve pattern index commit
        resolved_index_commit = self._resolve_pattern_index_commit(
            commit_hash, pattern_index_commit
        )
        logger.info(f"Resolved pattern index commit: {resolved_index_commit}")

        report_progress("Extracting files from commit...")

        # Extract commit files using direct git integration
        commit_files = self._extract_commit_files(commit_hash)

        if not commit_files:
            raise ValueError(
                f"No files found in commit {commit_hash} or commit does not exist"
            )

        # Build pattern indices if needed for domain-aware analysis
        if (
            self.config.enable_domain_adherence_analysis
            and self.config.build_pattern_indices
            and "domain_adherence" in self.analyzers
        ):
            domain_analyzer = self.analyzers["domain_adherence"]
            if isinstance(domain_analyzer, DomainAwareAdherenceAnalyzer):
                # Check if we need to build/rebuild indices
                needs_rebuild = (
                    not domain_analyzer._indices_built  # No indices built yet
                    or domain_analyzer._indices_commit
                    != resolved_index_commit  # Different commit
                )

                if needs_rebuild:
                    if domain_analyzer._indices_commit:
                        logger.info(
                            f"Rebuilding indices: previous commit was {domain_analyzer._indices_commit}, "
                            f"new commit is {resolved_index_commit}"
                        )
                    report_progress(
                        "Building pattern indices for domain-aware analysis..."
                    )
                    logger.info(
                        f"Building pattern indices from commit: {resolved_index_commit}"
                    )
                    self.build_pattern_indices_from_codebase(
                        target_commit=resolved_index_commit
                    )
                else:
                    logger.info(
                        f"Reusing existing pattern indices from commit: {resolved_index_commit}"
                    )

        # Run multi-dimensional analysis
        report_progress("Running multi-dimensional analysis...")
        dimensional_results = self._analyze_commit_dimensions(
            commit_files, progress_callback
        )

        # Aggregate all results
        report_progress("Aggregating results...")
        aggregated_results = self._aggregate_results(dimensional_results)

        # Generate enhanced output
        enhanced_results = self._create_enhanced_output(
            dimensional_results, aggregated_results
        )

        enhanced_results["processing_time"] = time.time() - start_time
        enhanced_results["commit_hash"] = commit_hash
        enhanced_results["config"] = {
            "weights": {
                "architectural": self.config.architectural_weight,
                "quality": self.config.quality_weight,
                "typescript": self.config.typescript_weight,
                "framework": self.config.framework_weight,
                "domain_adherence": self.config.domain_adherence_weight,
            },
            "analyzers_enabled": list(self.analyzers.keys()),
        }

        return enhanced_results

    def analyze_files(self, files: dict[str, str]) -> dict[str, Any]:
        """Analyze a set of files without git commit context.

        Args:
            files: Dictionary mapping file paths to their content

        Returns:
            Analysis results for the provided files
        """
        start_time = time.time()
        logger.info(f"Starting multi-dimensional analysis for {len(files)} files")

        # Run multi-dimensional analysis
        dimensional_results = self._analyze_commit_dimensions(files)

        # Aggregate results
        aggregated_results = self._aggregate_results(dimensional_results)

        # Generate output
        enhanced_results = self._create_enhanced_output(
            dimensional_results, aggregated_results
        )

        enhanced_results["processing_time"] = time.time() - start_time
        enhanced_results["files_analyzed"] = list(files.keys())

        return enhanced_results

    def _extract_commit_files(self, commit_hash: str) -> dict[str, str]:
        """Extract files changed in a commit using GitPython.

        For added files: extracts entire file content.
        For modified files: extracts only the changed lines.
        """
        try:
            commit = self.repo.commit(commit_hash)

            # Get the parent commit for comparison
            if commit.parents:
                parent = commit.parents[0]
                # Get diff between commit and parent
                diff = parent.diff(commit, create_patch=True)
            else:
                # First commit - compare against empty tree
                diff = commit.diff(git.NULL_TREE, create_patch=True)

            commit_files = {}

            for item in diff:
                if item.b_path is None:
                    continue

                file_path = item.b_path

                # Skip files based on exclude patterns
                if self._should_exclude_file(file_path):
                    continue

                try:
                    if item.new_file:
                        # Added file - analyze entire content
                        file_content = (
                            (commit.tree / file_path).data_stream.read().decode("utf-8")
                        )
                        commit_files[file_path] = file_content
                        logger.debug(
                            f"Extracted full content for added file: {file_path}"
                        )

                    elif not item.deleted_file:
                        # Modified file - extract only changed lines
                        changed_content = self._extract_changed_lines(item)
                        if changed_content.strip():
                            commit_files[file_path] = changed_content
                            logger.debug(
                                f"Extracted {len(changed_content.splitlines())} changed lines from: {file_path}"
                            )

                except (UnicodeDecodeError, Exception) as e:
                    logger.warning(f"Skipping file {file_path}: {e}")
                    continue

            logger.info(
                f"Extracted {len(commit_files)} files from commit {commit_hash}"
            )
            return commit_files

        except Exception as e:
            logger.error(f"Failed to extract commit files for {commit_hash}: {e}")
            raise ValueError(f"Invalid commit hash {commit_hash}: {e}") from e

    def _extract_changed_lines(self, diff_item: Any) -> str:
        """Extract only added/modified lines from a git diff item.

        Args:
            diff_item: GitPython diff item with patch information

        Returns:
            String containing only the added/modified lines
        """
        try:
            # Get the unified diff patch
            if diff_item.diff is None:
                return ""

            diff_text = diff_item.diff.decode("utf-8")
            added_lines = []

            for line in diff_text.split("\n"):
                # Lines starting with '+' are additions (but skip '+++ b/filename' header)
                if line.startswith("+") and not line.startswith("+++"):
                    # Remove the '+' prefix and add the actual line
                    added_lines.append(line[1:])

            return "\n".join(added_lines)

        except Exception as e:
            logger.warning(f"Failed to extract changed lines: {e}")
            return ""

    def _resolve_pattern_index_commit(
        self, commit_hash: str, pattern_index_commit: str
    ) -> str:
        """Resolve the pattern index commit reference to an actual commit hash.

        Args:
            commit_hash: The commit being analyzed
            pattern_index_commit: The pattern index commit specification

        Returns:
            Resolved commit hash for building pattern indices

        Raises:
            ValueError: If the commit reference is invalid
        """
        if pattern_index_commit == "parent":
            # Resolve to parent commit of the analyzed commit
            try:
                commit = self.repo.commit(commit_hash)
                if commit.parents:
                    resolved = commit.parents[0].hexsha
                    logger.info(
                        f"Resolved 'parent' to commit {resolved} (parent of {commit_hash})"
                    )
                    return resolved
                else:
                    # First commit has no parent - fall back to analyzing against empty tree
                    logger.warning(
                        f"Commit {commit_hash} has no parent (first commit). "
                        "Falling back to HEAD for pattern indices."
                    )
                    return "HEAD"
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve parent commit for {commit_hash}: {e}"
                ) from e
        else:
            # Use the provided commit reference as-is (could be "HEAD", "main", hash, etc.)
            try:
                # Validate that the commit exists
                resolved_commit = self.repo.commit(pattern_index_commit)
                logger.info(
                    f"Using explicit pattern index commit: {resolved_commit.hexsha}"
                )
                return pattern_index_commit
            except Exception as e:
                raise ValueError(
                    f"Invalid pattern index commit '{pattern_index_commit}': {e}"
                ) from e

    def _should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded from analysis."""
        # Check exclude patterns
        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if pattern in file_path:
                    return True

        # Exclude test files if configured
        if not self.config.include_test_files:
            test_patterns = ["test_", "_test.", ".test.", "spec_", "_spec.", ".spec."]
            if any(pattern in file_path.lower() for pattern in test_patterns):
                return True

        # Exclude generated files if configured
        if not self.config.include_generated_files:
            generated_patterns = [".generated.", "_generated.", ".d.ts"]
            if any(pattern in file_path for pattern in generated_patterns):
                return True

        return False

    def _analyze_commit_dimensions(
        self,
        commit_files: dict[str, str],
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, dict[str, AnalysisResult]]:
        """Run multi-dimensional analysis on commit files.

        Args:
            commit_files: Dictionary mapping file paths to their content
            progress_callback: Optional callback to report analysis progress

        Returns:
            Dictionary mapping analyzer names to their results
        """
        dimensional_results = {}

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if progress_callback:
                progress_callback(message)

        for analyzer_name, analyzer in self.analyzers.items():
            # Report current analyzer progress
            report_progress(
                f"Analyzing with {analyzer_name} ({len(commit_files)} files)..."
            )

            logger.debug(f"Running {analyzer_name} analysis")
            try:
                results = analyzer.analyze_commit(commit_files)
                dimensional_results[analyzer_name] = results
                logger.debug(f"{analyzer_name} analyzed {len(results)} files")
            except Exception as e:
                logger.error(f"Error in {analyzer_name} analysis: {e}")
                dimensional_results[analyzer_name] = {}

        return dimensional_results

    def _aggregate_results(
        self, dimensional_results: dict[str, dict[str, AnalysisResult]]
    ) -> AggregatedResult:
        """Aggregate multi-dimensional results using weighted scoring.

        Args:
            dimensional_results: Results from multi-dimensional analysis

        Returns:
            Aggregated results with overall scores
        """
        # Prepare weights
        weights = {
            "architectural": self.config.architectural_weight,
            "quality": self.config.quality_weight,
            "typescript": self.config.typescript_weight,
            "framework": self.config.framework_weight,
            "domain_adherence": self.config.domain_adherence_weight,
        }

        # Calculate average scores for each dimension
        # Filter out unknown domain files to avoid skewing results
        dimensional_scores = {}
        for analyzer_name, results in dimensional_results.items():
            if results:  # Only include if we have results
                # Filter out files with unknown domain (but only if there are also known-domain files)
                valid_results = {}
                excluded_files = []

                for file_path, result in results.items():
                    file_domain = result.metrics.get("domain", "unknown")
                    if file_domain == "unknown":
                        excluded_files.append((file_path, result))
                        logger.debug(
                            f"Potential exclusion - unknown domain file: {file_path} (score: {result.score:.2f})"
                        )
                    else:
                        valid_results[file_path] = result

                # Only exclude unknown files if there are ALSO known-domain files
                # If ALL files are unknown, include them (better than no data)
                if valid_results:
                    # We have some known-domain files, exclude the unknown ones
                    if excluded_files:
                        logger.info(
                            f"{analyzer_name}: Excluded {len(excluded_files)} unknown domain files from score: {[f[0] for f in excluded_files]}"
                        )
                    file_scores = [result.score for result in valid_results.values()]
                    dimensional_scores[analyzer_name] = (
                        sum(file_scores) / len(file_scores) if file_scores else 0.0
                    )
                    logger.info(
                        f"{analyzer_name}: Averaged {len(valid_results)} files (excluded {len(excluded_files)}), score: {dimensional_scores[analyzer_name]:.4f}"
                    )
                else:
                    # All files are unknown domain - include them rather than having no data
                    logger.warning(
                        f"{analyzer_name}: All files are unknown domain, including them in average"
                    )
                    file_scores = [result.score for _, result in results.items()]
                    dimensional_scores[analyzer_name] = (
                        sum(file_scores) / len(file_scores) if file_scores else 0.0
                    )

        # Aggregate using the weighted aggregator
        aggregated = self.aggregator.aggregate(dimensional_scores, weights)

        # Calculate domain-weighted score for code-focused evaluation
        # This gives higher weight to implementation domains (backend/frontend/testing)
        # and lower weight to documentation/configuration

        # First, check if we have any known-domain files
        has_known_domains = False
        for _analyzer_name, results in dimensional_results.items():
            for _file_path, result in results.items():
                if result.metrics.get("domain", "unknown") != "unknown":
                    has_known_domains = True
                    break
            if has_known_domains:
                break

        # Calculate weighted scores
        domain_weighted_scores = []
        domain_weights_used = []

        for _analyzer_name, results in dimensional_results.items():
            for _file_path, result in results.items():
                file_domain = result.metrics.get("domain", "unknown")
                domain_weight = DOMAIN_WEIGHTS.get(file_domain, 0.5)

                # Skip unknown domain files if there are known-domain files
                if file_domain == "unknown" and has_known_domains:
                    continue

                domain_weighted_scores.append(result.score * domain_weight)
                domain_weights_used.append(domain_weight)

        # Calculate weighted average
        if domain_weights_used:
            code_focused_score = sum(domain_weighted_scores) / sum(domain_weights_used)
            logger.info(
                f"Code-focused score (domain-weighted): {code_focused_score:.4f} "
                f"(vs simple average: {aggregated.overall_score:.4f})"
            )
        else:
            code_focused_score = aggregated.overall_score

        # Add code-focused score to aggregated result (monkey-patch for now)
        aggregated.code_focused_score = code_focused_score  # type: ignore[attr-defined]

        return aggregated

    def _create_enhanced_output(
        self,
        dimensional_results: dict[str, dict[str, AnalysisResult]],
        aggregated_results: AggregatedResult,
    ) -> dict[str, Any]:
        """Create enhanced output format with actionable insights.

        Args:
            dimensional_results: Multi-dimensional analysis results
            aggregated_results: Aggregated scoring results

        Returns:
            Enhanced output with all analysis results
        """
        enhanced_output = {
            "overall_adherence": aggregated_results.overall_score,
            "code_focused_score": getattr(
                aggregated_results,
                "code_focused_score",
                aggregated_results.overall_score,
            ),
            "dimensional_scores": aggregated_results.dimensional_scores,
            "confidence": aggregated_results.confidence,
        }

        # Aggregate patterns and recommendations across all files
        all_patterns: list[PatternMatch] = []
        all_recommendations: list[Recommendation] = []
        file_level_results: dict[str, dict[str, Any]] = {}

        # First pass: collect all results and identify unknown domain files
        unknown_domain_files = set()
        known_domain_files = set()

        for _analyzer_name, analyzer_results in dimensional_results.items():
            for file_path, result in analyzer_results.items():
                file_domain = result.metrics.get("domain", "unknown")
                if file_domain == "unknown":
                    unknown_domain_files.add(file_path)
                else:
                    known_domain_files.add(file_path)

        # Decide whether to exclude unknown files
        # Only exclude if there are ALSO known-domain files (mixed case)
        exclude_unknown = len(known_domain_files) > 0

        # Second pass: collect patterns and recommendations
        for analyzer_name, analyzer_results in dimensional_results.items():
            for file_path, result in analyzer_results.items():
                # Initialize file entry if not exists
                if file_path not in file_level_results:
                    file_level_results[file_path] = {
                        "scores": {},
                        "patterns_count": 0,
                        "recommendations_count": 0,
                    }

                # Add analyzer score for this file
                file_level_results[file_path]["scores"][analyzer_name] = result.score

                # Copy embedding_data from metrics if present (for explain command)
                if "embedding_data" in result.metrics:
                    file_level_results[file_path]["embedding_data"] = result.metrics[
                        "embedding_data"
                    ]

                # Include patterns and recommendations based on exclusion decision
                file_domain = result.metrics.get("domain", "unknown")
                if file_domain != "unknown" or not exclude_unknown:
                    all_patterns.extend(result.patterns_found)
                    all_recommendations.extend(result.recommendations)
                elif exclude_unknown:
                    logger.debug(
                        f"Skipping recommendations from unknown domain file: {file_path}"
                    )

                file_level_results[file_path]["patterns_count"] += len(
                    result.patterns_found
                )
                file_level_results[file_path]["recommendations_count"] += len(
                    result.recommendations
                )

        # Add pattern analysis summary
        enhanced_output["pattern_analysis"] = {
            "total_patterns_found": len(all_patterns),
            "patterns_by_type": self._group_patterns_by_type(all_patterns),
            "pattern_confidence_avg": self._calculate_average_confidence(all_patterns),
        }

        # Add domain-stratified statistics
        domain_breakdown = self._calculate_domain_statistics(dimensional_results)
        enhanced_output["domain_breakdown"] = domain_breakdown

        # Add actionable feedback
        if self.config.include_actionable_feedback:
            enhanced_output["actionable_feedback"] = self._generate_actionable_feedback(
                all_recommendations
            )

        # Add file-level breakdown
        enhanced_output["file_level_analysis"] = file_level_results

        # Add detailed pattern information if requested
        if self.config.include_pattern_details:
            enhanced_output["detailed_patterns"] = self._format_detailed_patterns(
                all_patterns
            )

        # Add analysis metadata
        enhanced_output["analysis_metadata"] = {
            "analyzers_used": list(self.analyzers.keys()),
            "total_files_analyzed": len(file_level_results),
            "config_weights": {
                "architectural": self.config.architectural_weight,
                "quality": self.config.quality_weight,
                "typescript": self.config.typescript_weight,
                "framework": self.config.framework_weight,
            },
        }

        return enhanced_output

    def _group_patterns_by_type(self, patterns: list[PatternMatch]) -> dict[str, int]:
        """Group patterns by their type."""
        pattern_counts: dict[str, int] = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        return pattern_counts

    def _calculate_average_confidence(self, patterns: list[PatternMatch]) -> float:
        """Calculate average confidence of patterns."""
        if not patterns:
            return 0.0
        return sum(pattern.confidence for pattern in patterns) / len(patterns)

    def _calculate_domain_statistics(
        self, dimensional_results: dict[str, dict[str, AnalysisResult]]
    ) -> dict[str, Any]:
        """Calculate statistics grouped by architectural domain.

        Args:
            dimensional_results: Results from multi-dimensional analysis

        Returns:
            Dictionary mapping domain names to their statistics
        """
        domain_stats: dict[str, dict[str, Any]] = {}

        # Collect files and scores by domain
        for _analyzer_name, analyzer_results in dimensional_results.items():
            for file_path, result in analyzer_results.items():
                domain = result.metrics.get("domain", "unknown")

                if domain not in domain_stats:
                    domain_stats[domain] = {
                        "files": [],
                        "scores": [],
                        "file_count": 0,
                    }

                domain_stats[domain]["files"].append(file_path)
                domain_stats[domain]["scores"].append(result.score)
                domain_stats[domain]["file_count"] += 1

        # Calculate average scores and confidence per domain
        for _domain, stats in domain_stats.items():
            if stats["scores"]:
                stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
                stats["min_score"] = min(stats["scores"])
                stats["max_score"] = max(stats["scores"])
            else:
                stats["avg_score"] = 0.0
                stats["min_score"] = 0.0
                stats["max_score"] = 0.0

        return domain_stats

    def _generate_actionable_feedback(
        self, recommendations: list[Recommendation]
    ) -> list[dict[str, Any]]:
        """Generate actionable feedback from recommendations.

        Args:
            recommendations: List of recommendations from all analyzers

        Returns:
            List of prioritized, actionable feedback items
        """
        if not recommendations:
            return []

        # Sort recommendations by severity and category
        severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}

        sorted_recommendations = sorted(
            recommendations,
            key=lambda r: (severity_order.get(r.severity.value, 4), r.category),
        )

        # Limit recommendations per file
        file_recommendation_count: dict[str, int] = {}
        filtered_recommendations = []

        for rec in sorted_recommendations:
            file_count = file_recommendation_count.get(rec.file_path, 0)
            if file_count < self.config.max_recommendations_per_file:
                filtered_recommendations.append(rec)
                file_recommendation_count[rec.file_path] = file_count + 1

        # Format for output
        actionable_feedback = []
        for rec in filtered_recommendations:
            actionable_feedback.append(
                {
                    "severity": rec.severity.value,
                    "category": rec.category,
                    "message": rec.message,
                    "file": rec.file_path,
                    "line": rec.line_number,
                    "suggested_fix": rec.suggested_fix,
                    "rule_id": rec.rule_id,
                }
            )

        return actionable_feedback

    def _format_detailed_patterns(
        self, patterns: list[PatternMatch]
    ) -> list[dict[str, Any]]:
        """Format patterns for detailed output."""
        detailed_patterns = []
        for pattern in patterns:
            detailed_patterns.append(
                {
                    "type": pattern.pattern_type.value,
                    "name": pattern.pattern_name,
                    "file": pattern.file_path,
                    "line": pattern.line_number,
                    "confidence": pattern.confidence,
                    "context": pattern.context,
                }
            )
        return detailed_patterns

    def save_results(
        self, results: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Save analysis results to file.

        Args:
            results: Analysis results to save
            output_file: Output file path (required - relative or absolute)

        Returns:
            Path where results were saved, or empty string if not saved
        """
        if output_file is None:
            logger.error("save_results called without output_file - not saving")
            return ""

        # Use the path exactly as specified by user (relative or absolute)
        output_path = Path(output_file)

        # Create parent directory if needed
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Analysis results saved to {output_path}")
        return str(output_path)

    def get_commit_info(self, commit_hash: str) -> dict[str, Any]:
        """Get basic information about a commit.

        Args:
            commit_hash: Git commit hash

        Returns:
            Dictionary with commit information
        """
        try:
            commit = self.repo.commit(commit_hash)
            return {
                "hash": commit_hash,
                "message": commit.message.strip(),
                "author": str(commit.author),
                "timestamp": commit.committed_date,
                "files_changed": [
                    item.b_path
                    for item in commit.diff(
                        commit.parents[0] if commit.parents else git.NULL_TREE
                    )
                    if item.b_path is not None
                ],
                "insertions": commit.stats.total["insertions"],
                "deletions": commit.stats.total["deletions"],
            }
        except Exception as e:
            logger.error(f"Failed to get commit info for {commit_hash}: {e}")
            return {}

    def compare_commits(
        self, base_commit: str, compare_commits: list[str]
    ) -> dict[str, Any]:
        """Compare multiple commits against a base implementation.

        Args:
            base_commit: Base commit to compare against
            compare_commits: List of commits to compare

        Returns:
            Comparison results with relative scoring
        """
        all_results = {}

        # Analyze base commit
        logger.info(f"Analyzing base commit: {base_commit}")
        all_results["base"] = self.analyze_commit(base_commit)

        # Analyze comparison commits
        for commit in compare_commits:
            logger.info(f"Analyzing comparison commit: {commit}")
            try:
                all_results[commit] = self.analyze_commit(commit)
            except Exception as e:
                logger.warning(f"Failed to analyze {commit}: {e}")

        # Generate comparison insights
        comparison_results = {
            "base_commit": base_commit,
            "comparison_commits": compare_commits,
            "individual_results": all_results,
            "comparison_summary": self._generate_comparison_summary(all_results),
        }

        return comparison_results

    def _generate_comparison_summary(
        self, all_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate summary of comparison between commits."""
        if "base" not in all_results:
            return {}

        base_score = all_results["base"]["overall_adherence"]
        base_dimensions = all_results["base"]["dimensional_scores"]

        summary = {"base_score": base_score, "comparisons": []}

        for commit_id, result in all_results.items():
            if commit_id != "base":
                commit_score = result["overall_adherence"]
                commit_dimensions = result["dimensional_scores"]

                improvement = base_score - commit_score
                percentage = (
                    (improvement / commit_score) * 100 if commit_score > 0 else 0
                )

                dimensional_improvements = {}
                for dim, base_dim_score in base_dimensions.items():
                    commit_dim_score = commit_dimensions.get(dim, 0)
                    if commit_dim_score > 0:
                        dim_improvement = (
                            (base_dim_score - commit_dim_score) / commit_dim_score * 100
                        )
                        dimensional_improvements[dim] = dim_improvement

                summary["comparisons"].append(
                    {
                        "commit": commit_id,
                        "score": commit_score,
                        "improvement": improvement,
                        "improvement_percentage": percentage,
                        "dimensional_improvements": dimensional_improvements,
                    }
                )

        return summary

    def get_hardware_fallback_report(self) -> dict[str, Any]:
        """Get hardware fallback report from analyzers for user notification.

        Returns:
            Dictionary with fallback statistics from all analyzers.
        """
        fallback_report: dict[str, Any] = {
            "has_any_fallbacks": False,
            "analyzer_reports": {},
            "summary_message": None,
            "performance_impact": None,
            "suggestions": [],
        }

        # Check domain adherence analyzer (most likely to have MPS issues)
        if "domain_adherence" in self.analyzers:
            analyzer = self.analyzers["domain_adherence"]
            # Check if it has a pattern_indexer attribute with fallback reporting
            if hasattr(analyzer, "pattern_indexer") and analyzer.pattern_indexer:
                pattern_indexer = analyzer.pattern_indexer
                if hasattr(pattern_indexer, "get_fallback_report"):
                    report = pattern_indexer.get_fallback_report()
                    fallback_report["analyzer_reports"]["domain_adherence"] = report

                    if report["has_fallbacks"]:
                        fallback_report["has_any_fallbacks"] = True
                        fallback_report["summary_message"] = report["user_message"]
                        fallback_report["performance_impact"] = report[
                            "performance_impact"
                        ]
                        fallback_report["suggestions"].extend(report["suggestions"])

        return fallback_report
