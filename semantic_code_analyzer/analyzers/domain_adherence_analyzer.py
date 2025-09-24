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

"""Domain-aware adherence analyzer for AI-generated code quality measurement.

This analyzer measures how well AI-generated code changes adhere to existing
repository patterns within specific architectural domains using GraphCodeBERT
embeddings and similarity-based pattern matching.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..embeddings import PatternIndexer, SimilarityMatch
else:
    try:
        from ..embeddings import PatternIndexer, SimilarityMatch
    except ImportError:
        PatternIndexer = None
        SimilarityMatch = None
from .base_analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    PatternMatch,
    PatternType,
    Recommendation,
    Severity,
)
from .domain_classifier import (
    ArchitecturalDomain,
    DomainClassificationResult,
    DomainClassifier,
)

logger = logging.getLogger(__name__)


@dataclass
class AdherenceScore:
    """Represents adherence measurement results for a code change."""

    overall_adherence: float  # 0.0 to 1.0
    domain_adherence: float  # Domain-specific adherence score
    pattern_similarity: float  # Average similarity to existing patterns
    confidence: float  # Confidence in the measurement
    similar_patterns_count: int  # Number of similar patterns found
    domain_match_quality: float  # How well the code fits its classified domain


@dataclass
class AdherenceAnalysisResult:
    """Extended analysis result with domain-aware adherence metrics."""

    base_result: AnalysisResult
    domain_classification: DomainClassificationResult
    adherence_score: AdherenceScore
    similar_patterns: list[SimilarityMatch]
    improvement_suggestions: list[str]


class DomainAwareAdherenceAnalyzer(BaseAnalyzer):
    """Main analyzer for measuring AI-generated code adherence to domain-specific patterns.

    This analyzer:
    1. Classifies code into architectural domains
    2. Searches for similar patterns within the domain
    3. Measures adherence based on similarity to existing high-quality patterns
    4. Provides actionable recommendations for improvement
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device_manager: Any | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Initialize the DomainAwareAdherenceAnalyzer.

        Args:
            config: Optional configuration dictionary for the analyzer.
            device_manager: DeviceManager for hardware acceleration (optional).
            progress_callback: Optional callback to report initialization progress.
        """
        logger.info("=== DOMAIN ADHERENCE ANALYZER INIT ===")
        logger.info("Starting DomainAwareAdherenceAnalyzer initialization")

        # Store progress callback for reporting
        self.progress_callback = progress_callback

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if self.progress_callback:
                self.progress_callback(message)

        report_progress("Initializing base analyzer...")
        try:
            logger.info("Calling super().__init__(config)...")
            super().__init__(config)
            logger.info("Parent BaseAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize parent BaseAnalyzer: {e}")
            raise

        # Initialize components
        report_progress("Initializing domain classifier...")
        logger.info("Initializing DomainClassifier...")
        try:
            self.domain_classifier = DomainClassifier(config)
            logger.info("DomainClassifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DomainClassifier: {e}")
            raise

        # Handle config safely
        config = config or {}
        logger.info(f"Using config: {config}")

        # Initialize pattern indexer if dependencies are available
        report_progress("Checking ML model dependencies...")
        logger.info("Checking PatternIndexer dependencies...")
        if PatternIndexer is not None:
            report_progress("Loading GraphCodeBERT model (this may take a while)...")
            logger.info("PatternIndexer dependencies available - initializing...")
            logger.info("WARNING: This is where the segfault likely occurs!")

            model_name = config.get("model_name", "microsoft/graphcodebert-base")
            cache_dir = config.get("cache_dir")
            logger.info(f"Model name: {model_name}")
            logger.info(f"Cache dir: {cache_dir}")
            logger.info(f"Device manager provided: {device_manager is not None}")

            def pattern_indexer_progress_callback(message: str) -> None:
                """Nested progress callback for pattern indexer."""
                report_progress(f"Model loading: {message}")

            try:
                logger.info(
                    "Creating PatternIndexer - this may load heavy ML models..."
                )
                self.pattern_indexer = PatternIndexer(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    device_manager=device_manager,
                    progress_callback=pattern_indexer_progress_callback,
                )
                logger.info("PatternIndexer initialized successfully!")
            except Exception as e:
                logger.error(f"CRITICAL: PatternIndexer initialization failed: {e}")
                logger.exception("Full traceback for PatternIndexer failure:")
                raise
        else:
            report_progress(
                "ML model dependencies not available, using fallback mode..."
            )
            logger.warning(
                "PatternIndexer dependencies not available. Similarity search disabled."
            )
            self.pattern_indexer = None

        # Configuration parameters
        report_progress("Finalizing analyzer configuration...")
        logger.info("Setting configuration parameters...")
        self.similarity_threshold = config.get("similarity_threshold", 0.3)
        self.min_patterns_for_analysis = config.get("min_patterns_for_analysis", 3)
        self.max_similar_patterns = config.get("max_similar_patterns", 10)
        self.domain_confidence_threshold = config.get(
            "domain_confidence_threshold", 0.6
        )
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Min patterns for analysis: {self.min_patterns_for_analysis}")
        logger.info(f"Max similar patterns: {self.max_similar_patterns}")
        logger.info(f"Domain confidence threshold: {self.domain_confidence_threshold}")

        # Track if indices have been built
        self._indices_built: set[str] = set()

        report_progress("Domain adherence analyzer ready!")
        logger.info("=== DOMAIN ADHERENCE ANALYZER INITIALIZATION COMPLETE ===")

        logger.info("=== DOMAIN ADHERENCE ANALYZER INIT COMPLETE ===")
        logger.info("DomainAwareAdherenceAnalyzer initialized successfully!")

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'domain_adherence'.
        """
        return "domain_adherence"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.25 for 25% of overall score).
        """
        return 0.25

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a file for domain-aware adherence patterns.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with domain-aware adherence analysis
        """
        start_time = time.time()

        # Perform domain classification
        domain_classification = self.domain_classifier.classify_domain(
            file_path, content
        )

        # Get adherence analysis
        adherence_analysis = self.analyze_domain_adherence(
            content, domain_classification, file_path
        )

        # Create pattern matches based on analysis
        patterns_found = self._create_adherence_patterns(adherence_analysis, file_path)

        # Generate recommendations
        recommendations = self._generate_adherence_recommendations(
            adherence_analysis, file_path
        )

        # Calculate overall score
        score = self._calculate_adherence_score(adherence_analysis)

        # Collect comprehensive metrics
        metrics = {
            "domain": adherence_analysis.domain_classification.domain.value,
            "domain_confidence": adherence_analysis.domain_classification.confidence,
            "overall_adherence": adherence_analysis.adherence_score.overall_adherence,
            "domain_adherence": adherence_analysis.adherence_score.domain_adherence,
            "pattern_similarity": adherence_analysis.adherence_score.pattern_similarity,
            "confidence": adherence_analysis.adherence_score.confidence,
            "similar_patterns_found": adherence_analysis.adherence_score.similar_patterns_count,
            "domain_match_quality": adherence_analysis.adherence_score.domain_match_quality,
        }

        analysis_time = time.time() - start_time

        return AnalysisResult(
            file_path=file_path,
            score=score,
            patterns_found=patterns_found,
            recommendations=recommendations,
            metrics=metrics,
            analysis_time=analysis_time,
        )

    def analyze_domain_adherence(
        self,
        code_content: str,
        domain_classification: DomainClassificationResult,
        file_path: str = "",
    ) -> AdherenceAnalysisResult:
        """Perform comprehensive domain-aware adherence analysis.

        Args:
            code_content: Code content to analyze
            domain_classification: Domain classification result
            file_path: Optional file path for context

        Returns:
            AdherenceAnalysisResult with detailed analysis
        """
        domain = domain_classification.domain
        domain_str = domain.value

        # Ensure we have a pattern index for this domain
        if (
            domain_str not in self._indices_built
            and domain != ArchitecturalDomain.UNKNOWN
        ):
            logger.warning(
                f"No pattern index found for domain {domain_str}. Adherence analysis limited."
            )

        # Search for similar patterns
        similar_patterns = []
        if (
            self.pattern_indexer is not None
            and domain_str in self.pattern_indexer.domain_indices
        ):
            similar_patterns = self.pattern_indexer.search_similar_patterns(
                query_code=code_content,
                domain=domain_str,
                top_k=self.max_similar_patterns,
                min_similarity=self.similarity_threshold,
            )

        # Calculate adherence scores
        adherence_score = self._calculate_detailed_adherence_scores(
            code_content, domain_classification, similar_patterns
        )

        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            domain_classification, similar_patterns, adherence_score
        )

        # Create base analysis result for compatibility
        base_result = AnalysisResult(
            file_path=file_path,
            score=adherence_score.overall_adherence,
            patterns_found=[],
            recommendations=[],
            metrics={},
            analysis_time=0.0,
        )

        return AdherenceAnalysisResult(
            base_result=base_result,
            domain_classification=domain_classification,
            adherence_score=adherence_score,
            similar_patterns=similar_patterns,
            improvement_suggestions=improvement_suggestions,
        )

    def build_pattern_indices(self, codebase_files: dict[str, str]) -> None:
        """Build pattern indices for all domains from a codebase.

        Args:
            codebase_files: Dictionary mapping file paths to their content
        """
        if self.pattern_indexer is None:
            logger.warning("Pattern indexer not available. Skipping index building.")
            return

        logger.info(f"Building pattern indices for {len(codebase_files)} files")

        # Classify all files by domain
        domain_files: dict[str, dict[str, str]] = {}

        for file_path, content in codebase_files.items():
            try:
                classification = self.domain_classifier.classify_domain(
                    file_path, content
                )
                domain_str = classification.domain.value

                if domain_str not in domain_files:
                    domain_files[domain_str] = {}

                domain_files[domain_str][file_path] = content

            except Exception as e:
                logger.warning(f"Failed to classify {file_path}: {e}")
                continue

        # Build indices for each domain
        for domain_str, files in domain_files.items():
            if (
                domain_str == ArchitecturalDomain.UNKNOWN.value
                or len(files) < self.min_patterns_for_analysis
            ):
                logger.info(
                    f"Skipping index building for domain {domain_str} ({len(files)} files)"
                )
                continue

            try:
                self.pattern_indexer.build_domain_index(
                    domain=domain_str,
                    codebase_files=files,
                    max_files=self.config.get("max_files_per_domain", 1000),
                )
                self._indices_built.add(domain_str)
                logger.info(
                    f"Built pattern index for domain {domain_str} with {len(files)} files"
                )

            except Exception as e:
                logger.error(f"Failed to build index for domain {domain_str}: {e}")

        logger.info(
            f"Successfully built indices for domains: {list(self._indices_built)}"
        )

    def _calculate_detailed_adherence_scores(
        self,
        code_content: str,
        domain_classification: DomainClassificationResult,
        similar_patterns: list[SimilarityMatch],
    ) -> AdherenceScore:
        """Calculate detailed adherence scores based on similarity analysis.

        Args:
            code_content: Code content being analyzed
            domain_classification: Domain classification result
            similar_patterns: Similar patterns found in the domain

        Returns:
            AdherenceScore with detailed metrics
        """
        # Domain match quality based on classification confidence
        domain_match_quality = domain_classification.confidence

        # Pattern similarity based on similar patterns found
        if similar_patterns:
            similarity_scores = [match.similarity_score for match in similar_patterns]
            pattern_similarity = statistics.mean(similarity_scores)
            max_similarity = max(similarity_scores)
        else:
            pattern_similarity = 0.0
            max_similarity = 0.0

        # Domain adherence calculation
        if domain_classification.domain == ArchitecturalDomain.UNKNOWN:
            domain_adherence = 0.3  # Neutral score for unknown domains
        else:
            # Weight by domain confidence and pattern similarity
            domain_adherence = (domain_match_quality * 0.6) + (pattern_similarity * 0.4)

        # Overall adherence combines multiple factors
        overall_adherence = self._calculate_weighted_adherence(
            domain_match_quality, pattern_similarity, len(similar_patterns)
        )

        # Confidence calculation
        confidence_factors = [
            domain_classification.confidence,
            min(1.0, len(similar_patterns) / 5.0),  # More patterns = higher confidence
            max_similarity,
        ]
        confidence = statistics.mean(confidence_factors)

        return AdherenceScore(
            overall_adherence=overall_adherence,
            domain_adherence=domain_adherence,
            pattern_similarity=pattern_similarity,
            confidence=confidence,
            similar_patterns_count=len(similar_patterns),
            domain_match_quality=domain_match_quality,
        )

    def _calculate_weighted_adherence(
        self, domain_quality: float, pattern_similarity: float, pattern_count: int
    ) -> float:
        """Calculate weighted overall adherence score.

        Args:
            domain_quality: Quality of domain classification
            pattern_similarity: Average pattern similarity
            pattern_count: Number of similar patterns found

        Returns:
            Overall adherence score (0.0 to 1.0)
        """
        # Base weights
        domain_weight = 0.3
        similarity_weight = 0.5
        coverage_weight = 0.2

        # Coverage bonus based on number of patterns
        coverage_score = min(1.0, pattern_count / 10.0)

        # Weighted combination
        overall_score = (
            domain_quality * domain_weight
            + pattern_similarity * similarity_weight
            + coverage_score * coverage_weight
        )

        return max(0.0, min(1.0, overall_score))

    def _create_adherence_patterns(
        self, analysis: AdherenceAnalysisResult, file_path: str
    ) -> list[PatternMatch]:
        """Create pattern matches from adherence analysis results.

        Args:
            analysis: Adherence analysis result
            file_path: File path being analyzed

        Returns:
            List of PatternMatch objects
        """
        patterns = []

        # Domain classification pattern
        patterns.append(
            PatternMatch(
                pattern_type=PatternType.ARCHITECTURAL,
                pattern_name=f"domain_adherence_{analysis.domain_classification.domain.value}",
                file_path=file_path,
                line_number=None,
                column=None,
                matched_text=f"Domain adherence analysis for {analysis.domain_classification.domain.value}",
                confidence=analysis.adherence_score.confidence,
                context={
                    "domain": analysis.domain_classification.domain.value,
                    "overall_adherence": analysis.adherence_score.overall_adherence,
                    "similar_patterns_count": analysis.adherence_score.similar_patterns_count,
                    "top_similarity": max(
                        [p.similarity_score for p in analysis.similar_patterns],
                        default=0.0,
                    ),
                },
            )
        )

        # Pattern similarity matches
        for i, similar_pattern in enumerate(
            analysis.similar_patterns[:3]
        ):  # Top 3 patterns
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.ARCHITECTURAL,
                    pattern_name=f"similar_pattern_{i + 1}",
                    file_path=file_path,
                    line_number=None,
                    column=None,
                    matched_text=f"Similar to pattern in {similar_pattern.file_path}",
                    confidence=similar_pattern.similarity_score,
                    context={
                        "reference_file": similar_pattern.file_path,
                        "similarity_score": similar_pattern.similarity_score,
                        "domain": similar_pattern.domain,
                    },
                )
            )

        return patterns

    def _generate_adherence_recommendations(
        self, analysis: AdherenceAnalysisResult, file_path: str
    ) -> list[Recommendation]:
        """Generate recommendations based on adherence analysis.

        Args:
            analysis: Adherence analysis result
            file_path: File path being analyzed

        Returns:
            List of Recommendation objects
        """
        recommendations = []

        # Low overall adherence
        if analysis.adherence_score.overall_adherence < 0.5:
            severity = (
                Severity.ERROR
                if analysis.adherence_score.overall_adherence < 0.3
                else Severity.WARNING
            )

            recommendations.append(
                Recommendation(
                    severity=severity,
                    category="domain_adherence",
                    message=f"Low adherence to {analysis.domain_classification.domain.value} domain patterns "
                    f"({analysis.adherence_score.overall_adherence:.2f})",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Review similar patterns in the codebase and align implementation",
                    rule_id="LOW_DOMAIN_ADHERENCE",
                )
            )

        # Poor domain classification
        if analysis.domain_classification.confidence < self.domain_confidence_threshold:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="domain_classification",
                    message=f"Unclear domain classification ({analysis.domain_classification.confidence:.2f} confidence)",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Add domain-specific patterns or move to appropriate directory",
                    rule_id="UNCLEAR_DOMAIN",
                )
            )

        # No similar patterns found
        if analysis.adherence_score.similar_patterns_count == 0:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="pattern_similarity",
                    message="No similar patterns found in the codebase",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Consider following established patterns for this domain",
                    rule_id="NO_SIMILAR_PATTERNS",
                )
            )

        # Add improvement suggestions as recommendations
        for i, suggestion in enumerate(analysis.improvement_suggestions[:3]):
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="adherence_improvement",
                    message=suggestion,
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Apply suggested improvement",
                    rule_id=f"ADHERENCE_IMPROVEMENT_{i + 1}",
                )
            )

        return recommendations

    def _generate_improvement_suggestions(
        self,
        domain_classification: DomainClassificationResult,
        similar_patterns: list[SimilarityMatch],
        adherence_score: AdherenceScore,
    ) -> list[str]:
        """Generate specific improvement suggestions based on analysis.

        Args:
            domain_classification: Domain classification result
            similar_patterns: Similar patterns found
            adherence_score: Calculated adherence scores

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        domain = domain_classification.domain

        # Domain-specific suggestions
        if (
            domain == ArchitecturalDomain.FRONTEND
            and adherence_score.pattern_similarity < 0.6
        ):
            suggestions.append(
                "Consider following React/Next.js component patterns like proper TypeScript interfaces, "
                "useState hooks, and component composition"
            )

        elif (
            domain == ArchitecturalDomain.BACKEND
            and adherence_score.pattern_similarity < 0.6
        ):
            suggestions.append(
                "Follow API route patterns with proper error handling, request validation, "
                "and response formatting"
            )

        elif (
            domain == ArchitecturalDomain.TESTING
            and adherence_score.pattern_similarity < 0.6
        ):
            suggestions.append(
                "Use established testing patterns like proper describe/it structure, "
                "meaningful test names, and appropriate assertions"
            )

        # Pattern-based suggestions
        if similar_patterns:
            best_match = max(similar_patterns, key=lambda p: p.similarity_score)
            suggestions.append(
                f"Consider patterns from {best_match.file_path} "
                f"(similarity: {best_match.similarity_score:.2f})"
            )

        # General adherence suggestions
        if adherence_score.domain_match_quality < 0.7:
            suggestions.append(
                f"Improve alignment with {domain.value} domain by using domain-specific "
                "imports, naming conventions, and structural patterns"
            )

        return suggestions

    def _calculate_adherence_score(self, analysis: AdherenceAnalysisResult) -> float:
        """Calculate the overall analyzer score from adherence analysis.

        Args:
            analysis: Adherence analysis result

        Returns:
            Overall score for this analyzer (0.0 to 1.0)
        """
        return analysis.adherence_score.overall_adherence

    def _get_supported_extensions(self) -> set[str]:
        """Return the file extensions this analyzer supports.

        Returns:
            Set of supported file extensions
        """
        return {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".py",
            ".sql",
            ".md",
            ".mdx",
            ".json",
            ".yml",
            ".yaml",
            ".css",
            ".scss",
            ".less",
        }

    def get_domain_statistics(self) -> dict[str, Any]:
        """Get statistics about built domain indices.

        Returns:
            Dictionary with domain statistics
        """
        statistics_dict = {
            "indices_built": sorted(self._indices_built),
            "total_domains": len(self._indices_built),
        }

        if self.pattern_indexer is not None:
            statistics_dict["pattern_indexer_stats"] = (
                self.pattern_indexer.get_cache_statistics()
            )

            for domain in self._indices_built:
                domain_stats = self.pattern_indexer.get_domain_statistics(domain)
                statistics_dict[f"domain_{domain}"] = domain_stats
        else:
            statistics_dict["pattern_indexer_stats"] = {"status": "not_available"}

        return statistics_dict
