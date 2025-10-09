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
    DomainDiagnostics,
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
        logger.debug("Starting DomainAwareAdherenceAnalyzer initialization")

        # Store progress callback for reporting
        self.progress_callback = progress_callback

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if self.progress_callback:
                self.progress_callback(message)

        report_progress("Initializing base analyzer...")
        try:
            super().__init__(config)
        except Exception as e:
            logger.error(f"Failed to initialize parent BaseAnalyzer: {e}")
            raise

        # Initialize components
        report_progress("Initializing domain classifier...")
        try:
            self.domain_classifier = DomainClassifier(config)
        except Exception as e:
            logger.error(f"Failed to initialize DomainClassifier: {e}")
            raise

        # Handle config safely
        config = config or {}

        # Initialize pattern indexer if dependencies are available
        report_progress("Checking ML model dependencies...")
        if PatternIndexer is not None:
            report_progress("Loading GraphCodeBERT model (this may take a while)...")

            model_name = config.get("model_name", "microsoft/graphcodebert-base")
            cache_dir = config.get("cache_dir")

            def pattern_indexer_progress_callback(message: str) -> None:
                """Nested progress callback for pattern indexer."""
                report_progress(f"Model loading: {message}")

            try:
                self.pattern_indexer = PatternIndexer(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    device_manager=device_manager,
                    progress_callback=pattern_indexer_progress_callback,
                )
            except Exception as e:
                logger.error(f"PatternIndexer initialization failed: {e}")
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
        self.similarity_threshold = config.get("similarity_threshold", 0.4)
        self.min_patterns_for_analysis = config.get("min_patterns_for_analysis", 3)
        self.max_similar_patterns = config.get("max_similar_patterns", 15)
        self.domain_confidence_threshold = config.get(
            "domain_confidence_threshold", 0.8
        )

        # Track if indices have been built and from which commit
        self._indices_built: set[str] = set()
        self._indices_commit: str | None = None
        self._warned_domains: set[str] = set()  # Track domains we've warned about

        report_progress("Domain adherence analyzer ready!")

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
            adherence_analysis, file_path, content
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
            and domain_str not in self._warned_domains
        ):
            logger.warning(
                f"No pattern index found for domain {domain_str}. Adherence analysis limited."
            )
            self._warned_domains.add(domain_str)

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
            domain_classification, similar_patterns, adherence_score, file_path
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

    def build_pattern_indices(
        self, codebase_files: dict[str, str], source_commit: str | None = None
    ) -> None:
        """Build pattern indices for all domains from a codebase.

        Args:
            codebase_files: Dictionary mapping file paths to their content
            source_commit: Commit hash these files were extracted from (for tracking)
        """
        if self.pattern_indexer is None:
            logger.warning("Pattern indexer not available. Skipping index building.")
            return

        logger.info(
            f"Building pattern indices for {len(codebase_files)} files from commit {source_commit}"
        )

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

        # Store which commit these indices were built from
        self._indices_commit = source_commit
        logger.info(
            f"Successfully built indices for domains: {list(self._indices_built)} from commit {source_commit}"
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

            # Debug logging
            logger.debug("=== PATTERN SIMILARITY ===")
            logger.debug(f"Found {len(similar_patterns)} similar patterns")
            logger.debug(
                f"Individual scores: {[f'{s:.4f}' for s in similarity_scores[:5]]}"
            )
            logger.debug(f"Mean similarity: {pattern_similarity:.4f}")
            logger.debug(f"Max similarity: {max_similarity:.4f}")
        else:
            pattern_similarity = 0.0
            max_similarity = 0.0
            logger.debug("No similar patterns found")

        # Enhanced domain adherence calculation
        if domain_classification.domain == ArchitecturalDomain.UNKNOWN:
            # Improved handling for unknown domains - less punitive
            if pattern_similarity > 0.5:
                domain_adherence = 0.6  # Good patterns even if domain unclear
            elif pattern_similarity > 0.3:
                domain_adherence = 0.5  # Moderate patterns
            else:
                domain_adherence = 0.4  # Neutral fallback
        else:
            # Enhanced weight by domain confidence and pattern similarity
            domain_adherence = (domain_match_quality * 0.5) + (pattern_similarity * 0.5)

            # Boost score for well-classified domains with good patterns
            if domain_match_quality > 0.8 and pattern_similarity > 0.6:
                domain_adherence = min(1.0, domain_adherence * 1.1)

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
        """Calculate improved weighted overall adherence score.

        Args:
            domain_quality: Quality of domain classification
            pattern_similarity: Average pattern similarity
            pattern_count: Number of similar patterns found

        Returns:
            Overall adherence score (0.0 to 1.0)
        """
        # Direct scoring based on actual similarity (no artificial compression)
        domain_weight = 0.25
        similarity_weight = 0.6  # Higher weight on actual pattern similarity
        coverage_weight = 0.15

        # Enhanced coverage bonus with better scaling
        coverage_score = min(1.0, pattern_count / 8.0)

        # Calculate weighted components - direct score, no compression
        weighted_components = (
            domain_quality * domain_weight
            + pattern_similarity * similarity_weight
            + coverage_score * coverage_weight
        )

        # Use weighted components directly (no base score or multiplier)
        overall_score = weighted_components

        # Additional bonuses for high-quality classifications
        bonuses = 0.0
        if domain_quality > 0.8:
            overall_score += 0.05
            bonuses += 0.05
        if pattern_similarity > 0.7:
            overall_score += 0.05
            bonuses += 0.05
        if pattern_count >= 5:
            overall_score += 0.03
            bonuses += 0.03

        final_score = max(0.0, min(1.0, overall_score))

        # Debug logging
        logger.debug("=== SCORE CALCULATION ===")
        logger.debug("Inputs:")
        logger.debug(f"  domain_quality: {domain_quality:.4f}")
        logger.debug(f"  pattern_similarity: {pattern_similarity:.4f}")
        logger.debug(f"  pattern_count: {pattern_count}")
        logger.debug("Components:")
        logger.debug(f"  coverage_score: {coverage_score:.4f}")
        logger.debug(f"  weighted_components: {weighted_components:.4f}")
        logger.debug(f"  bonuses: +{bonuses:.4f}")
        logger.debug(f"  pre_cap_total: {overall_score:.4f}")
        logger.debug(f"Final score (capped 0-1): {final_score:.4f}")

        return final_score

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
        self, analysis: AdherenceAnalysisResult, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate recommendations based on adherence analysis.

        Args:
            analysis: Adherence analysis result
            file_path: File path being analyzed
            content: File content

        Returns:
            List of Recommendation objects
        """
        recommendations = []

        # Enhanced adherence recommendations with better thresholds
        if analysis.adherence_score.overall_adherence < 0.4:
            # Only trigger error for very low scores
            severity = (
                Severity.ERROR
                if analysis.adherence_score.overall_adherence < 0.25
                else Severity.WARNING
            )

            # Provide more specific guidance
            domain_name = analysis.domain_classification.domain.value
            if domain_name == "unknown":
                message = f"Code patterns unclear - consider improving domain-specific implementation ({analysis.adherence_score.overall_adherence:.2f})"
                suggested_fix = "Add domain-specific imports, naming conventions, or move to appropriate directory structure"
            else:
                message = f"Low adherence to {domain_name} domain patterns ({analysis.adherence_score.overall_adherence:.2f})"
                suggested_fix = f"Review similar {domain_name} patterns in the codebase and align implementation"

            recommendations.append(
                Recommendation(
                    severity=severity,
                    category="domain_adherence",
                    message=message,
                    file_path=file_path,
                    line_number=None,
                    suggested_fix=suggested_fix,
                    rule_id="LOW_DOMAIN_ADHERENCE",
                )
            )

        # Enhanced domain classification recommendations with detailed diagnostics
        # Use more nuanced thresholds based on confidence levels
        if (
            analysis.domain_classification.confidence < 0.6
        ):  # Lowered threshold to catch more cases
            if analysis.domain_classification.confidence < 0.3:
                severity = Severity.ERROR  # Very low confidence needs attention
            elif analysis.domain_classification.confidence < 0.5:
                severity = Severity.WARNING  # Low confidence is concerning
            else:
                severity = Severity.INFO  # Moderate confidence is informational

            # Get detailed diagnostics for enhanced feedback
            enhanced_recommendation = self._generate_enhanced_domain_recommendation(
                file_path, analysis.domain_classification, severity, content
            )
            recommendations.append(enhanced_recommendation)

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
        file_path: str = "",
    ) -> list[str]:
        """Generate specific improvement suggestions based on analysis.

        Args:
            domain_classification: Domain classification result
            similar_patterns: Similar patterns found
            adherence_score: Calculated adherence scores
            file_path: Path to the file being analyzed (for extension filtering)

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

        # Pattern-based suggestions with file type filtering
        if similar_patterns:
            from pathlib import Path

            # Filter to only relevant file types for meaningful comparisons
            file_ext = Path(file_path).suffix if file_path else ""

            # Define file type categories for filtering
            code_extensions = {".ts", ".tsx", ".js", ".jsx", ".py"}
            sql_extensions = {".sql"}

            # Filter similar patterns to only include compatible file types
            relevant_patterns = []
            for pattern in similar_patterns:
                pattern_ext = Path(pattern.file_path).suffix

                # Same extension is always compatible
                if pattern_ext == file_ext:
                    relevant_patterns.append(pattern)
                # Code-to-code comparisons (any code language to any code language)
                elif file_ext in code_extensions and pattern_ext in code_extensions:
                    relevant_patterns.append(pattern)
                # SQL files only to SQL files
                elif file_ext in sql_extensions and pattern_ext in sql_extensions:
                    relevant_patterns.append(pattern)
                # Don't compare code to docs/config or vice versa
                # (already filtered by above conditions)

            if relevant_patterns:
                best_match = max(relevant_patterns, key=lambda p: p.similarity_score)
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

    def _generate_enhanced_domain_recommendation(
        self,
        file_path: str,
        classification: DomainClassificationResult,
        severity: Severity,
        content: str,
    ) -> Recommendation:
        """Generate enhanced domain classification recommendation with detailed diagnostics.

        Args:
            file_path: Path to the file
            classification: Domain classification result
            severity: Severity level for the recommendation
            content: File content (from git commit, not disk)

        Returns:
            Enhanced recommendation with detailed diagnostics
        """
        # Get diagnostic details from the domain classifier
        try:
            from pathlib import Path

            # Use provided content instead of reading from disk
            diagnostics = self.domain_classifier.get_classification_diagnostics(
                file_path, content
            )
            # Ensure diagnostics is a dictionary
            if not isinstance(diagnostics, dict):
                diagnostics = {}
        except Exception as e:
            logger.warning(f"Could not get diagnostics for {file_path}: {e}")
            diagnostics = {}

        # Create enhanced message with detailed breakdown
        confidence_desc = (
            "very low"
            if classification.confidence < 0.3
            else (
                "low"
                if classification.confidence < 0.5
                else "moderate" if classification.confidence < 0.6 else "unclear"
            )
        )
        message_parts = [
            f"Domain classification {confidence_desc} for '{Path(file_path).name}' ({classification.confidence:.2f} confidence)"
        ]

        # Add domain evaluation breakdown
        if (
            classification.classification_factors
            and "combined_scores" in classification.classification_factors
        ):
            combined_scores = classification.classification_factors["combined_scores"]
            # Get top 3 domains by score
            top_domains = sorted(
                [
                    (domain, score)
                    for domain, score in combined_scores.items()
                    if score > 0.05
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            if top_domains:
                message_parts.append("\nEvaluated domains:")
                for domain_name, score in top_domains:
                    domain_enum = self._get_domain_enum_by_name(domain_name)
                    missing_patterns = self._get_missing_patterns_summary(
                        diagnostics, domain_enum
                    )
                    missing_text = (
                        f" (missing: {missing_patterns})" if missing_patterns else ""
                    )
                    message_parts.append(
                        f"  - {domain_name.title()}: {score:.2f}{missing_text}"
                    )

        # Add specific improvements
        improvements = self._generate_domain_specific_improvements(
            file_path, classification, diagnostics
        )
        if improvements:
            message_parts.append("\nSpecific improvements:")
            for i, improvement in enumerate(improvements[:4], 1):
                message_parts.append(f"{i}. {improvement}")

        enhanced_message = "".join(message_parts)

        # Create more specific suggested fix based on improvements
        if improvements:
            # Use the first 3 most specific improvements as the suggested fix
            suggested_fix = "; ".join(improvements[:3])
        else:
            # Fallback to generic suggestions
            suggested_fixes = [
                "Review file location and ensure it's in the appropriate domain directory",
                "Add domain-specific imports and patterns",
                "Consider splitting mixed concerns into separate files",
            ]
            suggested_fix = "; ".join(suggested_fixes)

        return Recommendation(
            severity=severity,
            category="domain_classification",
            message=enhanced_message,
            file_path=file_path,
            line_number=None,
            suggested_fix=suggested_fix,
            rule_id="UNCLEAR_DOMAIN_ENHANCED",
        )

    def _get_domain_enum_by_name(self, domain_name: str) -> ArchitecturalDomain:
        """Get ArchitecturalDomain enum by name."""
        try:
            return ArchitecturalDomain(domain_name.lower())
        except ValueError:
            return ArchitecturalDomain.UNKNOWN

    def _get_missing_patterns_summary(
        self,
        diagnostics: dict[ArchitecturalDomain, list[DomainDiagnostics]],
        domain: ArchitecturalDomain,
    ) -> str:
        """Get a summary of missing patterns for a domain."""
        if not diagnostics or domain not in diagnostics:
            return ""

        all_missing = []
        for diag in diagnostics[domain]:
            # Prioritize the most actionable missing patterns
            if diag.pattern_category == "import":
                # Import patterns are most actionable - show top missing imports
                relevant_missing = [
                    p for p in diag.missing_patterns[:3] if not p.startswith("r")
                ]
                all_missing.extend(relevant_missing)
            elif diag.pattern_category == "content":
                # Content patterns are second most actionable
                relevant_missing = [
                    p for p in diag.missing_patterns[:2] if not p.startswith("r")
                ]
                all_missing.extend(relevant_missing)
            elif diag.pattern_category == "path":
                # Path patterns last (often just directory suggestions)
                relevant_missing = [
                    p for p in diag.missing_patterns[:1] if not p.startswith("r")
                ]
                all_missing.extend(relevant_missing)

        # Remove duplicates and limit total to most actionable
        unique_missing = list(dict.fromkeys(all_missing))[:3]
        return ", ".join(unique_missing)

    def _generate_domain_specific_improvements(
        self,
        file_path: str,
        classification: DomainClassificationResult,
        diagnostics: dict[ArchitecturalDomain, list[DomainDiagnostics]],
    ) -> list[str]:
        """Generate domain-specific improvement suggestions."""
        from pathlib import Path

        improvements = []
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        # Get the top domains from classification factors to provide targeted suggestions
        top_domains = []
        if (
            classification.classification_factors
            and "combined_scores" in classification.classification_factors
        ):
            combined_scores = classification.classification_factors["combined_scores"]
            top_domains = sorted(
                [
                    (domain, score)
                    for domain, score in combined_scores.items()
                    if score > 0.1
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

        # Analyze missing patterns from diagnostics to provide specific suggestions
        missing_by_category: dict[str, list[str]] = {
            "import": [],
            "content": [],
            "path": [],
        }
        for domain_diags in diagnostics.values():
            for diag in domain_diags:
                if diag.pattern_category in missing_by_category:
                    # Get human-readable missing patterns
                    missing_by_category[diag.pattern_category].extend(
                        diag.missing_patterns[:2]
                    )

        # Generate file-extension specific suggestions
        if file_ext in [".tsx", ".jsx", ".ts", ".js"]:
            improvements.extend(
                self._get_javascript_improvements(
                    file_path, file_name, file_ext, top_domains, missing_by_category
                )
            )
        elif file_ext in [".py"]:
            improvements.extend(
                self._get_python_improvements(
                    file_path, file_name, top_domains, missing_by_category
                )
            )
        elif file_ext in [".sql"]:
            improvements.extend(
                self._get_database_improvements(
                    file_path, file_name, missing_by_category
                )
            )
        elif file_ext in [".md", ".mdx"]:
            improvements.append(
                "Move to 'docs/' directory or add proper documentation structure"
            )
            improvements.append("Include frontmatter or proper markdown formatting")

        # Add domain-specific suggestions based on top scoring domains
        for domain_name, score in top_domains:
            if score > 0.2:  # Only suggest for domains with reasonable scores
                improvements.extend(
                    self._get_domain_specific_suggestions(domain_name, file_path)
                )

        # Generic improvements based on primary domain
        if classification.domain == ArchitecturalDomain.UNKNOWN:
            improvements.append(
                "Add clear domain indicators (imports, directory structure, or file naming)"
            )
            improvements.append(
                "Consider splitting file if it contains mixed domain concerns"
            )

        # Path-based suggestions
        if "src/" not in file_path and file_ext in [
            ".tsx",
            ".jsx",
            ".ts",
            ".js",
            ".py",
        ]:
            improvements.append("Move to 'src/' directory for better organization")

        # Remove duplicates and limit
        unique_improvements = list(dict.fromkeys(improvements))
        return unique_improvements[:5]  # Limit to 5 most relevant improvements

    def _get_javascript_improvements(
        self,
        file_path: str,
        file_name: str,
        file_ext: str,
        top_domains: list[tuple[str, float]],
        missing_by_category: dict[str, list[str]],
    ) -> list[str]:
        """Get JavaScript/TypeScript specific improvements."""
        improvements = []

        # Check if React patterns are missing
        if any(
            "React" in missing or "JSX" in missing
            for missing in missing_by_category["import"]
            + missing_by_category["content"]
        ):
            if file_ext in [".tsx", ".jsx"]:
                improvements.append("Add React imports: import React from 'react'")
                improvements.append(
                    "Include JSX elements with proper component structure"
                )

        # Check for Next.js patterns
        if any(
            "Next.js" in missing or "API" in missing
            for missing in missing_by_category["import"]
        ):
            if "api" in file_path.lower():
                improvements.append(
                    "Add Next.js API route structure: export async function GET(request) { ... }"
                )
                improvements.append(
                    "Include NextRequest/NextResponse imports from 'next/server'"
                )
            elif file_ext in [".tsx", ".jsx"]:
                improvements.append(
                    "Add Next.js imports like useRouter, Link, or Image from 'next/*'"
                )

        # Component vs Page suggestions
        if "component" in file_name.lower():
            improvements.append("Move to 'src/components/' directory")
            improvements.append(
                "Export component as default: export default function ComponentName()"
            )
        elif "page" in file_name.lower():
            improvements.append("Move to 'src/app/' or 'src/pages/' directory")

        # Testing patterns
        if any(domain_name == "testing" for domain_name, _ in top_domains):
            improvements.append(
                "Add testing imports: import { describe, test, expect } from 'jest' or 'vitest'"
            )
            improvements.append("Include test structure: describe() and test() blocks")

        return improvements

    def _get_python_improvements(
        self,
        file_path: str,
        file_name: str,
        top_domains: list[tuple[str, float]],
        missing_by_category: dict[str, list[str]],
    ) -> list[str]:
        """Get Python-specific improvements."""
        improvements = []

        # Backend framework suggestions
        backend_score = next(
            (score for domain, score in top_domains if domain == "backend"), 0
        )
        if backend_score > 0.1:
            improvements.append(
                "Add Python web framework imports (FastAPI, Flask, Django)"
            )
            improvements.append("Include request/response handling patterns")

        # Testing suggestions
        testing_score = next(
            (score for domain, score in top_domains if domain == "testing"), 0
        )
        if testing_score > 0.1:
            improvements.append("Add testing imports: import pytest or import unittest")
            improvements.append("Include test functions with test_ prefix")

        # Database patterns
        if "model" in file_name.lower() or "schema" in file_name.lower():
            improvements.append("Add ORM imports (SQLAlchemy, Django ORM, etc.)")
            improvements.append("Include database model definitions")

        return improvements

    def _get_database_improvements(
        self, file_path: str, file_name: str, missing_by_category: dict[str, list[str]]
    ) -> list[str]:
        """Get database-specific improvements."""
        improvements = []

        improvements.append(
            "Move to 'migrations/', 'schema/', or 'database/' directory"
        )
        improvements.append(
            "Include proper SQL DDL statements (CREATE TABLE, ALTER TABLE)"
        )

        if "migration" in file_name.lower():
            improvements.append(
                "Add migration-specific patterns (UP/DOWN, version control)"
            )

        return improvements

    def _get_domain_specific_suggestions(
        self, domain_name: str, file_path: str
    ) -> list[str]:
        """Get suggestions specific to a domain."""
        suggestions = []

        if domain_name == "frontend":
            suggestions.append("Add React hooks or component lifecycle patterns")
            suggestions.append("Include CSS-in-JS or styling imports")
        elif domain_name == "backend":
            suggestions.append("Add server-side request handling patterns")
            suggestions.append("Include middleware or authentication patterns")
        elif domain_name == "database":
            suggestions.append("Add database connection or ORM patterns")
            suggestions.append("Include proper schema definitions")
        elif domain_name == "testing":
            suggestions.append("Add test assertions and mocking patterns")
            suggestions.append("Include proper test setup/teardown")
        elif domain_name == "infrastructure":
            suggestions.append("Add deployment or configuration patterns")
            suggestions.append("Include environment-specific settings")

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
