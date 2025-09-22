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

"""Domain classification analyzer for identifying architectural domains.

This analyzer classifies code files into architectural domains (frontend, backend,
database, infrastructure, testing) using path analysis, import patterns, and
content analysis to enable domain-aware adherence measurement.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base_analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    PatternMatch,
    PatternType,
    Recommendation,
    Severity,
)

logger = logging.getLogger(__name__)


class ArchitecturalDomain(Enum):
    """Architectural domains for code classification."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


@dataclass
class DomainClassificationResult:
    """Results from domain classification analysis."""

    domain: ArchitecturalDomain
    confidence: float  # 0.0 to 1.0
    classification_factors: dict[str, Any]
    secondary_domains: list[tuple[ArchitecturalDomain, float]]


class DomainClassifier(BaseAnalyzer):
    """Classifies code changes by architectural domain.

    Uses multiple analysis techniques:
    - Path-based classification using directory and file patterns
    - Import pattern analysis to identify framework usage
    - Content pattern analysis for domain-specific code constructs
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the DomainClassifier with configuration.

        Args:
            config: Optional configuration dictionary for the classifier.
        """
        super().__init__(config)
        self.domain_patterns = self._init_domain_patterns()
        self.import_patterns = self._init_import_patterns()
        self.content_patterns = self._init_content_patterns()

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'domain_classifier'.
        """
        return "domain_classifier"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.20 for 20% of overall score).
        """
        return 0.20

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a file to determine its architectural domain.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with domain classification information
        """
        start_time = time.time()

        # Perform domain classification
        classification_result = self.classify_domain(file_path, content)

        # Create pattern matches based on classification
        patterns_found = [
            PatternMatch(
                pattern_type=PatternType.ARCHITECTURAL,
                pattern_name=f"domain_{classification_result.domain.value}",
                file_path=file_path,
                line_number=None,
                column=None,
                matched_text=f"Domain: {classification_result.domain.value}",
                confidence=classification_result.confidence,
                context={
                    "domain": classification_result.domain.value,
                    "factors": classification_result.classification_factors,
                    "secondary_domains": [
                        (d.value, conf)
                        for d, conf in classification_result.secondary_domains
                    ],
                },
            )
        ]

        # Generate recommendations based on classification confidence
        recommendations = self._generate_domain_recommendations(
            file_path, classification_result
        )

        # Calculate score based on classification confidence
        score = classification_result.confidence

        # Collect metrics
        metrics = {
            "domain": classification_result.domain.value,
            "confidence": classification_result.confidence,
            "classification_factors": classification_result.classification_factors,
            "secondary_domains_count": len(classification_result.secondary_domains),
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

    def classify_domain(
        self, file_path: str, content: str
    ) -> DomainClassificationResult:
        """Classify a file into its architectural domain.

        Args:
            file_path: Path to the file
            content: Content of the file

        Returns:
            DomainClassificationResult with classification details
        """
        # Analyze different classification factors
        path_scores = self._analyze_path_patterns(file_path)
        import_scores = self._analyze_import_patterns(content)
        content_scores = self._analyze_content_patterns(content)

        # Combine scores with weights
        combined_scores = self._combine_classification_scores(
            path_scores, import_scores, content_scores
        )

        # Determine primary domain
        primary_domain = max(combined_scores, key=lambda x: combined_scores[x])
        primary_confidence = combined_scores[primary_domain]

        # Get secondary domains (excluding primary)
        secondary_domains = [
            (domain, score)
            for domain, score in combined_scores.items()
            if domain != primary_domain and score > 0.1
        ]
        secondary_domains.sort(key=lambda x: x[1], reverse=True)

        return DomainClassificationResult(
            domain=primary_domain,
            confidence=primary_confidence,
            classification_factors={
                "path_scores": {d.value: s for d, s in path_scores.items()},
                "import_scores": {d.value: s for d, s in import_scores.items()},
                "content_scores": {d.value: s for d, s in content_scores.items()},
                "combined_scores": {d.value: s for d, s in combined_scores.items()},
            },
            secondary_domains=secondary_domains[:3],  # Top 3 secondary domains
        )

    def _init_domain_patterns(self) -> dict[ArchitecturalDomain, dict[str, Any]]:
        """Initialize domain-specific path patterns."""
        return {
            ArchitecturalDomain.BACKEND: {
                "path_patterns": [
                    r"src/app/api/.*\.ts$",  # Next.js API routes
                    r"src/api/.*\.(ts|js)$",  # API routes in src/api
                    r"api/.*\.(ts|js)$",
                    r"server/.*\.(ts|js|py)$",
                    r"backend/.*\.(ts|js|py)$",
                    r"src/lib/.*\.ts$",  # Server-side utilities
                    r".*\.server\.(ts|js)$",
                    r".*routes.*\.(ts|js|py)$",
                    r".*controllers.*\.(ts|js|py)$",
                    r".*services.*\.(ts|js|py)$",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.FRONTEND: {
                "path_patterns": [
                    r"src/components/.*\.(tsx?|jsx?)$",  # React components
                    r"src/pages/.*\.(tsx?|jsx?)$",  # Next.js pages
                    r"src/app/(?!api).*\.(tsx?|jsx?)$",  # Next.js app directory (excluding API)
                    r"components/.*\.(tsx?|jsx?)$",
                    r"src/(?!api/|lib/|utils\.)[^/]*\.(tsx?|jsx?)$",  # Direct React/TypeScript files in src root (excluding api/lib subdirs and utils.* files)
                    r".*\.css$",
                    r".*\.scss$",
                    r".*\.less$",
                    r"src/styles/.*",
                    r"public/.*",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.DATABASE: {
                "path_patterns": [
                    r".*migrations?/.*\.(sql|ts|js|py)$",
                    r".*models?/.*\.(ts|js|py)$",
                    r".*schema.*\.(ts|js|py|sql)$",
                    r"(^|.*/)?database/.*\.(ts|js|py|sql)$",  # Database directory only, not any file with "database" in name
                    r".*\.sql$",
                    r"prisma/.*",
                    r".*drizzle.*",
                ],
                "weight": 0.9,
            },
            ArchitecturalDomain.INFRASTRUCTURE: {
                "path_patterns": [
                    r".*dockerfile.*$",
                    r".*docker-compose.*\.(yml|yaml)$",
                    r".*\.dockerignore$",
                    r".*\.github/.*\.(yml|yaml)$",
                    r".*infrastructure/.*",
                    r".*deploy.*\.(yml|yaml|sh)$",
                    r".*terraform.*\.(tf|tfvars)$",
                    r".*ansible.*\.(yml|yaml)$",
                    r".*kubernetes.*\.(yml|yaml)$",
                ],
                "weight": 0.9,
            },
            ArchitecturalDomain.TESTING: {
                "path_patterns": [
                    r".*test.*\.(ts|tsx|js|jsx|py)$",
                    r".*spec.*\.(ts|tsx|js|jsx|py)$",
                    r".*/tests?/.*",
                    r".*/__tests__/.*",
                    r".*\.test\.(ts|tsx|js|jsx|py)$",
                    r".*\.spec\.(ts|tsx|js|jsx|py)$",
                    r".*cypress.*",
                    r".*playwright.*",
                    r".*e2e.*",
                ],
                "weight": 0.9,
            },
            ArchitecturalDomain.CONFIGURATION: {
                "path_patterns": [
                    r".*\.config\.(ts|js|json)$",
                    r".*\.json$",
                    r".*\.env.*$",
                    r".*package\.json$",
                    r".*tsconfig.*\.json$",
                    r".*\.eslintrc.*$",
                    r".*\.prettierrc.*$",
                    r".*next\.config\.(ts|js)$",
                    r".*tailwind\.config\.(ts|js)$",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.DOCUMENTATION: {
                "path_patterns": [
                    r".*\.md$",
                    r".*\.mdx$",
                    r".*readme.*$",
                    r".*changelog.*$",
                    r".*\.txt$",
                    r"docs?/.*",
                ],
                "weight": 0.7,
            },
        }

    def _init_import_patterns(self) -> dict[ArchitecturalDomain, dict[str, Any]]:
        """Initialize domain-specific import patterns."""
        return {
            ArchitecturalDomain.FRONTEND: {
                "patterns": [
                    r"import.*from\s+['\"]react['\"]",
                    r"import.*from\s+['\"]next/.*['\"]",
                    r"import.*from\s+['\"]@/components/.*['\"]",
                    r"use client",
                    r"import.*useState|useEffect|useCallback",
                    r"import.*from\s+['\"]styled-components['\"]",
                    r"import.*from\s+['\"]@emotion/.*['\"]",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.BACKEND: {
                "patterns": [
                    r"import.*from\s+['\"]next/server['\"]",
                    r"NextRequest|NextResponse",
                    r"import.*from\s+['\"]express['\"]",
                    r"import.*from\s+['\"]fastify['\"]",
                    r"import.*from\s+['\"]koa['\"]",
                    r"import.*from\s+['\"]node:.*['\"]",
                    r"import.*fs|path|crypto",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.DATABASE: {
                "patterns": [
                    r"import.*from\s+['\"]prisma/.*['\"]",
                    r"import.*from\s+['\"]drizzle-orm.*['\"]",
                    r"import.*from\s+['\"]mongoose['\"]",
                    r"import.*from\s+['\"]typeorm['\"]",
                    r"import.*from\s+['\"]sequelize['\"]",
                    r"import.*from\s+['\"]pg['\"]",
                    r"import.*from\s+['\"]mysql.*['\"]",
                ],
                "weight": 0.9,
            },
            ArchitecturalDomain.TESTING: {
                "patterns": [
                    r"import.*from\s+['\"]@testing-library/.*['\"]",
                    r"import.*from\s+['\"]jest['\"]",
                    r"import.*from\s+['\"]vitest['\"]",
                    r"import.*from\s+['\"]cypress['\"]",
                    r"import.*from\s+['\"]playwright['\"]",
                    r"describe|test|it|expect",
                ],
                "weight": 0.9,
            },
        }

    def _init_content_patterns(self) -> dict[ArchitecturalDomain, dict[str, Any]]:
        """Initialize domain-specific content patterns."""
        return {
            ArchitecturalDomain.FRONTEND: {
                "patterns": [
                    r"return\s*\(\s*<",  # JSX return
                    r"<\w+.*>",  # JSX elements
                    r"className=",
                    r"onClick=|onChange=|onSubmit=",
                    r"React\.FC|FC<",
                    r"useState|useEffect|useContext",
                ],
                "weight": 0.7,
            },
            ArchitecturalDomain.BACKEND: {
                "patterns": [
                    r"export\s+async\s+function\s+(GET|POST|PUT|DELETE|PATCH)",
                    r"req\.|res\.|request\.|response\.",
                    r"\.json\(\)|\.status\(\)",
                    r"async\s+function.*handler",
                    r"middleware",
                    r"cors|helmet|morgan",
                ],
                "weight": 0.8,
            },
            ArchitecturalDomain.DATABASE: {
                "patterns": [
                    r"SELECT|INSERT|UPDATE|DELETE|CREATE TABLE",
                    r"\.findMany\(\)|\.findUnique\(\)|\.create\(\)|\.update\(\)",
                    r"schema\.|model\.|entity\.",
                    r"migration|migrate",
                    r"connection|connect|disconnect",
                ],
                "weight": 0.9,
            },
            ArchitecturalDomain.TESTING: {
                "patterns": [
                    r"describe\(|test\(|it\(",
                    r"expect\(.*\)\.(toBe|toEqual|toContain)",
                    r"beforeEach|afterEach|beforeAll|afterAll",
                    r"mock|spy|stub",
                    r"render\(|screen\.|fireEvent",
                ],
                "weight": 0.9,
            },
        }

    def _analyze_path_patterns(
        self, file_path: str
    ) -> dict[ArchitecturalDomain, float]:
        """Analyze file path to determine domain scores.

        Args:
            file_path: Path to analyze

        Returns:
            Dictionary mapping domains to their path-based scores
        """
        scores = dict.fromkeys(ArchitecturalDomain, 0.0)

        for domain, config in self.domain_patterns.items():
            for pattern in config["path_patterns"]:
                if re.match(pattern, file_path, re.IGNORECASE):
                    scores[domain] = max(scores[domain], config["weight"])
                    break

        # Normalize scores
        max_score = max(scores.values()) if any(scores.values()) else 1.0
        if max_score > 0:
            scores = {domain: score / max_score for domain, score in scores.items()}

        return scores

    def _analyze_import_patterns(
        self, content: str
    ) -> dict[ArchitecturalDomain, float]:
        """Analyze import patterns to determine domain scores.

        Args:
            content: File content to analyze

        Returns:
            Dictionary mapping domains to their import-based scores
        """
        scores = dict.fromkeys(ArchitecturalDomain, 0.0)

        for domain, config in self.import_patterns.items():
            pattern_matches = 0
            for pattern in config["patterns"]:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    pattern_matches += 1

            if pattern_matches > 0:
                scores[domain] = min(
                    1.0, (pattern_matches / len(config["patterns"])) * config["weight"]
                )

        return scores

    def _analyze_content_patterns(
        self, content: str
    ) -> dict[ArchitecturalDomain, float]:
        """Analyze content patterns to determine domain scores.

        Args:
            content: File content to analyze

        Returns:
            Dictionary mapping domains to their content-based scores
        """
        scores = dict.fromkeys(ArchitecturalDomain, 0.0)

        for domain, config in self.content_patterns.items():
            pattern_matches = 0
            for pattern in config["patterns"]:
                matches = len(
                    re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                )
                if matches > 0:
                    pattern_matches += min(
                        matches, 5
                    )  # Cap at 5 to avoid over-weighting

            if pattern_matches > 0:
                scores[domain] = min(
                    1.0,
                    (pattern_matches / (len(config["patterns"]) * 3))
                    * config["weight"],
                )

        return scores

    def _combine_classification_scores(
        self,
        path_scores: dict[ArchitecturalDomain, float],
        import_scores: dict[ArchitecturalDomain, float],
        content_scores: dict[ArchitecturalDomain, float],
    ) -> dict[ArchitecturalDomain, float]:
        """Combine different classification scores with weighted averaging.

        Args:
            path_scores: Scores from path analysis
            import_scores: Scores from import analysis
            content_scores: Scores from content analysis

        Returns:
            Combined scores for each domain
        """
        # Weights for different score types
        path_weight = 0.4
        import_weight = 0.3
        content_weight = 0.3

        combined_scores = {}
        for domain in ArchitecturalDomain:
            path_score = path_scores.get(domain, 0.0)
            import_score = import_scores.get(domain, 0.0)
            content_score = content_scores.get(domain, 0.0)

            combined_score = (
                path_score * path_weight
                + import_score * import_weight
                + content_score * content_weight
            )

            combined_scores[domain] = combined_score

        # Apply confidence boost for strong classifications
        # This handles both single-signal and reinforcing multi-signal cases
        max_combined_score = max(combined_scores.values())
        max_domain = max(combined_scores, key=lambda x: combined_scores[x])

        # Check if we have a clear winner
        second_highest = (
            sorted(combined_scores.values(), reverse=True)[1]
            if len([s for s in combined_scores.values() if s > 0]) > 1
            else 0
        )

        # Special case: if testing is the winner and has clear test patterns, boost even with small margin
        if max_domain == ArchitecturalDomain.TESTING and max_combined_score >= 0.35:
            # Testing should be boosted even with mixed signals
            pass  # Continue to boost logic
        elif (
            max_combined_score >= 0.3 and (max_combined_score - second_highest) >= 0.05
        ):
            pass  # Normal margin check
        else:
            # Skip boosting
            max_combined_score = 0  # Signal to skip boost logic

        if max_combined_score >= 0.3:
            max_domain_path = path_scores.get(max_domain, 0.0)
            max_domain_import = import_scores.get(max_domain, 0.0)
            max_domain_content = content_scores.get(max_domain, 0.0)

            domain_signals = [max_domain_path, max_domain_import, max_domain_content]
            max_signal = max(domain_signals)
            signal_count = len([s for s in domain_signals if s > 0.1])

            # Case 1: Strong single signal (original boost)
            strong_signal_threshold = (
                0.15  # Threshold for what counts as a meaningful signal
            )
            meaningful_signals = [
                s for s in domain_signals if s >= strong_signal_threshold
            ]
            if max_signal >= 0.8 and len(meaningful_signals) == 1:
                boost_factor = 2.2
                combined_scores[max_domain] = min(
                    1.0, combined_scores[max_domain] * boost_factor
                )

            # Case 2: Strong reinforcing signals (path + content/import both present)
            elif signal_count >= 2 and max_signal >= 0.7:
                # At least 2 signals present with strong primary signal
                strong_signals = [s for s in domain_signals if s >= 0.2]
                if len(strong_signals) >= 2:
                    boost_factor = 1.8  # Strong reinforcement
                else:
                    boost_factor = 1.5  # Moderate reinforcement
                combined_scores[max_domain] = min(
                    1.0, combined_scores[max_domain] * boost_factor
                )

            # Case 3: Moderate single signal (smaller boost)
            elif max_signal >= 0.7 and signal_count == 1:
                boost_factor = 1.4
                combined_scores[max_domain] = min(
                    1.0, combined_scores[max_domain] * boost_factor
                )

        # Set unknown as default if no strong signals
        max_score = max(combined_scores.values()) if combined_scores else 0.0
        if max_score < 0.3:
            combined_scores[ArchitecturalDomain.UNKNOWN] = max(0.5, max_score)

        return combined_scores

    def _generate_domain_recommendations(
        self, file_path: str, classification: DomainClassificationResult
    ) -> list[Recommendation]:
        """Generate recommendations based on domain classification.

        Args:
            file_path: Path to the file
            classification: Domain classification result

        Returns:
            List of recommendations for improving domain clarity
        """
        recommendations = []

        # Low confidence recommendations
        if classification.confidence < 0.6:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="domain_classification",
                    message=f"Low confidence domain classification ({classification.confidence:.2f}). Consider improving file organization.",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Move to appropriate domain directory or add clear domain indicators",
                    rule_id="LOW_DOMAIN_CONFIDENCE",
                )
            )

        # Unknown domain recommendations
        if classification.domain == ArchitecturalDomain.UNKNOWN:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="domain_classification",
                    message="Unable to classify file into a clear architectural domain",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Add domain-specific patterns or move to appropriate directory structure",
                    rule_id="UNKNOWN_DOMAIN",
                )
            )

        # Multiple domain recommendations
        if len(classification.secondary_domains) > 2:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="domain_classification",
                    message="File shows patterns from multiple domains. Consider splitting responsibilities.",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Separate concerns into domain-specific files",
                    rule_id="MIXED_DOMAIN_CONCERNS",
                )
            )

        return recommendations

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
            ".html",
            ".htm",
            ".tf",
            ".dockerfile",
            ".sh",
            ".env",
            ".txt",
        }
