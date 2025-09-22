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

"""Base analyzer abstract class and common data structures.

This module provides the foundation for all code analysis modules, defining
the interface and common patterns for analyzing code adherence.
"""

from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for pattern violations and recommendations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PatternType(Enum):
    """Types of patterns that can be analyzed."""

    ARCHITECTURAL = "architectural"
    COMPONENT = "component"
    IMPORT = "import"
    NAMING = "naming"
    TYPE_SAFETY = "type_safety"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    TESTING = "testing"
    FRAMEWORK = "framework"


@dataclass
class PatternMatch:
    """Represents a matched pattern in the code."""

    pattern_type: PatternType
    pattern_name: str
    file_path: str
    line_number: int | None
    column: int | None
    matched_text: str
    confidence: float  # 0.0 to 1.0
    context: dict[str, Any]


@dataclass
class Recommendation:
    """Represents an actionable recommendation for code improvement."""

    severity: Severity
    category: str
    message: str
    file_path: str
    line_number: int | None
    suggested_fix: str | None
    rule_id: str


@dataclass
class AnalysisResult:
    """Results from analyzing a single file or commit."""

    file_path: str
    score: float  # 0.0 to 1.0
    patterns_found: list[PatternMatch]
    recommendations: list[Recommendation]
    metrics: dict[str, Any]
    analysis_time: float


class BaseAnalyzer(ABC):
    """Abstract base class for all code analyzers.

    Each analyzer focuses on a specific dimension of code quality or adherence
    and provides scoring and recommendations for that dimension.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the base analyzer with configuration.

        Args:
            config: Optional configuration dictionary for the analyzer.
        """
        self.config = config or {}
        self.patterns: dict[str, Any] = {}
        self.reference_files: set[str] = set()
        self._load_patterns()

    @abstractmethod
    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a single file and return results.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult containing scores, patterns, and recommendations
        """
        pass

    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Return the name of this analyzer."""
        pass

    @abstractmethod
    def get_weight(self) -> float:
        """Return the default weight for this analyzer in overall scoring."""
        pass

    def analyze_commit(self, commit_files: dict[str, str]) -> dict[str, AnalysisResult]:
        """Analyze all files in a commit.

        Args:
            commit_files: Dictionary mapping file paths to their content

        Returns:
            Dictionary mapping file paths to their analysis results
        """
        results = {}
        for file_path, content in commit_files.items():
            if self._should_analyze_file(file_path):
                try:
                    results[file_path] = self.analyze_file(file_path, content)
                except Exception as e:
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    # Create a minimal result for failed analysis
                    results[file_path] = AnalysisResult(
                        file_path=file_path,
                        score=0.0,
                        patterns_found=[],
                        recommendations=[
                            Recommendation(
                                severity=Severity.ERROR,
                                category="analysis_error",
                                message=f"Analysis failed: {e}",
                                file_path=file_path,
                                line_number=None,
                                suggested_fix=None,
                                rule_id="ANALYSIS_ERROR",
                            )
                        ],
                        metrics={"analysis_failed": True},
                        analysis_time=0.0,
                    )
        return results

    def learn_from_reference(self, reference_files: dict[str, str]) -> None:
        """Learn patterns from high-quality reference files.

        Args:
            reference_files: Dictionary mapping file paths to their content
        """
        logger.info(f"Learning patterns from {len(reference_files)} reference files")
        for file_path, content in reference_files.items():
            if self._should_analyze_file(file_path):
                self._extract_patterns_from_reference(file_path, content)
        self._update_pattern_weights()

    def _should_analyze_file(self, file_path: str) -> bool:
        """Determine if a file should be analyzed by this analyzer.

        Args:
            file_path: Path to the file

        Returns:
            True if the file should be analyzed
        """
        path = Path(file_path)

        # Default file extensions this analyzer handles
        supported_extensions = self._get_supported_extensions()
        if path.suffix not in supported_extensions:
            return False

        # Skip common exclusions
        exclude_patterns = [
            "node_modules",
            ".git",
            "__pycache__",
            ".next",
            "dist",
            "build",
        ]

        for pattern in exclude_patterns:
            if pattern in file_path:
                return False

        return True

    def _get_supported_extensions(self) -> set[str]:
        """Return the file extensions this analyzer supports.

        Returns:
            Set of supported file extensions (including the dot)
        """
        return {".ts", ".tsx", ".js", ".jsx"}

    def _load_patterns(self) -> None:
        """Load analyzer-specific patterns from configuration or defaults."""
        # Base implementation - subclasses should override
        pass

    def _extract_patterns_from_reference(self, file_path: str, content: str) -> None:
        """Extract patterns from a reference file to improve analysis.

        Args:
            file_path: Path to the reference file
            content: Content of the reference file
        """
        # Base implementation - subclasses should override
        pass

    def _update_pattern_weights(self) -> None:
        """Update pattern weights based on learned patterns."""
        # Base implementation - subclasses should override
        pass

    def _parse_typescript_ast(self, content: str) -> ast.AST | None:
        """Parse TypeScript/JavaScript content into an AST.

        This is a simplified parser - in practice, you'd want to use a proper
        TypeScript parser like the TypeScript compiler API or a Python binding.

        Args:
            content: Source code content

        Returns:
            AST node or None if parsing fails
        """
        try:
            # For now, we'll use Python's AST parser which can handle some JS/TS
            # In a real implementation, you'd use a proper TypeScript parser
            return ast.parse(content)
        except SyntaxError:
            # If Python AST fails, we could fall back to regex-based analysis
            return None

    def _extract_imports(self, content: str) -> list[dict[str, Any]]:
        """Extract import statements from the content.

        Args:
            content: Source code content

        Returns:
            List of import information dictionaries
        """
        imports = []

        # Regex patterns for different import styles
        patterns = [
            # import { x, y } from 'module'
            r"import\s*{\s*([^}]+)\s*}\s*from\s*['\"]([^'\"]+)['\"]",
            # import x from 'module'
            r"import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import * as x from 'module'
            r"import\s*\*\s*as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]",
            # import 'module'
            r"import\s+['\"]([^'\"]+)['\"]",
        ]

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("import"):
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        import_info = {
                            "line_number": line_num,
                            "full_line": line,
                            "module": match.groups()[
                                -1
                            ],  # Last group is always the module
                            "type": self._classify_import_type(match.groups()[-1]),
                        }

                        # Add imported names if available
                        if len(match.groups()) > 1:
                            import_info["imports"] = match.groups()[0]

                        imports.append(import_info)
                        break

        return imports

    def _classify_import_type(self, module: str) -> str:
        """Classify the type of import based on the module path.

        Args:
            module: Module path

        Returns:
            Import type classification
        """
        if module.startswith("@/"):
            return "absolute_internal"
        elif module.startswith("./") or module.startswith("../"):
            return "relative"
        elif not module.startswith("."):
            return "external"
        else:
            return "unknown"

    def _calculate_file_score(
        self, patterns_found: list[PatternMatch], recommendations: list[Recommendation]
    ) -> float:
        """Calculate a score for the file based on patterns and recommendations.

        Args:
            patterns_found: List of patterns found in the file
            recommendations: List of recommendations for the file

        Returns:
            Score between 0.0 and 1.0
        """
        if not patterns_found and not recommendations:
            return 0.5  # Neutral score for files with no patterns

        # Start with base score
        score = 1.0

        # Subtract points for recommendations based on severity
        severity_penalties = {
            Severity.INFO: 0.01,
            Severity.WARNING: 0.05,
            Severity.ERROR: 0.15,
            Severity.CRITICAL: 0.25,
        }

        for rec in recommendations:
            score -= severity_penalties.get(rec.severity, 0.05)

        # Add points for good patterns
        pattern_bonus = min(0.3, len(patterns_found) * 0.02)
        score += pattern_bonus

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
