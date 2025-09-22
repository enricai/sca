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

"""Architectural analyzer for Next.js and React application patterns.

This analyzer focuses on architectural adherence including file structure,
import patterns, Next.js conventions, and overall code organization.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .base_analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    PatternMatch,
    PatternType,
    Recommendation,
    Severity,
)


class ArchitecturalAnalyzer(BaseAnalyzer):
    """Analyzes architectural patterns specific to Next.js and React applications.

    Focuses on:
    - File and directory structure
    - Import organization and patterns
    - Next.js conventions (app router, layouts, etc.)
    - Code organization and modularity
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the ArchitecturalAnalyzer with configuration.

        Args:
            config: Optional configuration dictionary for the analyzer.
        """
        super().__init__(config)
        self.nextjs_patterns = self._init_nextjs_patterns()
        self.import_patterns = self._init_import_patterns()

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'architectural'.
        """
        return "architectural"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.30 for 30% of overall score).
        """
        return 0.30  # 30% weight in overall scoring

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a file for architectural patterns and adherence.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with architectural analysis
        """
        start_time = time.time()
        patterns_found = []
        recommendations = []

        # Analyze different architectural aspects
        patterns_found.extend(self._analyze_file_structure(file_path))
        patterns_found.extend(self._analyze_import_organization(content))
        patterns_found.extend(self._analyze_nextjs_conventions(file_path, content))

        # Generate recommendations based on analysis
        recommendations.extend(
            self._generate_import_recommendations(file_path, content)
        )
        recommendations.extend(self._generate_structure_recommendations(file_path))
        recommendations.extend(
            self._generate_nextjs_recommendations(file_path, content)
        )

        # Calculate score based on patterns and recommendations
        score = self._calculate_architectural_score(patterns_found, recommendations)

        # Collect metrics
        metrics = {
            "import_count": len(self._extract_imports(content)),
            "file_size_lines": len(content.split("\n")),
            "nextjs_conventions_followed": len(
                [p for p in patterns_found if p.pattern_type == PatternType.FRAMEWORK]
            ),
            "import_organization_score": self._calculate_import_organization_score(
                content
            ),
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

    def _init_nextjs_patterns(self) -> dict[str, Any]:
        """Initialize Next.js specific patterns to look for."""
        return {
            "app_router_structure": {
                "patterns": [
                    r"src/app/\[[^/]+\]/.*",  # Dynamic routes
                    r"src/app/.*/layout\.tsx?$",  # Layout files
                    r"src/app/.*/page\.tsx?$",  # Page files
                    r"src/app/.*/loading\.tsx?$",  # Loading files
                    r"src/app/.*/error\.tsx?$",  # Error files
                    r"src/app/api/.*/route\.ts?$",  # API routes
                ],
                "score_bonus": 0.1,
            },
            "metadata_exports": {
                "patterns": [
                    r"export\s+(const\s+)?metadata\s*[:=]",
                    r"export\s+(const\s+)?viewport\s*[:=]",
                    r"export\s+async\s+function\s+generateMetadata",
                ],
                "score_bonus": 0.05,
            },
            "proper_exports": {
                "patterns": [
                    r"export\s+default\s+function\s+\w+",  # Default function export
                    r"export\s+\{\s*\w+\s*\}\s*$",  # Named exports
                ],
                "score_bonus": 0.03,
            },
        }

    def _init_import_patterns(self) -> dict[str, Any]:
        """Initialize import organization patterns."""
        return {
            "absolute_imports": {
                "pattern": r"import\s+.*\s+from\s+['\"]@/",
                "score_bonus": 0.1,
                "description": "Uses absolute imports with @/ alias",
            },
            "grouped_imports": {
                "description": "Imports are properly grouped (external, internal, relative)",
                "score_bonus": 0.05,
            },
            "sorted_imports": {
                "description": "Imports are alphabetically sorted within groups",
                "score_bonus": 0.03,
            },
        }

    def _analyze_file_structure(self, file_path: str) -> list[PatternMatch]:
        """Analyze file structure and naming conventions."""
        patterns = []
        path = Path(file_path)

        # Check if file follows Next.js app router structure
        if "src/app/" in file_path:
            for pattern_name, pattern_config in self.nextjs_patterns.items():
                if pattern_name == "app_router_structure":
                    for pattern in pattern_config["patterns"]:
                        if re.match(pattern, file_path):
                            patterns.append(
                                PatternMatch(
                                    pattern_type=PatternType.ARCHITECTURAL,
                                    pattern_name=f"nextjs_{pattern_name}",
                                    file_path=file_path,
                                    line_number=None,
                                    column=None,
                                    matched_text=file_path,
                                    confidence=0.9,
                                    context={
                                        "pattern": pattern,
                                        "structure": "app_router",
                                    },
                                )
                            )

        # Check component file naming
        if path.suffix in {".tsx", ".jsx"} and path.name[0].isupper():
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.NAMING,
                    pattern_name="component_naming",
                    file_path=file_path,
                    line_number=None,
                    column=None,
                    matched_text=path.name,
                    confidence=0.8,
                    context={"naming_convention": "PascalCase"},
                )
            )

        # Check directory structure
        if "src/components/" in file_path or "src/app/" in file_path:
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.ARCHITECTURAL,
                    pattern_name="proper_directory_structure",
                    file_path=file_path,
                    line_number=None,
                    column=None,
                    matched_text=str(path.parent),
                    confidence=0.7,
                    context={"directory": str(path.parent)},
                )
            )

        return patterns

    def _analyze_import_organization(self, content: str) -> list[PatternMatch]:
        """Analyze import organization and patterns."""
        patterns: list[PatternMatch] = []
        imports = self._extract_imports(content)

        if not imports:
            return patterns

        # Check for absolute imports
        absolute_imports = [
            imp for imp in imports if imp["type"] == "absolute_internal"
        ]
        if absolute_imports:
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.IMPORT,
                    pattern_name="absolute_imports",
                    file_path="",  # Will be set by caller
                    line_number=absolute_imports[0]["line_number"],
                    column=None,
                    matched_text=absolute_imports[0]["full_line"],
                    confidence=0.9,
                    context={"count": len(absolute_imports)},
                )
            )

        # Check import grouping
        if self._check_import_grouping(imports):
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.IMPORT,
                    pattern_name="grouped_imports",
                    file_path="",
                    line_number=imports[0]["line_number"],
                    column=None,
                    matched_text="Import grouping",
                    confidence=0.8,
                    context={"well_organized": True},
                )
            )

        return patterns

    def _analyze_nextjs_conventions(
        self, file_path: str, content: str
    ) -> list[PatternMatch]:
        """Analyze Next.js specific conventions."""
        patterns = []

        # Check for metadata exports
        for pattern_name, pattern_config in self.nextjs_patterns.items():
            if pattern_name == "metadata_exports":
                for pattern in pattern_config["patterns"]:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        patterns.append(
                            PatternMatch(
                                pattern_type=PatternType.FRAMEWORK,
                                pattern_name=f"nextjs_{pattern_name}",
                                file_path=file_path,
                                line_number=line_num,
                                column=match.start(),
                                matched_text=match.group(),
                                confidence=0.9,
                                context={"export_type": "metadata"},
                            )
                        )

        # Check for proper default exports in pages/layouts
        if any(x in file_path for x in ["/page.", "/layout.", "/loading.", "/error."]):
            default_export = re.search(r"export\s+default\s+function\s+(\w+)", content)
            if default_export:
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.FRAMEWORK,
                        pattern_name="nextjs_default_export",
                        file_path=file_path,
                        line_number=content[: default_export.start()].count("\n") + 1,
                        column=default_export.start(),
                        matched_text=default_export.group(),
                        confidence=0.9,
                        context={"function_name": default_export.group(1)},
                    )
                )

        return patterns

    def _generate_import_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate recommendations for import organization."""
        recommendations = []
        imports = self._extract_imports(content)

        # Check for relative imports that could be absolute
        relative_imports = [imp for imp in imports if imp["type"] == "relative"]
        deep_relative = [
            imp for imp in relative_imports if imp["module"].count("../") > 2
        ]

        if deep_relative:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="import_organization",
                    message="Consider using absolute imports (@/) instead of deep relative imports",
                    file_path=file_path,
                    line_number=deep_relative[0]["line_number"],
                    suggested_fix="Replace '../../../' with '@/' for better maintainability",
                    rule_id="PREFER_ABSOLUTE_IMPORTS",
                )
            )

        # Check import grouping
        if not self._check_import_grouping(imports):
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="import_organization",
                    message="Group imports: external packages, then internal modules, then relative imports",
                    file_path=file_path,
                    line_number=imports[0]["line_number"] if imports else None,
                    suggested_fix="Organize imports in groups separated by blank lines",
                    rule_id="GROUP_IMPORTS",
                )
            )

        return recommendations

    def _generate_structure_recommendations(
        self, file_path: str
    ) -> list[Recommendation]:
        """Generate recommendations for file structure."""
        recommendations = []
        path = Path(file_path)

        # Check component naming
        if path.suffix in {".tsx", ".jsx"} and "components" in file_path:
            if not path.stem[0].isupper():
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="naming_convention",
                        message="Component files should use PascalCase naming",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix=f"Rename to {path.stem.capitalize()}{path.suffix}",
                        rule_id="COMPONENT_PASCAL_CASE",
                    )
                )

        # Check for proper directory structure
        if path.suffix in {".tsx", ".jsx", ".ts", ".js"}:
            if not any(
                x in file_path
                for x in ["src/", "app/", "components/", "lib/", "types/"]
            ):
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="file_structure",
                        message="Consider organizing files in standard directories (src/, components/, lib/, etc.)",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Move file to appropriate directory structure",
                        rule_id="STANDARD_DIRECTORY_STRUCTURE",
                    )
                )

        return recommendations

    def _generate_nextjs_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate Next.js specific recommendations."""
        recommendations = []

        # Check for missing metadata in layout files
        if "/layout." in file_path and "metadata" not in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="nextjs_conventions",
                    message="Consider adding metadata export for SEO optimization",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="export const metadata: Metadata = { title: '...', description: '...' };",
                    rule_id="MISSING_METADATA",
                )
            )

        # Check for proper async/await in server components
        if "/page." in file_path and "async" in content and "use client" not in content:
            if not re.search(r"export\s+default\s+async\s+function", content):
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="nextjs_conventions",
                        message="Server components with async operations should be declared as async functions",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="export default async function ComponentName()",
                        rule_id="ASYNC_SERVER_COMPONENT",
                    )
                )

        return recommendations

    def _check_import_grouping(self, imports: list[dict[str, Any]]) -> bool:
        """Check if imports are properly grouped.

        Args:
            imports: List of import information

        Returns:
            True if imports are well organized
        """
        if len(imports) < 3:
            return True  # Too few imports to judge grouping

        # Group imports by type
        external = [i for i in imports if i["type"] == "external"]
        absolute_internal = [i for i in imports if i["type"] == "absolute_internal"]
        relative = [i for i in imports if i["type"] == "relative"]

        # Check if groups are in order and separated
        last_external_line = (
            max([i["line_number"] for i in external]) if external else 0
        )
        first_internal_line = (
            min([i["line_number"] for i in absolute_internal])
            if absolute_internal
            else float("inf")
        )
        first_relative_line = (
            min([i["line_number"] for i in relative]) if relative else float("inf")
        )

        # Basic grouping check: external -> internal -> relative
        return last_external_line < first_internal_line < first_relative_line

    def _calculate_import_organization_score(self, content: str) -> float:
        """Calculate a score for import organization quality."""
        imports = self._extract_imports(content)
        if not imports:
            return 1.0

        score = 0.5  # Base score

        # Bonus for absolute imports
        absolute_count = len([i for i in imports if i["type"] == "absolute_internal"])
        if absolute_count > 0:
            score += 0.2

        # Bonus for proper grouping
        if self._check_import_grouping(imports):
            score += 0.2

        # Penalty for deep relative imports
        deep_relative = len(
            [
                i
                for i in imports
                if i["type"] == "relative" and i["module"].count("../") > 2
            ]
        )
        if deep_relative > 0:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_architectural_score(
        self, patterns_found: list[PatternMatch], recommendations: list[Recommendation]
    ) -> float:
        """Calculate overall architectural score."""
        base_score = 0.6  # Start with neutral score

        # Add points for good patterns
        pattern_bonuses = {
            "nextjs_app_router_structure": 0.15,
            "nextjs_metadata_exports": 0.1,
            "absolute_imports": 0.1,
            "component_naming": 0.05,
            "proper_directory_structure": 0.05,
            "grouped_imports": 0.05,
        }

        for pattern in patterns_found:
            bonus = pattern_bonuses.get(pattern.pattern_name, 0.02)
            base_score += bonus * pattern.confidence

        # Subtract points for recommendations (issues found)
        severity_penalties = {
            Severity.INFO: 0.02,
            Severity.WARNING: 0.05,
            Severity.ERROR: 0.15,
            Severity.CRITICAL: 0.25,
        }

        for rec in recommendations:
            base_score -= severity_penalties.get(rec.severity, 0.02)

        return max(0.0, min(1.0, base_score))
