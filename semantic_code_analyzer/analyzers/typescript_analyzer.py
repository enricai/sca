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

"""TypeScript analyzer for type safety and TypeScript best practices.

This analyzer focuses on TypeScript usage patterns, type safety,
and TypeScript-specific best practices.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .base_analyzer import (
    AnalysisResult,
    BaseAnalyzer,
    PatternMatch,
    PatternType,
    Recommendation,
    Severity,
)


class TypeScriptAnalyzer(BaseAnalyzer):
    """Analyzes TypeScript usage patterns and type safety.

    Focuses on:
    - Type annotations and interfaces
    - Generic usage
    - Type safety patterns
    - TypeScript configuration adherence
    - Proper typing of React components
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the TypeScriptAnalyzer with configuration.

        Args:
            config: Optional configuration dictionary for the analyzer.
        """
        super().__init__(config)
        self.type_patterns = self._init_type_patterns()
        self.generic_patterns = self._init_generic_patterns()
        self.interface_patterns = self._init_interface_patterns()

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'typescript'.
        """
        return "typescript"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.20 for 20% of overall score).
        """
        return 0.20  # 20% weight in overall scoring

    def _get_supported_extensions(self) -> set[str]:
        """Typescript analyzer only handles TypeScript files."""
        return {".ts", ".tsx"}

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a TypeScript file for type safety and best practices.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with TypeScript analysis
        """
        start_time = time.time()
        patterns_found = []
        recommendations = []

        # Analyze TypeScript patterns
        patterns_found.extend(self._analyze_type_annotations(content))
        patterns_found.extend(self._analyze_interfaces_and_types(content))
        patterns_found.extend(self._analyze_generic_usage(content))
        patterns_found.extend(self._analyze_utility_types(content))

        # Generate recommendations
        recommendations.extend(
            self._generate_type_safety_recommendations(file_path, content)
        )
        recommendations.extend(
            self._generate_interface_recommendations(file_path, content)
        )
        recommendations.extend(
            self._generate_generic_recommendations(file_path, content)
        )

        score = self._calculate_typescript_score(
            patterns_found, recommendations, content
        )

        metrics = {
            "interfaces_defined": self._count_interfaces(content),
            "type_aliases_defined": self._count_type_aliases(content),
            "generic_usage_count": self._count_generic_usage(content),
            "any_usage_count": self._count_any_usage(content),
            "function_return_types": self._count_typed_functions(content),
            "strict_typing_score": self._calculate_strict_typing_score(content),
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

    def _init_type_patterns(self) -> dict[str, Any]:
        """Initialize TypeScript type patterns."""
        return {
            "explicit_return_types": {
                "pattern": r"function\s+\w+\s*\([^)]*\)\s*:\s*\w+",
                "score_bonus": 0.1,
                "description": "Functions have explicit return types",
            },
            "typed_parameters": {
                "pattern": r"function\s+\w+\s*\([^)]*:\s*\w+[^)]*\)",
                "score_bonus": 0.08,
                "description": "Function parameters are typed",
            },
            "const_assertions": {
                "pattern": r"as\s+const",
                "score_bonus": 0.05,
                "description": "Uses const assertions for literal types",
            },
            "type_guards": {
                "pattern": r"is\s+\w+",
                "score_bonus": 0.1,
                "description": "Implements type guards",
            },
        }

    def _init_generic_patterns(self) -> dict[str, Any]:
        """Initialize generic usage patterns."""
        return {
            "generic_functions": {
                "pattern": r"function\s+\w+<[^>]+>\s*\(",
                "score_bonus": 0.1,
                "description": "Uses generic functions",
            },
            "generic_interfaces": {
                "pattern": r"interface\s+\w+<[^>]+>",
                "score_bonus": 0.1,
                "description": "Defines generic interfaces",
            },
            "conditional_types": {
                "pattern": r"\?\s*:\s*[A-Z]\w*",
                "score_bonus": 0.15,
                "description": "Uses conditional types",
            },
            "mapped_types": {
                "pattern": r"\[K\s+in\s+keyof",
                "score_bonus": 0.15,
                "description": "Uses mapped types",
            },
        }

    def _init_interface_patterns(self) -> dict[str, Any]:
        """Initialize interface and type definition patterns."""
        return {
            "proper_interfaces": {
                "pattern": r"interface\s+[A-Z]\w*\s*\{",
                "score_bonus": 0.1,
                "description": "Defines interfaces with proper naming",
            },
            "type_aliases": {
                "pattern": r"type\s+[A-Z]\w*\s*=",
                "score_bonus": 0.08,
                "description": "Uses type aliases",
            },
            "union_types": {
                "pattern": r":\s*\w+\s*\|\s*\w+",
                "score_bonus": 0.05,
                "description": "Uses union types",
            },
            "intersection_types": {
                "pattern": r":\s*\w+\s*&\s*\w+",
                "score_bonus": 0.05,
                "description": "Uses intersection types",
            },
        }

    def _analyze_type_annotations(self, content: str) -> list[PatternMatch]:
        """Analyze type annotations in the code."""
        patterns = []

        # Check for explicit return types
        return_types = list(
            re.finditer(self.type_patterns["explicit_return_types"]["pattern"], content)
        )
        for match in return_types:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="explicit_return_type",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"type_safety": "return_type"},
                )
            )

        # Check for typed parameters
        typed_params = list(
            re.finditer(self.type_patterns["typed_parameters"]["pattern"], content)
        )
        for match in typed_params:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="typed_parameters",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.8,
                    context={"type_safety": "parameter_typing"},
                )
            )

        # Check for const assertions
        const_assertions = list(
            re.finditer(self.type_patterns["const_assertions"]["pattern"], content)
        )
        for match in const_assertions:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="const_assertion",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"type_safety": "const_assertion"},
                )
            )

        # Check for type guards
        type_guards = list(
            re.finditer(self.type_patterns["type_guards"]["pattern"], content)
        )
        for match in type_guards:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="type_guard",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"type_safety": "type_guard"},
                )
            )

        return patterns

    def _analyze_interfaces_and_types(self, content: str) -> list[PatternMatch]:
        """Analyze interface and type definitions."""
        patterns = []

        # Check for interfaces
        interfaces = list(
            re.finditer(
                self.interface_patterns["proper_interfaces"]["pattern"], content
            )
        )
        for match in interfaces:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="interface_definition",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"definition_type": "interface"},
                )
            )

        # Check for type aliases
        type_aliases = list(
            re.finditer(self.interface_patterns["type_aliases"]["pattern"], content)
        )
        for match in type_aliases:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="type_alias",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.8,
                    context={"definition_type": "type_alias"},
                )
            )

        # Check for union types
        union_types = list(
            re.finditer(self.interface_patterns["union_types"]["pattern"], content)
        )
        for match in union_types:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="union_type",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.7,
                    context={"type_composition": "union"},
                )
            )

        return patterns

    def _analyze_generic_usage(self, content: str) -> list[PatternMatch]:
        """Analyze generic type usage."""
        patterns = []

        # Check for generic functions
        generic_functions = list(
            re.finditer(self.generic_patterns["generic_functions"]["pattern"], content)
        )
        for match in generic_functions:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="generic_function",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"generic_usage": "function"},
                )
            )

        # Check for generic interfaces
        generic_interfaces = list(
            re.finditer(self.generic_patterns["generic_interfaces"]["pattern"], content)
        )
        for match in generic_interfaces:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.TYPE_SAFETY,
                    pattern_name="generic_interface",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.9,
                    context={"generic_usage": "interface"},
                )
            )

        return patterns

    def _analyze_utility_types(self, content: str) -> list[PatternMatch]:
        """Analyze usage of TypeScript utility types."""
        patterns = []

        utility_types = [
            "Partial",
            "Required",
            "Readonly",
            "Record",
            "Pick",
            "Omit",
            "Exclude",
            "Extract",
            "NonNullable",
            "Parameters",
            "ReturnType",
        ]

        for utility_type in utility_types:
            utility_matches = list(re.finditer(rf"\b{utility_type}<", content))
            for match in utility_matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.TYPE_SAFETY,
                        pattern_name="utility_type",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=utility_type,
                        confidence=0.9,
                        context={"utility_type": utility_type},
                    )
                )

        return patterns

    def _generate_type_safety_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate type safety recommendations."""
        recommendations = []

        # Check for 'any' usage
        any_usage = list(re.finditer(r":\s*any\b", content))
        if any_usage:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="type_safety",
                    message="Avoid using 'any' type - use specific types instead",
                    file_path=file_path,
                    line_number=content[: any_usage[0].start()].count("\n") + 1,
                    suggested_fix="Replace 'any' with specific type or unknown",
                    rule_id="AVOID_ANY_TYPE",
                )
            )

        # Check for untyped function parameters
        untyped_params = list(
            re.finditer(r"function\s+\w+\s*\([^:)]*\w+[^:)]*\)", content)
        )
        for match in untyped_params:
            if ":" not in match.group():
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="type_safety",
                        message="Add type annotations to function parameters",
                        file_path=file_path,
                        line_number=content[: match.start()].count("\n") + 1,
                        suggested_fix="function name(param: Type): ReturnType",
                        rule_id="ADD_PARAMETER_TYPES",
                    )
                )

        # Check for missing return types
        functions_without_return_type = list(
            re.finditer(r"function\s+\w+\s*\([^)]*\)\s*\{", content)
        )
        for match in functions_without_return_type:
            if ":" not in match.group():
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="type_safety",
                        message="Add explicit return type to functions",
                        file_path=file_path,
                        line_number=content[: match.start()].count("\n") + 1,
                        suggested_fix="function name(): ReturnType",
                        rule_id="ADD_RETURN_TYPES",
                    )
                )

        # Check for non-null assertions
        non_null_assertions = list(re.finditer(r"!\s*\.", content))
        if len(non_null_assertions) > 3:  # Too many non-null assertions
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="type_safety",
                    message="Excessive use of non-null assertions - consider proper null checking",
                    file_path=file_path,
                    line_number=content[: non_null_assertions[0].start()].count("\n")
                    + 1,
                    suggested_fix="Use optional chaining (?.) or proper null checks",
                    rule_id="EXCESSIVE_NON_NULL_ASSERTIONS",
                )
            )

        return recommendations

    def _generate_interface_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate interface and type definition recommendations."""
        recommendations = []

        # Check for inline object types that could be interfaces
        inline_objects = list(re.finditer(r":\s*\{[^}]+\}", content))
        if len(inline_objects) > 2:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="type_organization",
                    message="Consider extracting inline object types to interfaces",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="interface TypeName { ... } and use TypeName",
                    rule_id="EXTRACT_INLINE_TYPES",
                )
            )

        # Check for repetitive type patterns
        if content.count("string | null") > 3:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="type_organization",
                    message="Consider creating a type alias for repeated type patterns",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="type NullableString = string | null",
                    rule_id="CREATE_TYPE_ALIAS",
                )
            )

        return recommendations

    def _generate_generic_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate generic usage recommendations."""
        recommendations = []

        # Check for functions that could benefit from generics
        if re.search(r"function\s+\w+.*Object\.keys", content) and "<" not in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="generics",
                    message="Consider using generics for type-safe object manipulation",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="function name<T>(obj: T): (keyof T)[]",
                    rule_id="USE_GENERICS_FOR_OBJECT_UTILS",
                )
            )

        return recommendations

    def _count_interfaces(self, content: str) -> int:
        """Count interface definitions."""
        return len(re.findall(r"interface\s+\w+", content))

    def _count_type_aliases(self, content: str) -> int:
        """Count type alias definitions."""
        return len(re.findall(r"type\s+\w+\s*=", content))

    def _count_generic_usage(self, content: str) -> int:
        """Count generic usage instances."""
        return len(re.findall(r"<[^>]+>", content))

    def _count_any_usage(self, content: str) -> int:
        """Count 'any' type usage."""
        return len(re.findall(r":\s*any\b", content))

    def _count_typed_functions(self, content: str) -> int:
        """Count functions with return type annotations."""
        return len(re.findall(r"function\s+\w+\s*\([^)]*\)\s*:\s*\w+", content))

    def _calculate_strict_typing_score(self, content: str) -> float:
        """Calculate a score for strict typing adherence."""
        total_functions = len(re.findall(r"function\s+\w+", content))
        if total_functions == 0:
            return 1.0

        typed_functions = self._count_typed_functions(content)
        any_usage = self._count_any_usage(content)

        # Base score from typed functions ratio
        typed_ratio = typed_functions / total_functions
        score = typed_ratio

        # Penalty for any usage
        any_penalty = min(0.3, any_usage * 0.05)
        score -= any_penalty

        # Bonus for interfaces and type aliases
        interfaces = self._count_interfaces(content)
        type_aliases = self._count_type_aliases(content)
        definition_bonus = min(0.2, (interfaces + type_aliases) * 0.02)
        score += definition_bonus

        return max(0.0, min(1.0, score))

    def _calculate_typescript_score(
        self,
        patterns_found: list[PatternMatch],
        recommendations: list[Recommendation],
        content: str,
    ) -> float:
        """Calculate overall TypeScript score."""
        base_score = 0.5

        # Add points for good patterns
        pattern_bonuses = {
            "explicit_return_type": 0.1,
            "typed_parameters": 0.08,
            "interface_definition": 0.1,
            "type_alias": 0.08,
            "generic_function": 0.12,
            "generic_interface": 0.12,
            "utility_type": 0.1,
            "type_guard": 0.15,
            "const_assertion": 0.05,
        }

        for pattern in patterns_found:
            bonus = pattern_bonuses.get(pattern.pattern_name, 0.02)
            base_score += bonus * pattern.confidence

        # Subtract for issues
        severity_penalties = {
            Severity.INFO: 0.02,
            Severity.WARNING: 0.08,
            Severity.ERROR: 0.15,
            Severity.CRITICAL: 0.25,
        }

        for rec in recommendations:
            base_score -= severity_penalties.get(rec.severity, 0.02)

        # Factor in strict typing score
        strict_typing_score = self._calculate_strict_typing_score(content)
        base_score = (base_score + strict_typing_score) / 2

        return max(0.0, min(1.0, base_score))
