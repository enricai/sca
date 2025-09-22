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

"""Quality analyzer for React components and general code quality patterns.

This analyzer focuses on React component best practices, performance patterns,
code quality, security, and accessibility considerations.
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


class QualityAnalyzer(BaseAnalyzer):
    """Analyzes code quality patterns with focus on React components.

    Focuses on:
    - React component patterns and best practices
    - Performance optimization patterns
    - Security considerations
    - Accessibility patterns
    - Error handling patterns
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the QualityAnalyzer with configuration.

        Args:
            config: Optional configuration dictionary for the analyzer.
        """
        super().__init__(config)
        self.react_patterns = self._init_react_patterns()
        self.performance_patterns = self._init_performance_patterns()
        self.security_patterns = self._init_security_patterns()
        self.accessibility_patterns = self._init_accessibility_patterns()

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'quality'.
        """
        return "quality"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.25 for 25% of overall score).
        """
        return 0.25  # 25% weight in overall scoring

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a file for quality patterns and best practices.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with quality analysis
        """
        start_time = time.time()
        patterns_found = []
        recommendations = []

        # Only analyze React/TypeScript files for component patterns
        if self._is_component_file(file_path):
            patterns_found.extend(self._analyze_react_patterns(content))
            patterns_found.extend(self._analyze_performance_patterns(content))
            patterns_found.extend(self._analyze_accessibility_patterns(content))

            recommendations.extend(
                self._generate_react_recommendations(file_path, content)
            )
            recommendations.extend(
                self._generate_performance_recommendations(file_path, content)
            )
            recommendations.extend(
                self._generate_accessibility_recommendations(file_path, content)
            )

        # Security analysis for all file types
        patterns_found.extend(self._analyze_security_patterns(content))
        recommendations.extend(
            self._generate_security_recommendations(file_path, content)
        )

        # General quality analysis
        patterns_found.extend(self._analyze_error_handling(content))
        recommendations.extend(
            self._generate_error_handling_recommendations(file_path, content)
        )

        score = self._calculate_quality_score(patterns_found, recommendations, content)

        metrics = {
            "component_patterns_found": len(
                [p for p in patterns_found if p.pattern_type == PatternType.COMPONENT]
            ),
            "security_patterns_found": len(
                [p for p in patterns_found if p.pattern_type == PatternType.SECURITY]
            ),
            "performance_patterns_found": len(
                [p for p in patterns_found if p.pattern_type == PatternType.PERFORMANCE]
            ),
            "accessibility_patterns_found": len(
                [
                    p
                    for p in patterns_found
                    if p.pattern_type == PatternType.ACCESSIBILITY
                ]
            ),
            "hooks_count": self._count_react_hooks(content),
            "function_component": self._is_function_component(content),
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

    def _init_react_patterns(self) -> dict[str, Any]:
        """Initialize React component patterns."""
        return {
            "function_component": {
                "pattern": r"(export\s+default\s+)?function\s+\w+\s*\([^)]*\)\s*:\s*React\.ReactElement",
                "score_bonus": 0.1,
                "description": "Uses function components with proper TypeScript typing",
            },
            "proper_props_interface": {
                "pattern": r"interface\s+\w+Props\s*\{",
                "score_bonus": 0.1,
                "description": "Defines proper TypeScript interface for props",
            },
            "use_client_directive": {
                "pattern": r"['\"]use client['\"]",
                "score_bonus": 0.05,
                "description": "Properly marks client components",
            },
            "react_hooks": {
                "patterns": [
                    r"useState\s*\(",
                    r"useEffect\s*\(",
                    r"useCallback\s*\(",
                    r"useMemo\s*\(",
                    r"useContext\s*\(",
                    r"useReducer\s*\(",
                ],
                "score_bonus": 0.05,
            },
            "custom_hooks": {
                "pattern": r"use[A-Z]\w*\s*\(",
                "score_bonus": 0.1,
                "description": "Uses custom hooks",
            },
        }

    def _init_performance_patterns(self) -> dict[str, Any]:
        """Initialize performance optimization patterns."""
        return {
            "memo_usage": {
                "pattern": r"React\.memo\s*\(",
                "score_bonus": 0.1,
                "description": "Uses React.memo for performance optimization",
            },
            "use_callback": {
                "pattern": r"useCallback\s*\(",
                "score_bonus": 0.1,
                "description": "Uses useCallback to memoize functions",
            },
            "use_memo": {
                "pattern": r"useMemo\s*\(",
                "score_bonus": 0.1,
                "description": "Uses useMemo for expensive calculations",
            },
            "lazy_loading": {
                "patterns": [
                    r"React\.lazy\s*\(",
                    r"dynamic\s*\(",
                    r"import\s*\(['\"].*['\"]\)",
                ],
                "score_bonus": 0.15,
                "description": "Implements lazy loading",
            },
        }

    def _init_security_patterns(self) -> dict[str, Any]:
        """Initialize security patterns."""
        return {
            "dangerously_set_innerHTML": {
                "pattern": r"dangerouslySetInnerHTML",
                "score_penalty": 0.2,
                "description": "Uses potentially unsafe dangerouslySetInnerHTML",
            },
            "eval_usage": {
                "pattern": r"\beval\s*\(",
                "score_penalty": 0.3,
                "description": "Uses dangerous eval() function",
            },
            "console_log": {
                "pattern": r"console\.(log|debug|info|warn|error)",
                "score_penalty": 0.05,
                "description": "Contains console statements (should be removed in production)",
            },
            "proper_validation": {
                "patterns": [
                    r"zod\.",
                    r"\.safeParse\(",
                    r"joi\.",
                    r"yup\.",
                ],
                "score_bonus": 0.1,
                "description": "Uses proper validation library",
            },
        }

    def _init_accessibility_patterns(self) -> dict[str, Any]:
        """Initialize accessibility patterns."""
        return {
            "aria_attributes": {
                "pattern": r"aria-\w+",
                "score_bonus": 0.1,
                "description": "Uses ARIA attributes for accessibility",
            },
            "semantic_html": {
                "patterns": [
                    r"<(main|nav|header|footer|section|article|aside)",
                    r"role=['\"]",
                ],
                "score_bonus": 0.1,
                "description": "Uses semantic HTML elements",
            },
            "alt_text": {
                "pattern": r"alt=['\"][^'\"]*['\"]",
                "score_bonus": 0.05,
                "description": "Provides alt text for images",
            },
            "tab_index": {
                "pattern": r"tabIndex",
                "score_bonus": 0.05,
                "description": "Manages keyboard navigation",
            },
        }

    def _is_component_file(self, file_path: str) -> bool:
        """Check if file is likely a React component."""
        return file_path.endswith((".tsx", ".jsx")) and (
            "components" in file_path
            or file_path.endswith("/page.tsx")
            or file_path.endswith("/layout.tsx")
        )

    def _analyze_react_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze React component patterns."""
        patterns = []

        # Check function component pattern
        func_component = re.search(
            self.react_patterns["function_component"]["pattern"], content
        )
        if func_component:
            line_num = content[: func_component.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.COMPONENT,
                    pattern_name="function_component",
                    file_path="",
                    line_number=line_num,
                    column=func_component.start(),
                    matched_text=func_component.group(),
                    confidence=0.9,
                    context={"component_type": "function"},
                )
            )

        # Check props interface
        props_interface = re.search(
            self.react_patterns["proper_props_interface"]["pattern"], content
        )
        if props_interface:
            line_num = content[: props_interface.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.COMPONENT,
                    pattern_name="props_interface",
                    file_path="",
                    line_number=line_num,
                    column=props_interface.start(),
                    matched_text=props_interface.group(),
                    confidence=0.9,
                    context={"typing": "typescript_interface"},
                )
            )

        # Check React hooks usage
        for hook_pattern in self.react_patterns["react_hooks"]["patterns"]:
            hooks = list(re.finditer(hook_pattern, content))
            for hook in hooks:
                line_num = content[: hook.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.COMPONENT,
                        pattern_name="react_hook",
                        file_path="",
                        line_number=line_num,
                        column=hook.start(),
                        matched_text=hook.group(),
                        confidence=0.8,
                        context={"hook_type": hook.group().split("(")[0]},
                    )
                )

        # Check use client directive
        use_client = re.search(
            self.react_patterns["use_client_directive"]["pattern"], content
        )
        if use_client:
            line_num = content[: use_client.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.COMPONENT,
                    pattern_name="use_client_directive",
                    file_path="",
                    line_number=line_num,
                    column=use_client.start(),
                    matched_text=use_client.group(),
                    confidence=0.9,
                    context={"component_type": "client"},
                )
            )

        return patterns

    def _analyze_performance_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze performance optimization patterns."""
        patterns = []

        # Check for React.memo
        memo_usage = re.search(
            self.performance_patterns["memo_usage"]["pattern"], content
        )
        if memo_usage:
            line_num = content[: memo_usage.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.PERFORMANCE,
                    pattern_name="memo_usage",
                    file_path="",
                    line_number=line_num,
                    column=memo_usage.start(),
                    matched_text=memo_usage.group(),
                    confidence=0.9,
                    context={"optimization": "memoization"},
                )
            )

        # Check for lazy loading patterns
        for lazy_pattern in self.performance_patterns["lazy_loading"]["patterns"]:
            lazy_matches = list(re.finditer(lazy_pattern, content))
            for match in lazy_matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.PERFORMANCE,
                        pattern_name="lazy_loading",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"optimization": "lazy_loading"},
                    )
                )

        return patterns

    def _analyze_security_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze security patterns and potential issues."""
        patterns = []

        # Check for validation libraries
        for validation_pattern in self.security_patterns["proper_validation"][
            "patterns"
        ]:
            validation_matches = list(re.finditer(validation_pattern, content))
            for match in validation_matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.SECURITY,
                        pattern_name="proper_validation",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"security": "input_validation"},
                    )
                )

        return patterns

    def _analyze_accessibility_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze accessibility patterns."""
        patterns = []

        # Check for ARIA attributes
        aria_matches = list(
            re.finditer(
                self.accessibility_patterns["aria_attributes"]["pattern"], content
            )
        )
        for match in aria_matches:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.ACCESSIBILITY,
                    pattern_name="aria_attributes",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.8,
                    context={"accessibility": "aria"},
                )
            )

        # Check for semantic HTML
        for semantic_pattern in self.accessibility_patterns["semantic_html"][
            "patterns"
        ]:
            semantic_matches = list(re.finditer(semantic_pattern, content))
            for match in semantic_matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.ACCESSIBILITY,
                        pattern_name="semantic_html",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.7,
                        context={"accessibility": "semantic_html"},
                    )
                )

        return patterns

    def _analyze_error_handling(self, content: str) -> list[PatternMatch]:
        """Analyze error handling patterns."""
        patterns = []

        # Check for try-catch blocks
        try_catch = list(
            re.finditer(r"try\s*\{[^}]*\}\s*catch\s*\([^)]*\)\s*\{", content, re.DOTALL)
        )
        for match in try_catch:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.COMPONENT,
                    pattern_name="error_handling",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text="try-catch block",
                    confidence=0.8,
                    context={"error_handling": "try_catch"},
                )
            )

        # Check for error boundaries
        error_boundary = re.search(
            r"componentDidCatch|getDerivedStateFromError", content
        )
        if error_boundary:
            line_num = content[: error_boundary.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.COMPONENT,
                    pattern_name="error_boundary",
                    file_path="",
                    line_number=line_num,
                    column=error_boundary.start(),
                    matched_text=error_boundary.group(),
                    confidence=0.9,
                    context={"error_handling": "error_boundary"},
                )
            )

        return patterns

    def _generate_react_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate React-specific recommendations."""
        recommendations = []

        # Check if it's a component file but doesn't use function components
        if self._is_component_file(file_path):
            if not self._is_function_component(content):
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="react_patterns",
                        message="Consider using function components instead of class components",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Convert to function component with hooks",
                        rule_id="PREFER_FUNCTION_COMPONENTS",
                    )
                )

            # Check for props interface
            if "interface" not in content and "type.*Props" not in content:
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="react_patterns",
                        message="Define TypeScript interface or type for component props",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="interface ComponentNameProps { ... }",
                        rule_id="MISSING_PROPS_INTERFACE",
                    )
                )

        return recommendations

    def _generate_performance_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate performance-related recommendations."""
        recommendations = []

        # Check for missing useCallback on functions passed as props
        if "onClick" in content and "useCallback" not in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="performance",
                    message="Consider using useCallback for event handlers to prevent unnecessary re-renders",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="const handleClick = useCallback(() => { ... }, [dependencies])",
                    rule_id="USE_CALLBACK_FOR_HANDLERS",
                )
            )

        # Check for expensive operations without useMemo
        if (
            re.search(r"\.filter\(|\.map\(|\.reduce\(", content)
            and "useMemo" not in content
        ):
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="performance",
                    message="Consider using useMemo for expensive array operations",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="const processedData = useMemo(() => data.filter(...), [data])",
                    rule_id="USE_MEMO_FOR_EXPENSIVE_OPS",
                )
            )

        return recommendations

    def _generate_security_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate security-related recommendations."""
        recommendations = []

        # Check for dangerouslySetInnerHTML
        dangerous_html = re.search(
            self.security_patterns["dangerously_set_innerHTML"]["pattern"], content
        )
        if dangerous_html:
            line_num = content[: dangerous_html.start()].count("\n") + 1
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="security",
                    message="dangerouslySetInnerHTML can lead to XSS vulnerabilities",
                    file_path=file_path,
                    line_number=line_num,
                    suggested_fix="Use proper sanitization or avoid dangerouslySetInnerHTML",
                    rule_id="DANGEROUS_SET_INNER_HTML",
                )
            )

        # Check for console statements
        console_statements = list(
            re.finditer(self.security_patterns["console_log"]["pattern"], content)
        )
        if console_statements:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="security",
                    message="Remove console statements before production deployment",
                    file_path=file_path,
                    line_number=content[: console_statements[0].start()].count("\n")
                    + 1,
                    suggested_fix="Remove or replace with proper logging",
                    rule_id="REMOVE_CONSOLE_STATEMENTS",
                )
            )

        return recommendations

    def _generate_accessibility_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate accessibility recommendations."""
        recommendations = []

        # Check for images without alt text
        img_without_alt = re.search(r"<img(?![^>]*alt=)", content)
        if img_without_alt:
            line_num = content[: img_without_alt.start()].count("\n") + 1
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="accessibility",
                    message="Images should have alt text for screen readers",
                    file_path=file_path,
                    line_number=line_num,
                    suggested_fix='Add alt="description" attribute to images',
                    rule_id="MISSING_ALT_TEXT",
                )
            )

        # Check for interactive elements without proper accessibility
        buttons_without_aria = re.search(r"<button(?![^>]*aria-)", content)
        if buttons_without_aria and "onClick" in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="accessibility",
                    message="Consider adding ARIA labels for complex interactive elements",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Add aria-label or aria-describedby attributes",
                    rule_id="MISSING_ARIA_LABELS",
                )
            )

        return recommendations

    def _generate_error_handling_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate error handling recommendations."""
        recommendations = []

        # Check for async operations without error handling
        if "async" in content and "await" in content and "try" not in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="error_handling",
                    message="Async operations should include proper error handling",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Wrap async operations in try-catch blocks",
                    rule_id="MISSING_ASYNC_ERROR_HANDLING",
                )
            )

        return recommendations

    def _count_react_hooks(self, content: str) -> int:
        """Count the number of React hooks used."""
        hook_patterns = [
            r"useState\s*\(",
            r"useEffect\s*\(",
            r"useCallback\s*\(",
            r"useMemo\s*\(",
            r"useContext\s*\(",
            r"useReducer\s*\(",
        ]

        count = 0
        for pattern in hook_patterns:
            count += len(re.findall(pattern, content))

        return count

    def _is_function_component(self, content: str) -> bool:
        """Check if the file contains a function component."""
        return bool(
            re.search(r"function\s+\w+\s*\([^)]*\)\s*:\s*React\.ReactElement", content)
        )

    def _calculate_quality_score(
        self,
        patterns_found: list[PatternMatch],
        recommendations: list[Recommendation],
        content: str,
    ) -> float:
        """Calculate overall quality score."""
        base_score = 0.6

        # Bonus for good patterns
        pattern_bonuses = {
            PatternType.COMPONENT: 0.05,
            PatternType.PERFORMANCE: 0.1,
            PatternType.SECURITY: 0.1,
            PatternType.ACCESSIBILITY: 0.08,
        }

        for pattern in patterns_found:
            bonus = pattern_bonuses.get(pattern.pattern_type, 0.02)
            base_score += bonus * pattern.confidence

        # Penalty for issues
        severity_penalties = {
            Severity.INFO: 0.02,
            Severity.WARNING: 0.08,
            Severity.ERROR: 0.15,
            Severity.CRITICAL: 0.25,
        }

        for rec in recommendations:
            base_score -= severity_penalties.get(rec.severity, 0.02)

        # Additional scoring factors
        lines = len(content.split("\n"))
        if lines > 500:  # Penalty for very long files
            base_score -= 0.05

        return max(0.0, min(1.0, base_score))
