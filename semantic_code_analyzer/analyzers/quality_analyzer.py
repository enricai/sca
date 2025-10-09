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
        """Initialize comprehensive React component patterns including React 18+ features."""
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
            # Next.js App Router patterns
            "use_client_directive": {
                "pattern": r"['\"]use client['\"]",
                "score_bonus": 0.08,
                "description": "Properly marks client components for Next.js App Router",
            },
            "use_server_directive": {
                "pattern": r"['\"]use server['\"]",
                "score_bonus": 0.08,
                "description": "Properly marks server actions for Next.js App Router",
            },
            "app_router_layout": {
                "pattern": r"export\s+default\s+function\s+.*Layout\s*\(",
                "score_bonus": 0.1,
                "description": "Uses Next.js App Router layout pattern",
            },
            "app_router_page": {
                "pattern": r"export\s+default\s+function\s+.*Page\s*\(",
                "score_bonus": 0.08,
                "description": "Uses Next.js App Router page pattern",
            },
            "app_router_loading": {
                "pattern": r"export\s+default\s+function\s+Loading\s*\(",
                "score_bonus": 0.06,
                "description": "Implements Next.js App Router loading UI",
            },
            "app_router_error": {
                "pattern": r"export\s+default\s+function\s+Error\s*\(",
                "score_bonus": 0.08,
                "description": "Implements Next.js App Router error UI",
            },
            "app_router_not_found": {
                "pattern": r"export\s+default\s+function\s+NotFound\s*\(",
                "score_bonus": 0.06,
                "description": "Implements Next.js App Router 404 UI",
            },
            # React 18+ concurrent features
            "suspense_usage": {
                "pattern": r"<Suspense\s+fallback",
                "score_bonus": 0.12,
                "description": "Uses React 18 Suspense for concurrent rendering",
            },
            "concurrent_features": {
                "patterns": [
                    r"useDeferredValue\s*\(",
                    r"useTransition\s*\(",
                    r"startTransition\s*\(",
                    r"React\.startTransition",
                ],
                "score_bonus": 0.15,
                "description": "Uses React 18 concurrent features",
            },
            "react_hooks": {
                "patterns": [
                    r"useState\s*\(",
                    r"useEffect\s*\(",
                    r"useCallback\s*\(",
                    r"useMemo\s*\(",
                    r"useContext\s*\(",
                    r"useReducer\s*\(",
                    r"useRef\s*\(",
                    r"useImperativeHandle\s*\(",
                    r"useLayoutEffect\s*\(",
                    r"useDebugValue\s*\(",
                    # React 18+ hooks
                    r"useId\s*\(",
                    r"useSyncExternalStore\s*\(",
                    r"useInsertionEffect\s*\(",
                ],
                "score_bonus": 0.05,
                "description": "Uses React hooks including React 18+ hooks",
            },
            "custom_hooks": {
                "pattern": r"use[A-Z]\w*\s*\(",
                "score_bonus": 0.1,
                "description": "Uses custom hooks",
            },
            # Server components patterns
            "server_component": {
                "pattern": r"async\s+function\s+\w+\s*\([^)]*\)\s*:\s*Promise<",
                "score_bonus": 0.12,
                "description": "Uses React Server Components pattern",
            },
            "server_only_import": {
                "pattern": r"import\s+.*['\"]server-only['\"]",
                "score_bonus": 0.08,
                "description": "Properly isolates server-only code",
            },
            "client_only_import": {
                "pattern": r"import\s+.*['\"]client-only['\"]",
                "score_bonus": 0.08,
                "description": "Properly isolates client-only code",
            },
            # Advanced React patterns
            "forward_ref": {
                "pattern": r"forwardRef\s*\(",
                "score_bonus": 0.08,
                "description": "Uses React forwardRef for ref forwarding",
            },
            "react_portal": {
                "pattern": r"createPortal\s*\(",
                "score_bonus": 0.08,
                "description": "Uses React Portal for rendering outside tree",
            },
            "context_provider": {
                "pattern": r"\.Provider\s+value=",
                "score_bonus": 0.10,
                "description": "Implements React Context provider pattern",
            },
            # Modern TypeScript React patterns
            "generic_components": {
                "pattern": r"function\s+\w+<[^>]+>\s*\(",
                "score_bonus": 0.12,
                "description": "Uses generic TypeScript React components",
            },
            "discriminated_props": {
                "pattern": r"type\s+\w+Props\s*=\s*\{[^}]*\}\s*&\s*\(",
                "score_bonus": 0.10,
                "description": "Uses discriminated union props pattern",
            },
            "as_prop_pattern": {
                "pattern": r"as\?:\s*(keyof\s+JSX\.IntrinsicElements|React\.ElementType)",
                "score_bonus": 0.10,
                "description": "Implements polymorphic 'as' prop pattern",
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
        """Analyze comprehensive React component patterns including modern features."""
        patterns = []

        # Single pattern checks
        single_patterns = [
            "function_component",
            "proper_props_interface",
            "use_client_directive",
            "use_server_directive",
            "app_router_layout",
            "app_router_page",
            "app_router_loading",
            "app_router_error",
            "app_router_not_found",
            "suspense_usage",
            "server_component",
            "server_only_import",
            "client_only_import",
            "forward_ref",
            "react_portal",
            "context_provider",
            "generic_components",
            "discriminated_props",
            "as_prop_pattern",
            "custom_hooks",
        ]

        for pattern_name in single_patterns:
            if pattern_name in self.react_patterns:
                pattern_config = self.react_patterns[pattern_name]
                if "pattern" in pattern_config:
                    matches = list(re.finditer(pattern_config["pattern"], content))
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        patterns.append(
                            PatternMatch(
                                pattern_type=PatternType.COMPONENT,
                                pattern_name=pattern_name,
                                file_path="",
                                line_number=line_num,
                                column=match.start(),
                                matched_text=match.group(),
                                confidence=0.9,
                                context=self._get_pattern_context(
                                    pattern_name, match.group()
                                ),
                            )
                        )

        # Multiple pattern checks (patterns with "patterns" array)
        multi_patterns = ["react_hooks", "concurrent_features"]

        for pattern_name in multi_patterns:
            if (
                pattern_name in self.react_patterns
                and "patterns" in self.react_patterns[pattern_name]
            ):
                for pattern in self.react_patterns[pattern_name]["patterns"]:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        patterns.append(
                            PatternMatch(
                                pattern_type=PatternType.COMPONENT,
                                pattern_name=pattern_name,
                                file_path="",
                                line_number=line_num,
                                column=match.start(),
                                matched_text=match.group(),
                                confidence=0.8,
                                context=self._get_pattern_context(
                                    pattern_name, match.group()
                                ),
                            )
                        )

        return patterns

    def _get_pattern_context(
        self, pattern_name: str, matched_text: str
    ) -> dict[str, Any]:
        """Get contextual information for a matched pattern."""
        context = {"pattern_name": pattern_name}

        if pattern_name == "function_component":
            context["component_type"] = "function"
        elif pattern_name == "proper_props_interface":
            context["typing"] = "typescript_interface"
        elif pattern_name == "use_client_directive":
            context["component_type"] = "client"
            context["nextjs_feature"] = "app_router"
        elif pattern_name == "use_server_directive":
            context["component_type"] = "server_action"
            context["nextjs_feature"] = "app_router"
        elif pattern_name.startswith("app_router_"):
            context["nextjs_feature"] = "app_router"
            context["special_file"] = pattern_name.replace("app_router_", "")
        elif pattern_name == "server_component":
            context["component_type"] = "server"
            context["react_feature"] = "server_components"
        elif pattern_name in ["server_only_import", "client_only_import"]:
            context["boundary"] = pattern_name.replace("_import", "")
        elif pattern_name == "suspense_usage":
            context["react_feature"] = "concurrent_rendering"
        elif pattern_name == "concurrent_features":
            context["react_feature"] = "concurrent_features"
            context["hook_type"] = (
                matched_text.split("(")[0] if "(" in matched_text else matched_text
            )
        elif pattern_name == "react_hooks":
            hook_name = (
                matched_text.split("(")[0] if "(" in matched_text else matched_text
            )
            context["hook_type"] = hook_name
            # Identify React 18+ hooks
            if hook_name in [
                "useId",
                "useDeferredValue",
                "useTransition",
                "useSyncExternalStore",
                "useInsertionEffect",
            ]:
                context["react_version"] = "18+"
        elif pattern_name == "generic_components":
            context["typescript_feature"] = "generics"
        elif pattern_name == "custom_hooks":
            context["hook_type"] = "custom"

        return context

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
        """Analyze error handling patterns including async/await."""
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

        # Enhanced async error handling detection
        async_try_catch = list(
            re.finditer(
                r"try\s*\{[^}]*await[^}]*\}\s*catch\s*\([^)]*\)\s*\{",
                content,
                re.DOTALL,
            )
        )
        for match in async_try_catch:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.PERFORMANCE,
                    pattern_name="async_error_handling",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text="async try-catch with await",
                    confidence=0.9,
                    context={"error_handling": "async_try_catch"},
                )
            )

        # Check for Promise error handling
        promise_catch = list(re.finditer(r"\.catch\s*\(", content))
        for match in promise_catch:
            line_num = content[: match.start()].count("\n") + 1
            patterns.append(
                PatternMatch(
                    pattern_type=PatternType.PERFORMANCE,
                    pattern_name="promise_error_handling",
                    file_path="",
                    line_number=line_num,
                    column=match.start(),
                    matched_text=match.group(),
                    confidence=0.7,
                    context={"error_handling": "promise_catch"},
                )
            )

        # Check for React Error Boundaries (modern pattern)
        error_boundary_patterns = [
            r"static\s+getDerivedStateFromError",
            r"componentDidCatch",
            r"ErrorBoundary",
            r"useErrorBoundary",
            r"withErrorBoundary",
        ]

        for pattern in error_boundary_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.COMPONENT,
                        pattern_name="react_error_boundary",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.9,
                        context={"error_handling": "react_error_boundary"},
                    )
                )

        # Check for proper resource cleanup patterns
        cleanup_patterns = [
            r"return\s*\(\s*\)\s*=>\s*\{",  # useEffect cleanup
            r"controller\.abort\(\)",  # AbortController
            r"clearInterval|clearTimeout",  # Timer cleanup
            r"removeEventListener",  # Event cleanup
        ]

        for pattern in cleanup_patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.PERFORMANCE,
                        pattern_name="resource_cleanup",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"error_handling": "resource_cleanup"},
                    )
                )

        return patterns

    def _generate_react_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate comprehensive React-specific recommendations."""
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

            # Check for props interface - only if component has props
            # Detect if component has props parameter
            has_props_param = bool(
                re.search(r"function\s+\w+\s*\(\s*\{", content)
                or re.search(r"=\s*\(\s*\{", content)
            )

            if has_props_param:
                # Check for different TypeScript typing approaches
                has_interface = bool(re.search(r"interface\s+\w+Props", content))
                has_type_alias = bool(re.search(r"type\s+\w+Props\s*=", content))
                has_inline_types = bool(
                    re.search(r"\}\s*:\s*\{", content)
                )  # Inline: }: {

                # Only warn if truly missing types
                if not (has_interface or has_type_alias or has_inline_types):
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

            # Check for missing key props in list rendering
            map_without_key = list(
                re.finditer(
                    r"\.map\s*\([^)]*\)\s*=>.*<\w+(?!\s*key=)", content, re.DOTALL
                )
            )
            if map_without_key:
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="react_performance",
                        message="Add key prop to list items for better performance",
                        file_path=file_path,
                        line_number=content[: map_without_key[0].start()].count("\n")
                        + 1,
                        suggested_fix="Add key={item.id} or key={index} to list items",
                        rule_id="MISSING_KEY_PROP",
                    )
                )

            # Check for client-side features without 'use client' directive
            client_features = [
                r"useState\s*\(",
                r"useEffect\s*\(",
                r"onClick",
                r"onSubmit",
                r"window\.",
                r"document\.",
                r"localStorage",
                r"sessionStorage",
            ]

            has_client_features = any(
                re.search(pattern, content) for pattern in client_features
            )
            has_use_client = bool(re.search(r"['\"]use client['\"]", content))

            if (
                has_client_features
                and not has_use_client
                and "layout.tsx" not in file_path
            ):
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="nextjs_app_router",
                        message="Component uses client features but missing 'use client' directive",
                        file_path=file_path,
                        line_number=1,
                        suggested_fix="Add 'use client' at the top of the file",
                        rule_id="MISSING_USE_CLIENT",
                    )
                )

            # Check for server components that could benefit from async
            is_server_component = not has_use_client and not bool(
                re.search(r"['\"]use client['\"]", content)
            )
            has_data_fetching = bool(re.search(r"fetch\s*\(|await\s+", content))

            if (
                is_server_component
                and has_data_fetching
                and not bool(re.search(r"async\s+function", content))
            ):
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="react_server_components",
                        message="Consider making this a React Server Component with async data fetching",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Make component function async and fetch data directly",
                        rule_id="CONSIDER_SERVER_COMPONENT",
                    )
                )

            # Check for missing dependency arrays in hooks
            use_effect_without_deps = list(
                re.finditer(r"useEffect\s*\(\s*[^,)]+\s*\)", content)
            )
            if use_effect_without_deps:
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="react_hooks",
                        message="useEffect missing dependency array",
                        file_path=file_path,
                        line_number=content[: use_effect_without_deps[0].start()].count(
                            "\n"
                        )
                        + 1,
                        suggested_fix="Add dependency array as second parameter: [dependency1, dependency2]",
                        rule_id="MISSING_USEEFFECT_DEPS",
                    )
                )

            # Check for React 18 opportunities
            if not bool(
                re.search(
                    r"Suspense|startTransition|useDeferredValue|useTransition", content
                )
            ):
                has_expensive_operations = bool(
                    re.search(r"\.filter\(.*\.map\(|\.sort\(|for\s*\(.*length", content)
                )
                if has_expensive_operations:
                    recommendations.append(
                        Recommendation(
                            severity=Severity.INFO,
                            category="react_concurrent",
                            message="Consider using React 18 concurrent features for better performance",
                            file_path=file_path,
                            line_number=None,
                            suggested_fix="Use useDeferredValue or startTransition for expensive operations",
                            rule_id="CONSIDER_CONCURRENT_FEATURES",
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
        """Generate comprehensive error handling recommendations."""
        recommendations = []

        # Enhanced async error handling detection
        awaits_in_file = list(re.finditer(r"await\s+", content))
        try_blocks = list(re.finditer(r"try\s*\{", content))

        # Check for async operations without proper error handling
        if awaits_in_file and len(try_blocks) == 0:
            recommendations.append(
                Recommendation(
                    severity=Severity.ERROR,
                    category="async_error_handling",
                    message="Async operations should include proper error handling",
                    file_path=file_path,
                    line_number=awaits_in_file[0].start() // len(content.split("\n")[0])
                    + 1,
                    suggested_fix="Wrap async operations in try-catch blocks",
                    rule_id="MISSING_ASYNC_ERROR_HANDLING",
                )
            )

        # Check for promises without catch handlers
        promise_chains = list(re.finditer(r"\.then\s*\([^)]*\)", content))
        promise_catches = list(re.finditer(r"\.catch\s*\([^)]*\)", content))

        if promise_chains and len(promise_catches) < len(promise_chains):
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="promise_error_handling",
                    message="Promise chains should include .catch() error handlers",
                    file_path=file_path,
                    line_number=promise_chains[0].start() // len(content.split("\n")[0])
                    + 1,
                    suggested_fix="Add .catch(error => { console.error(error); }) to promise chains",
                    rule_id="MISSING_PROMISE_CATCH",
                )
            )

        # Check for useEffect without cleanup when needed
        use_effects = list(re.finditer(r"useEffect\s*\(", content))
        cleanup_returns = list(re.finditer(r"return\s*\(\s*\)\s*=>", content))

        # Patterns that likely need cleanup
        needs_cleanup_patterns = [
            r"setInterval|setTimeout",
            r"addEventListener",
            r"subscribe|subscribe\(",
            r"new\s+WebSocket",
            r"new\s+EventSource",
        ]

        has_cleanup_needs = any(
            re.search(pattern, content) for pattern in needs_cleanup_patterns
        )

        if use_effects and has_cleanup_needs and len(cleanup_returns) == 0:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="resource_cleanup",
                    message="useEffect with side effects should include cleanup function",
                    file_path=file_path,
                    line_number=use_effects[0].start() // len(content.split("\n")[0])
                    + 1,
                    suggested_fix="Return cleanup function from useEffect: () => { /* cleanup code */ }",
                    rule_id="MISSING_USEEFFECT_CLEANUP",
                )
            )

        # Check for missing error boundaries in component hierarchies
        if self._is_component_file(file_path):
            has_error_boundary = bool(
                re.search(
                    r"ErrorBoundary|componentDidCatch|getDerivedStateFromError", content
                )
            )
            has_async_components = bool(
                re.search(r"Suspense|lazy\(|React\.lazy", content)
            )

            if has_async_components and not has_error_boundary:
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="react_error_handling",
                        message="Components with async loading should be wrapped in Error Boundaries",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Wrap components in <ErrorBoundary> or implement error boundary",
                        rule_id="MISSING_ERROR_BOUNDARY",
                    )
                )

        # Check for fetch without error handling
        fetch_calls = list(re.finditer(r"fetch\s*\(", content))
        response_ok_checks = list(re.finditer(r"\.ok\b|response\.status", content))

        if fetch_calls and len(response_ok_checks) == 0 and len(try_blocks) == 0:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="api_error_handling",
                    message="Fetch requests should include response status checking",
                    file_path=file_path,
                    line_number=fetch_calls[0].start() // len(content.split("\n")[0])
                    + 1,
                    suggested_fix="Check response.ok or response.status before processing",
                    rule_id="MISSING_FETCH_ERROR_CHECK",
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
        """Calculate overall quality score with enhanced modern codebase scoring."""
        # Improved base score for modern codebases
        base_score = 0.7

        # Enhanced bonus for good patterns with weighted scoring
        pattern_bonuses = {
            PatternType.COMPONENT: 0.08,  # Increased for React components
            PatternType.PERFORMANCE: 0.12,  # Higher weight for performance
            PatternType.SECURITY: 0.15,  # Critical importance
            PatternType.ACCESSIBILITY: 0.10,  # Increased for a11y compliance
            PatternType.ARCHITECTURAL: 0.06,  # Architecture patterns
            PatternType.TYPE_SAFETY: 0.08,  # TypeScript patterns
        }

        # Calculate pattern bonuses with confidence weighting
        pattern_score = 0.0
        for pattern in patterns_found:
            bonus = pattern_bonuses.get(pattern.pattern_type, 0.03)
            pattern_score += bonus * pattern.confidence

        # Quality indicator bonuses for modern best practices
        quality_bonuses = self._calculate_quality_bonuses(content)

        # Penalty for issues with adjusted severity weighting
        severity_penalties = {
            Severity.INFO: 0.01,  # Reduced penalty for info
            Severity.WARNING: 0.05,  # Reduced for warnings
            Severity.ERROR: 0.12,  # Reduced but still significant
            Severity.CRITICAL: 0.20,  # High penalty for critical issues
        }

        penalty_score = 0.0
        for rec in recommendations:
            penalty_score += severity_penalties.get(rec.severity, 0.01)

        # Additional scoring factors
        complexity_penalty = self._calculate_complexity_penalty(content)

        # Final score calculation
        final_score = (
            base_score
            + pattern_score
            + quality_bonuses
            - penalty_score
            - complexity_penalty
        )

        return max(0.0, min(1.0, final_score))

    def _calculate_quality_bonuses(self, content: str) -> float:
        """Calculate bonuses for modern quality indicators."""
        bonuses = 0.0

        # TypeScript strict mode indicators
        if "strict" in content or "@typescript-eslint/strict" in content:
            bonuses += 0.1

        # Modern React patterns
        if "use client" in content:
            bonuses += 0.05
        if "use server" in content:
            bonuses += 0.05
        if re.search(r"Suspense|ErrorBoundary|lazy\(", content):
            bonuses += 0.08

        # Test coverage indicators
        if re.search(r"describe\(|test\(|it\(|expect\(", content):
            bonuses += 0.1

        # Security best practices
        if re.search(r"helmet|cors|rate-?limit", content):
            bonuses += 0.08

        # Performance optimizations
        if re.search(r"React\.memo|useCallback|useMemo|dynamic\(", content):
            bonuses += 0.06

        # Accessibility compliance
        aria_count = len(re.findall(r"aria-\w+", content))
        if aria_count > 0:
            bonuses += min(0.1, aria_count * 0.02)

        return bonuses

    def _calculate_complexity_penalty(self, content: str) -> float:
        """Calculate penalties based on code complexity."""
        penalty = 0.0

        lines = len(content.split("\n"))

        # Graduated penalty for file length
        if lines > 1000:
            penalty += 0.08
        elif lines > 500:
            penalty += 0.04
        elif lines > 300:
            penalty += 0.02

        # Complexity indicators
        nesting_level = self._estimate_nesting_level(content)
        if nesting_level > 5:
            penalty += 0.06
        elif nesting_level > 3:
            penalty += 0.03

        # Function length penalty
        function_lengths = self._estimate_function_lengths(content)
        avg_function_length = (
            sum(function_lengths) / len(function_lengths) if function_lengths else 0
        )
        if avg_function_length > 50:
            penalty += 0.05
        elif avg_function_length > 30:
            penalty += 0.02

        return penalty

    def _estimate_nesting_level(self, content: str) -> int:
        """Estimate the maximum nesting level in the code."""
        max_nesting = 0
        current_nesting = 0

        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            # Count opening braces
            current_nesting += stripped.count("{")
            current_nesting -= stripped.count("}")

            max_nesting = max(max_nesting, current_nesting)

        return max_nesting

    def _estimate_function_lengths(self, content: str) -> list[int]:
        """Estimate function lengths in the code."""
        function_lengths = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Look for function declarations
            if re.search(
                r"(function\s+\w+|const\s+\w+\s*=\s*.*=>|\w+\s*\(.*\)\s*\{)", line
            ):
                function_start = i
                brace_count = line.count("{") - line.count("}")
                i += 1

                # Find function end
                while i < len(lines) and brace_count > 0:
                    current_line = lines[i]
                    brace_count += current_line.count("{") - current_line.count("}")
                    i += 1

                if brace_count == 0:
                    function_lengths.append(i - function_start)
            else:
                i += 1

        return function_lengths
