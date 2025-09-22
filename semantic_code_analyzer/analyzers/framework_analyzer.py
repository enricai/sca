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

"""Framework-specific analyzer for Next.js, React, and related frameworks.

This analyzer focuses on framework-specific patterns, conventions,
and best practices that go beyond basic architectural concerns.
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


class FrameworkAnalyzer(BaseAnalyzer):
    """Analyzes framework-specific patterns and conventions.

    Focuses on:
    - Next.js specific patterns (app router, API routes, etc.)
    - React patterns (hooks, context, etc.)
    - Framework integrations (next-intl, etc.)
    - Build and deployment patterns
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the FrameworkAnalyzer with configuration.

        Args:
            config: Optional configuration dictionary for the analyzer.
        """
        super().__init__(config)
        self.nextjs_patterns = self._init_nextjs_patterns()
        self.react_patterns = self._init_react_patterns()
        self.integration_patterns = self._init_integration_patterns()

    def get_analyzer_name(self) -> str:
        """Return the name identifier for this analyzer.

        Returns:
            The string identifier 'framework'.
        """
        return "framework"

    def get_weight(self) -> float:
        """Return the weight for this analyzer in overall scoring.

        Returns:
            The weight value (0.25 for 25% of overall score).
        """
        return 0.25  # 25% weight in overall scoring

    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze a file for framework-specific patterns.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            AnalysisResult with framework analysis
        """
        start_time = time.time()
        patterns_found = []
        recommendations = []

        # Analyze different framework aspects
        patterns_found.extend(self._analyze_nextjs_patterns(file_path, content))
        patterns_found.extend(self._analyze_react_patterns(content))
        patterns_found.extend(self._analyze_integration_patterns(content))
        patterns_found.extend(self._analyze_api_patterns(file_path, content))

        # Generate recommendations
        recommendations.extend(
            self._generate_nextjs_recommendations(file_path, content)
        )
        recommendations.extend(self._generate_react_recommendations(file_path, content))
        recommendations.extend(
            self._generate_integration_recommendations(file_path, content)
        )

        score = self._calculate_framework_score(
            patterns_found, recommendations, file_path
        )

        metrics = {
            "nextjs_patterns_count": len(
                [p for p in patterns_found if "nextjs" in p.pattern_name]
            ),
            "react_patterns_count": len(
                [p for p in patterns_found if "react" in p.pattern_name]
            ),
            "integration_patterns_count": len(
                [p for p in patterns_found if "integration" in p.pattern_name]
            ),
            "api_route": self._is_api_route(file_path),
            "page_component": self._is_page_component(file_path),
            "layout_component": self._is_layout_component(file_path),
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
        """Initialize Next.js specific patterns."""
        return {
            "app_router_conventions": {
                "patterns": {
                    "page": r"export\s+default\s+function\s+\w+.*\{\s*$",
                    "layout": r"export\s+default\s+function\s+\w+Layout",
                    "loading": r"export\s+default\s+function\s+Loading",
                    "error": r"export\s+default\s+function\s+Error",
                    "not_found": r"export\s+default\s+function\s+NotFound",
                },
                "score_bonus": 0.1,
            },
            "metadata_patterns": {
                "patterns": {
                    "static_metadata": r"export\s+const\s+metadata\s*[:=]",
                    "dynamic_metadata": r"export\s+async\s+function\s+generateMetadata",
                    "viewport": r"export\s+const\s+viewport\s*[:=]",
                },
                "score_bonus": 0.15,
            },
            "server_actions": {
                "patterns": {
                    "server_action": r"['\"]use server['\"]",
                    "form_action": r"action=\{.*\}",
                },
                "score_bonus": 0.1,
            },
            "streaming_patterns": {
                "patterns": {
                    "suspense": r"<Suspense",
                    "streaming_response": r"ReadableStream",
                },
                "score_bonus": 0.05,
            },
        }

    def _init_react_patterns(self) -> dict[str, Any]:
        """Initialize React specific patterns."""
        return {
            "modern_react": {
                "patterns": {
                    "hooks": r"use[A-Z]\w*\s*\(",
                    "context": r"createContext\s*\(",
                    "provider": r"\.Provider",
                    "forward_ref": r"forwardRef\s*\(",
                },
                "score_bonus": 0.05,
            },
            "error_boundaries": {
                "patterns": {
                    "error_boundary": r"componentDidCatch|getDerivedStateFromError",
                    "error_fallback": r"fallback=\{",
                },
                "score_bonus": 0.1,
            },
            "performance_patterns": {
                "patterns": {
                    "memo": r"React\.memo\s*\(",
                    "callback": r"useCallback\s*\(",
                    "memo_hook": r"useMemo\s*\(",
                    "lazy": r"React\.lazy\s*\(",
                },
                "score_bonus": 0.08,
            },
        }

    def _init_integration_patterns(self) -> dict[str, Any]:
        """Initialize framework integration patterns."""
        return {
            "internationalization": {
                "patterns": {
                    "next_intl": r"useTranslations\s*\(",
                    "locale_handling": r"params:\s*\{\s*locale",
                    "translation_keys": r"t\s*\(['\"]",
                },
                "score_bonus": 0.1,
            },
            "authentication": {
                "patterns": {
                    "auth_context": r"AuthContext|useAuth",
                    "jwt_handling": r"jsonwebtoken|jwt",
                    "session_management": r"useSession|getSession",
                },
                "score_bonus": 0.1,
            },
            "database_patterns": {
                "patterns": {
                    "orm_usage": r"prisma\.|drizzle\.|sequelize\.",
                    "query_patterns": r"findMany|findUnique|create|update|delete",
                },
                "score_bonus": 0.05,
            },
        }

    def _analyze_nextjs_patterns(
        self, file_path: str, content: str
    ) -> list[PatternMatch]:
        """Analyze Next.js specific patterns."""
        patterns = []

        # Check app router conventions
        if self._is_app_router_file(file_path):
            for pattern_name, pattern_regex in self.nextjs_patterns[
                "app_router_conventions"
            ]["patterns"].items():
                matches = list(re.finditer(pattern_regex, content, re.MULTILINE))
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
                            context={
                                "framework": "nextjs",
                                "pattern_type": pattern_name,
                            },
                        )
                    )

        # Check metadata patterns
        for pattern_name, pattern_regex in self.nextjs_patterns["metadata_patterns"][
            "patterns"
        ].items():
            matches = list(re.finditer(pattern_regex, content))
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
                        context={"framework": "nextjs", "pattern_type": "metadata"},
                    )
                )

        # Check server actions
        for pattern_name, pattern_regex in self.nextjs_patterns["server_actions"][
            "patterns"
        ].items():
            matches = list(re.finditer(pattern_regex, content))
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
                        confidence=0.8,
                        context={
                            "framework": "nextjs",
                            "pattern_type": "server_action",
                        },
                    )
                )

        return patterns

    def _analyze_react_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze React specific patterns."""
        patterns = []

        # Check modern React patterns
        for pattern_name, pattern_regex in self.react_patterns["modern_react"][
            "patterns"
        ].items():
            matches = list(re.finditer(pattern_regex, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.FRAMEWORK,
                        pattern_name=f"react_{pattern_name}",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"framework": "react", "pattern_type": pattern_name},
                    )
                )

        # Check performance patterns
        for pattern_name, pattern_regex in self.react_patterns["performance_patterns"][
            "patterns"
        ].items():
            matches = list(re.finditer(pattern_regex, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.PERFORMANCE,
                        pattern_name=f"react_{pattern_name}",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.9,
                        context={"framework": "react", "optimization": pattern_name},
                    )
                )

        return patterns

    def _analyze_integration_patterns(self, content: str) -> list[PatternMatch]:
        """Analyze framework integration patterns."""
        patterns = []

        # Check internationalization patterns
        for pattern_name, pattern_regex in self.integration_patterns[
            "internationalization"
        ]["patterns"].items():
            matches = list(re.finditer(pattern_regex, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.FRAMEWORK,
                        pattern_name=f"integration_{pattern_name}",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"integration": "i18n", "pattern": pattern_name},
                    )
                )

        # Check authentication patterns
        for pattern_name, pattern_regex in self.integration_patterns["authentication"][
            "patterns"
        ].items():
            matches = list(re.finditer(pattern_regex, content))
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                patterns.append(
                    PatternMatch(
                        pattern_type=PatternType.SECURITY,
                        pattern_name=f"integration_{pattern_name}",
                        file_path="",
                        line_number=line_num,
                        column=match.start(),
                        matched_text=match.group(),
                        confidence=0.8,
                        context={"integration": "auth", "pattern": pattern_name},
                    )
                )

        return patterns

    def _analyze_api_patterns(self, file_path: str, content: str) -> list[PatternMatch]:
        """Analyze API route patterns."""
        patterns = []

        if self._is_api_route(file_path):
            # Check for proper HTTP method exports
            http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
            for method in http_methods:
                method_pattern = rf"export\s+async\s+function\s+{method}"
                matches = list(re.finditer(method_pattern, content))
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    patterns.append(
                        PatternMatch(
                            pattern_type=PatternType.FRAMEWORK,
                            pattern_name="api_method_export",
                            file_path=file_path,
                            line_number=line_num,
                            column=match.start(),
                            matched_text=match.group(),
                            confidence=0.9,
                            context={"api": "route_handler", "method": method},
                        )
                    )

            # Check for NextRequest/NextResponse usage
            next_types = ["NextRequest", "NextResponse"]
            for next_type in next_types:
                type_pattern = rf"\b{next_type}\b"
                matches = list(re.finditer(type_pattern, content))
                for match in matches:
                    line_num = content[: match.start()].count("\n") + 1
                    patterns.append(
                        PatternMatch(
                            pattern_type=PatternType.FRAMEWORK,
                            pattern_name="nextjs_api_types",
                            file_path=file_path,
                            line_number=line_num,
                            column=match.start(),
                            matched_text=match.group(),
                            confidence=0.8,
                            context={"api": "type_usage", "type": next_type},
                        )
                    )

        return patterns

    def _generate_nextjs_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate Next.js specific recommendations."""
        recommendations = []

        # Check for missing metadata in layout files
        if self._is_layout_component(file_path):
            if not re.search(r"export\s+(const\s+)?metadata", content):
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="nextjs_seo",
                        message="Layout components should export metadata for SEO",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="export const metadata: Metadata = { title: '...', description: '...' };",
                        rule_id="MISSING_LAYOUT_METADATA",
                    )
                )

        # Check for missing error handling in API routes
        if self._is_api_route(file_path):
            if "try" not in content or "catch" not in content:
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="nextjs_api",
                        message="API routes should include proper error handling",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Wrap API logic in try-catch blocks",
                        rule_id="MISSING_API_ERROR_HANDLING",
                    )
                )

            # Check for proper status codes
            if "NextResponse" not in content:
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="nextjs_api",
                        message="Use NextResponse for proper HTTP responses in API routes",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="return NextResponse.json(data, { status: 200 })",
                        rule_id="USE_NEXT_RESPONSE",
                    )
                )

        # Check for client/server component clarity
        if file_path.endswith(".tsx") and "components" in file_path:
            has_client_features = any(
                feature in content for feature in ["onClick", "useState", "useEffect"]
            )
            has_use_client = '"use client"' in content or "'use client'" in content

            if has_client_features and not has_use_client:
                recommendations.append(
                    Recommendation(
                        severity=Severity.WARNING,
                        category="nextjs_components",
                        message="Component uses client features but missing 'use client' directive",
                        file_path=file_path,
                        line_number=1,
                        suggested_fix="Add 'use client' at the top of the file",
                        rule_id="MISSING_USE_CLIENT",
                    )
                )

        return recommendations

    def _generate_react_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate React specific recommendations."""
        recommendations = []

        # Check for missing key props in lists
        if ".map(" in content and "key=" not in content:
            recommendations.append(
                Recommendation(
                    severity=Severity.WARNING,
                    category="react_performance",
                    message="Add key prop to list items for better performance",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="Add key={item.id} or key={index} to list items",
                    rule_id="MISSING_KEY_PROP",
                )
            )

        # Check for missing dependencies in useEffect
        use_effect_matches = list(
            re.finditer(
                r"useEffect\s*\(\s*\(\s*\)\s*=>\s*\{([^}]*)\},\s*\[([^\]]*)\]",
                content,
                re.DOTALL,
            )
        )
        for match in use_effect_matches:
            effect_body = match.group(1)
            deps_array = match.group(2)

            # Simple check for variables used in effect but not in deps
            variables_in_effect = re.findall(r"\b[a-zA-Z_]\w*\b", effect_body)
            deps_list = [dep.strip() for dep in deps_array.split(",") if dep.strip()]

            if (
                len(variables_in_effect) > len(deps_list) + 2
            ):  # +2 for common exceptions
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="react_hooks",
                        message="Review useEffect dependencies to prevent stale closures",
                        file_path=file_path,
                        line_number=content[: match.start()].count("\n") + 1,
                        suggested_fix="Add all used variables to dependency array",
                        rule_id="REVIEW_EFFECT_DEPENDENCIES",
                    )
                )

        return recommendations

    def _generate_integration_recommendations(
        self, file_path: str, content: str
    ) -> list[Recommendation]:
        """Generate framework integration recommendations."""
        recommendations = []

        # Check for hardcoded strings that should be internationalized
        if "useTranslations" in content:
            hardcoded_text = re.findall(r">[^<{]*[a-zA-Z]{3,}[^<}]*<", content)
            if len(hardcoded_text) > 2:  # More than 2 potential hardcoded strings
                recommendations.append(
                    Recommendation(
                        severity=Severity.INFO,
                        category="internationalization",
                        message="Consider internationalizing hardcoded text",
                        file_path=file_path,
                        line_number=None,
                        suggested_fix="Replace text with t('key') from useTranslations",
                        rule_id="INTERNATIONALIZE_TEXT",
                    )
                )

        # Check for missing error boundaries around async components
        if (
            "async" in content
            and "Suspense" not in content
            and file_path.endswith(".tsx")
        ):
            recommendations.append(
                Recommendation(
                    severity=Severity.INFO,
                    category="react_async",
                    message="Consider wrapping async components with Suspense boundaries",
                    file_path=file_path,
                    line_number=None,
                    suggested_fix="<Suspense fallback={<Loading />}><AsyncComponent /></Suspense>",
                    rule_id="MISSING_SUSPENSE_BOUNDARY",
                )
            )

        return recommendations

    def _is_app_router_file(self, file_path: str) -> bool:
        """Check if file is part of Next.js app router."""
        return "src/app/" in file_path

    def _is_api_route(self, file_path: str) -> bool:
        """Check if file is an API route."""
        return file_path.endswith("/route.ts") or file_path.endswith("/route.js")

    def _is_page_component(self, file_path: str) -> bool:
        """Check if file is a page component."""
        return file_path.endswith("/page.tsx") or file_path.endswith("/page.jsx")

    def _is_layout_component(self, file_path: str) -> bool:
        """Check if file is a layout component."""
        return file_path.endswith("/layout.tsx") or file_path.endswith("/layout.jsx")

    def _calculate_framework_score(
        self,
        patterns_found: list[PatternMatch],
        recommendations: list[Recommendation],
        file_path: str,
    ) -> float:
        """Calculate framework adherence score."""
        base_score = 0.6

        # File-type specific scoring
        if self._is_api_route(file_path):
            base_score = 0.7  # Higher expectations for API routes
        elif self._is_page_component(file_path) or self._is_layout_component(file_path):
            base_score = 0.65  # Higher expectations for pages/layouts

        # Add points for good patterns
        pattern_bonuses = {
            "nextjs_page": 0.1,
            "nextjs_layout": 0.1,
            "nextjs_static_metadata": 0.15,
            "nextjs_dynamic_metadata": 0.2,
            "nextjs_server_action": 0.1,
            "react_hooks": 0.05,
            "react_memo": 0.08,
            "react_callback": 0.08,
            "integration_next_intl": 0.1,
            "integration_auth_context": 0.1,
            "api_method_export": 0.15,
            "nextjs_api_types": 0.1,
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

        return max(0.0, min(1.0, base_score))
