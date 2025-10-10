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

# MIT License
#
# Copyright (c) 2024 Semantic Code Analyzer Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, publish, distribute, sublicense, and/or sell
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

"""Code feature extraction utilities for embedding divergence analysis.

This module provides functions to extract measurable features from source code
and compare them against reference patterns to generate actionable insights.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def extract_code_features(code: str, file_path: str = "") -> dict[str, Any]:
    """Extract measurable features from source code.

    Args:
        code: Source code content
        file_path: Optional file path for context

    Returns:
        Dictionary with extracted features including imports, functions,
        complexity metrics, and structural patterns
    """
    features: dict[str, Any] = {}

    # File metadata
    features["file_path"] = file_path
    features["file_extension"] = Path(file_path).suffix if file_path else ""

    # Basic metrics
    lines = code.split("\n")
    features["line_count"] = len(lines)
    features["char_count"] = len(code)
    features["avg_line_length"] = (
        np.mean([len(line) for line in lines]) if lines else 0.0
    )

    # Import analysis
    import_patterns = [
        r"^import\s+(\S+)",
        r"^from\s+(\S+)\s+import",
        r'^import\s*\{([^}]+)\}\s*from\s*["\']([^"\']+)["\']',  # JS/TS
    ]
    imports = []
    for pattern in import_patterns:
        imports.extend(re.findall(pattern, code, re.MULTILINE))

    # Flatten and clean imports
    clean_imports = []
    for imp in imports:
        if isinstance(imp, tuple):
            clean_imports.extend([i.strip() for i in imp if i.strip()])
        else:
            clean_imports.append(imp.strip())

    features["imports"] = clean_imports
    features["import_count"] = len(clean_imports)

    # Function and class definitions
    features["function_count"] = len(
        re.findall(r"^\s*(?:def|function|const\s+\w+\s*=\s*\()", code, re.MULTILINE)
    )
    features["class_count"] = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))

    # TypeScript/JavaScript specific
    if features["file_extension"] in [".ts", ".tsx", ".js", ".jsx"]:
        features["has_interface"] = bool(re.search(r"\binterface\s+\w+", code))
        features["has_type_annotation"] = bool(re.search(r":\s*\w+\s*[=;,)]", code))
        features["has_jsx"] = bool(re.search(r"<\w+.*?>", code))
        features["arrow_function_count"] = len(re.findall(r"=>\s*{", code))
        features["has_react_import"] = any(
            "react" in imp.lower() for imp in clean_imports
        )
        features["hook_count"] = len(re.findall(r"\buse[A-Z]\w+\(", code))

    # Python specific
    elif features["file_extension"] == ".py":
        features["has_type_hints"] = bool(re.search(r":\s*\w+\s*=", code))
        features["has_docstring"] = bool(re.search(r'""".*?"""', code, re.DOTALL))
        features["decorator_count"] = len(re.findall(r"^\s*@\w+", code, re.MULTILINE))

    # Code complexity indicators
    features["complexity_score"] = (
        code.count("if ")
        + code.count("for ")
        + code.count("while ")
        + code.count("try ")
        + code.count("except ")
        + code.count("switch ")
        + code.count("case ")
    )

    # Comment and documentation
    features["comment_count"] = len(re.findall(r"//.*$|#.*$", code, re.MULTILINE))
    features["comment_density"] = (
        features["comment_count"] / features["line_count"]
        if features["line_count"] > 0
        else 0.0
    )

    # Structural patterns
    features["empty_line_count"] = sum(1 for line in lines if not line.strip())
    features["max_line_length"] = max((len(line) for line in lines), default=0)

    # Common keywords (for vocabulary analysis)
    keywords = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)
    features["unique_keywords"] = len(set(keywords))
    features["total_keywords"] = len(keywords)
    features["keyword_diversity"] = (
        features["unique_keywords"] / features["total_keywords"]
        if features["total_keywords"] > 0
        else 0.0
    )

    return features


def compare_features(
    query_features: dict[str, Any], reference_features_list: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare query features against reference patterns.

    Args:
        query_features: Features extracted from the file being analyzed
        reference_features_list: List of features from reference patterns

    Returns:
        Dictionary with comparison metrics showing differences and gaps
    """
    if not reference_features_list:
        return {}

    comparison: dict[str, Any] = {}

    # Numeric feature comparison
    numeric_features = [
        "line_count",
        "char_count",
        "avg_line_length",
        "import_count",
        "function_count",
        "class_count",
        "complexity_score",
        "comment_count",
        "comment_density",
        "unique_keywords",
    ]

    for feature_name in numeric_features:
        query_value = query_features.get(feature_name, 0)

        # Get values from references
        ref_values = [
            ref.get(feature_name, 0)
            for ref in reference_features_list
            if feature_name in ref
        ]

        if ref_values:
            ref_mean = float(np.mean(ref_values))
            ref_std = float(np.std(ref_values))

            # Calculate difference percentage
            if ref_mean > 0:
                diff_pct = ((query_value - ref_mean) / ref_mean) * 100
            else:
                diff_pct = 0.0

            comparison[feature_name] = {
                "query": query_value,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "diff_pct": diff_pct,
                "significance": _assess_significance(abs(diff_pct)),
            }

    # Import analysis
    query_imports = set(query_features.get("imports", []))
    ref_import_sets = [set(ref.get("imports", [])) for ref in reference_features_list]

    # Find common imports across references
    if ref_import_sets:
        common_ref_imports = (
            set.intersection(*ref_import_sets)
            if len(ref_import_sets) > 1
            else ref_import_sets[0]
        )
        missing_imports = list(common_ref_imports - query_imports)

        # Calculate import overlap
        all_ref_imports = set.union(*ref_import_sets) if ref_import_sets else set()
        if all_ref_imports:
            import_overlap = len(query_imports & all_ref_imports) / len(all_ref_imports)
        else:
            import_overlap = 0.0

        comparison["imports"] = {
            "missing_common": missing_imports[:10],  # Top 10
            "overlap_ratio": import_overlap,
        }

    # Boolean feature comparison (TypeScript/React specific)
    boolean_features = [
        "has_interface",
        "has_type_annotation",
        "has_jsx",
        "has_react_import",
    ]

    for feature_name in boolean_features:
        if feature_name in query_features:
            query_value = query_features.get(feature_name, False)
            ref_true_count = sum(
                1 for ref in reference_features_list if ref.get(feature_name, False)
            )
            ref_true_pct = (
                ref_true_count / len(reference_features_list) * 100
                if reference_features_list
                else 0.0
            )

            if ref_true_pct > 70 and not query_value:
                comparison[feature_name] = {
                    "query": query_value,
                    "ref_prevalence_pct": ref_true_pct,
                    "missing": True,
                }

    return comparison


def generate_feature_insights(
    comparison: dict[str, Any],
    query_code: str,
    reference_patterns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate human-readable insights from feature comparison.

    Args:
        comparison: Output from compare_features()
        query_code: Source code of the file being analyzed
        reference_patterns: List of reference pattern data

    Returns:
        List of actionable insights with severity, message, and suggestions
    """
    insights: list[dict[str, Any]] = []

    # Check line count differences
    if "line_count" in comparison:
        line_diff = comparison["line_count"]
        if line_diff["diff_pct"] < -50:  # 50% shorter
            insights.append(
                {
                    "category": "structure",
                    "severity": "high",
                    "message": f"File is significantly shorter ({line_diff['query']} lines) than similar files (avg: {line_diff['ref_mean']:.0f} lines)",
                    "suggestion": "Consider if this file is incomplete or missing essential patterns like error handling, loading states, or proper type definitions",
                    "estimated_improvement": 0.15,
                }
            )
        elif line_diff["diff_pct"] > 100:  # 100% longer
            insights.append(
                {
                    "category": "structure",
                    "severity": "medium",
                    "message": f"File is much longer ({line_diff['query']} lines) than similar files (avg: {line_diff['ref_mean']:.0f} lines)",
                    "suggestion": "Consider splitting into smaller, more focused modules",
                    "estimated_improvement": 0.08,
                }
            )

    # Check import deficiencies
    if "imports" in comparison:
        import_data = comparison["imports"]
        missing_imports = import_data.get("missing_common", [])

        if missing_imports and len(missing_imports) > 0:
            insights.append(
                {
                    "category": "imports",
                    "severity": "high",
                    "message": f"Missing common imports found in {len(reference_patterns)} similar files",
                    "suggestion": f"Add imports: {', '.join(missing_imports[:5])}",
                    "estimated_improvement": 0.12,
                    "details": {"missing_imports": missing_imports[:5]},
                }
            )

        if import_data.get("overlap_ratio", 1.0) < 0.3:
            insights.append(
                {
                    "category": "imports",
                    "severity": "medium",
                    "message": f"Low import overlap ({import_data['overlap_ratio']:.1%}) with established patterns",
                    "suggestion": "Review import structure and align with codebase conventions",
                    "estimated_improvement": 0.10,
                }
            )

    # Check function count
    if "function_count" in comparison:
        func_diff = comparison["function_count"]
        if func_diff["diff_pct"] < -60:  # 60% fewer functions
            insights.append(
                {
                    "category": "structure",
                    "severity": "high",
                    "message": f"Fewer functions ({func_diff['query']}) than similar files (avg: {func_diff['ref_mean']:.1f})",
                    "suggestion": "Consider breaking down logic into smaller, reusable functions or adding helper methods",
                    "estimated_improvement": 0.10,
                }
            )

    # Check TypeScript/interface usage
    if "has_interface" in comparison:
        interface_data = comparison["has_interface"]
        if interface_data.get("missing"):
            insights.append(
                {
                    "category": "typescript",
                    "severity": "medium",
                    "message": f"Missing TypeScript interfaces (present in {interface_data['ref_prevalence_pct']:.0f}% of similar files)",
                    "suggestion": "Define proper TypeScript interfaces for props and data structures",
                    "estimated_improvement": 0.08,
                }
            )

    # Check type annotations
    if "has_type_annotation" in comparison:
        type_data = comparison["has_type_annotation"]
        if type_data.get("missing"):
            insights.append(
                {
                    "category": "typescript",
                    "severity": "medium",
                    "message": f"Missing type annotations (present in {type_data['ref_prevalence_pct']:.0f}% of similar files)",
                    "suggestion": "Add type annotations to function parameters and return values",
                    "estimated_improvement": 0.07,
                }
            )

    # Check React-specific patterns
    if "has_react_import" in comparison:
        react_data = comparison["has_react_import"]
        if react_data.get("missing"):
            insights.append(
                {
                    "category": "imports",
                    "severity": "high",
                    "message": "Missing React import (required for component files)",
                    "suggestion": "Add: import React from 'react'",
                    "estimated_improvement": 0.10,
                }
            )

    # Check comment density
    if "comment_density" in comparison:
        comment_diff = comparison["comment_density"]
        if (
            comment_diff["diff_pct"] < -80 and comment_diff["ref_mean"] > 0.05
        ):  # Much lower than average
            insights.append(
                {
                    "category": "documentation",
                    "severity": "low",
                    "message": f"Low comment density ({comment_diff['query']:.1%}) compared to similar files ({comment_diff['ref_mean']:.1%})",
                    "suggestion": "Add comments to explain complex logic or document public APIs",
                    "estimated_improvement": 0.05,
                }
            )

    # Sort by estimated improvement (descending)
    insights.sort(key=lambda x: x.get("estimated_improvement", 0), reverse=True)

    return insights


def _assess_significance(diff_pct: float) -> str:
    """Assess the significance of a percentage difference.

    Args:
        diff_pct: Absolute percentage difference

    Returns:
        Significance level: "high", "medium", or "low"
    """
    if diff_pct > 50:
        return "high"
    elif diff_pct > 20:
        return "medium"
    else:
        return "low"
