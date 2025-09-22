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

"""Enhanced Multi-Dimensional Code Analyzer.

This package provides comprehensive code quality analysis through multi-dimensional
pattern recognition, focusing on:

- Architectural patterns and Next.js conventions
- Code quality, security, and best practices
- TypeScript usage and type safety patterns
- Framework-specific implementations and integrations

Main Components:
- MultiDimensionalScorer: Main analysis orchestrator
- Specialized Analyzers: Pattern-specific analysis modules
- WeightedAggregator: Mathematical score aggregation

Example Usage:
    from semantic_code_analyzer import MultiDimensionalScorer, EnhancedScorerConfig

    config = EnhancedScorerConfig(
        architectural_weight=0.30,
        quality_weight=0.30,
        typescript_weight=0.25,
        framework_weight=0.15
    )

    scorer = MultiDimensionalScorer(config, repo_path=".")
    results = scorer.analyze_commit("commit_hash")
    # Use logging instead of print for library code
"""

from __future__ import annotations

from .analyzers import (
    AnalysisResult,
    ArchitecturalAnalyzer,
    ArchitecturalDomain,
    BaseAnalyzer,
    DomainAwareAdherenceAnalyzer,
    DomainClassifier,
    FrameworkAnalyzer,
    PatternMatch,
    QualityAnalyzer,
    Recommendation,
    TypeScriptAnalyzer,
)
from .scorers import EnhancedScorerConfig, MultiDimensionalScorer, WeightedAggregator

__version__ = "0.3.0"
__author__ = "SCA Team"

__all__ = [
    "MultiDimensionalScorer",
    "EnhancedScorerConfig",
    "WeightedAggregator",
    "ArchitecturalAnalyzer",
    "QualityAnalyzer",
    "TypeScriptAnalyzer",
    "FrameworkAnalyzer",
    "DomainAwareAdherenceAnalyzer",
    "DomainClassifier",
    "ArchitecturalDomain",
    "BaseAnalyzer",
    "AnalysisResult",
    "PatternMatch",
    "Recommendation",
]
