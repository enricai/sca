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

"""Multi-dimensional code analysis modules.

This package provides specialized analyzers for different aspects of code quality
and adherence to patterns, best practices, and framework conventions.
"""

from .architectural_analyzer import ArchitecturalAnalyzer
from .base_analyzer import AnalysisResult, BaseAnalyzer, PatternMatch, Recommendation
from .domain_adherence_analyzer import DomainAwareAdherenceAnalyzer
from .domain_classifier import ArchitecturalDomain, DomainClassifier
from .framework_analyzer import FrameworkAnalyzer
from .quality_analyzer import QualityAnalyzer
from .typescript_analyzer import TypeScriptAnalyzer

__all__ = [
    "BaseAnalyzer",
    "AnalysisResult",
    "PatternMatch",
    "Recommendation",
    "ArchitecturalAnalyzer",
    "QualityAnalyzer",
    "FrameworkAnalyzer",
    "TypeScriptAnalyzer",
    "DomainAwareAdherenceAnalyzer",
    "DomainClassifier",
    "ArchitecturalDomain",
]
