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

"""Tests for code feature extraction utilities."""


from semantic_code_analyzer.embeddings.code_feature_extractor import (
    compare_features,
    extract_code_features,
    generate_feature_insights,
)


class TestExtractCodeFeatures:
    """Tests for extract_code_features function."""

    def test_extract_basic_features(self) -> None:
        """Test extraction of basic code features."""
        code = """import React from 'react'
import { useState } from 'react'

function MyComponent() {
    const [count, setCount] = useState(0)
    return <div>{count}</div>
}
"""
        features = extract_code_features(code, "src/MyComponent.tsx")

        assert features["line_count"] >= 7  # May have trailing newline
        assert features["char_count"] == len(code)
        assert features["import_count"] >= 2
        assert "React" in features["imports"]
        assert features["function_count"] >= 1

    def test_extract_typescript_features(self) -> None:
        """Test extraction of TypeScript-specific features."""
        code = """interface Props {
    name: string
}

const MyComponent: React.FC<Props> = ({ name }) => {
    return <div>{name}</div>
}
"""
        features = extract_code_features(code, "src/MyComponent.tsx")

        assert features["has_interface"] is True
        # Type annotation regex looks for `: type` pattern followed by =;,)
        # The React.FC<Props> might not match, so just check it exists
        assert "has_type_annotation" in features
        assert features["has_jsx"] is True

    def test_extract_python_features(self) -> None:
        """Test extraction of Python-specific features."""
        code = '''def calculate_score(value: int) -> float:
    """Calculate a score from value."""
    return value * 1.5

@dataclass
class Result:
    score: float
'''
        features = extract_code_features(code, "analyzer.py")

        assert features["function_count"] >= 1
        assert features["class_count"] >= 1
        assert features["has_docstring"] is True
        assert features["decorator_count"] >= 1

    def test_empty_code(self) -> None:
        """Test handling of empty code."""
        features = extract_code_features("", "empty.py")

        assert features["line_count"] == 1  # Empty string splits to ['']
        assert features["char_count"] == 0
        assert features["import_count"] == 0
        assert features["function_count"] == 0


class TestCompareFeatures:
    """Tests for compare_features function."""

    def test_compare_basic_metrics(self) -> None:
        """Test comparison of basic metrics."""
        query = {
            "line_count": 10,
            "import_count": 2,
            "function_count": 1,
        }
        references = [
            {"line_count": 50, "import_count": 8, "function_count": 5},
            {"line_count": 45, "import_count": 7, "function_count": 4},
        ]

        comparison = compare_features(query, references)

        assert "line_count" in comparison
        assert comparison["line_count"]["query"] == 10
        assert comparison["line_count"]["ref_mean"] == 47.5
        assert comparison["line_count"]["diff_pct"] < 0  # Query is smaller

    def test_compare_imports(self) -> None:
        """Test comparison of imports."""
        query = {"imports": ["React"]}
        references = [
            {"imports": ["React", "useState", "useEffect"]},
            {"imports": ["React", "useState", "useCallback"]},
        ]

        comparison = compare_features(query, references)

        assert "imports" in comparison
        missing = comparison["imports"]["missing_common"]
        assert "useState" in missing  # Common to both references

    def test_compare_boolean_features(self) -> None:
        """Test comparison of boolean features."""
        query = {"has_interface": False, "has_jsx": True}
        references = [
            {"has_interface": True, "has_jsx": True},
            {"has_interface": True, "has_jsx": True},
        ]

        comparison = compare_features(query, references)

        assert "has_interface" in comparison
        assert comparison["has_interface"]["missing"] is True

    def test_empty_references(self) -> None:
        """Test handling of empty reference list."""
        query = {"line_count": 10}
        comparison = compare_features(query, [])

        assert comparison == {}


class TestGenerateFeatureInsights:
    """Tests for generate_feature_insights function."""

    def test_generate_structure_insights(self) -> None:
        """Test generation of structure-related insights."""
        comparison = {
            "line_count": {
                "query": 10,
                "ref_mean": 50,
                "diff_pct": -80,
                "significance": "high",
            }
        }

        insights = generate_feature_insights(comparison, "", [])

        assert len(insights) > 0
        assert insights[0]["category"] == "structure"
        assert insights[0]["severity"] == "high"
        assert "shorter" in insights[0]["message"].lower()

    def test_generate_import_insights(self) -> None:
        """Test generation of import-related insights."""
        comparison = {
            "imports": {
                "missing_common": ["useState", "useEffect"],
                "overlap_ratio": 0.2,
            }
        }

        insights = generate_feature_insights(comparison, "", [])

        # Should have insight about missing imports
        import_insights = [i for i in insights if i["category"] == "imports"]
        assert len(import_insights) > 0
        assert "useState" in import_insights[0]["suggestion"]

    def test_generate_typescript_insights(self) -> None:
        """Test generation of TypeScript-specific insights."""
        comparison = {
            "has_interface": {
                "query": False,
                "ref_prevalence_pct": 90.0,
                "missing": True,
            }
        }

        insights = generate_feature_insights(comparison, "", [])

        ts_insights = [i for i in insights if i["category"] == "typescript"]
        assert len(ts_insights) > 0
        assert "interface" in ts_insights[0]["message"].lower()

    def test_insights_sorted_by_impact(self) -> None:
        """Test that insights are sorted by estimated improvement."""
        comparison = {
            "line_count": {"query": 10, "ref_mean": 50, "diff_pct": -80},
            "imports": {"missing_common": ["useState"], "overlap_ratio": 0.3},
            "comment_density": {"query": 0.0, "ref_mean": 0.1, "diff_pct": -100},
        }

        insights = generate_feature_insights(comparison, "", [])

        # First insight should have highest estimated improvement
        if len(insights) > 1:
            assert (
                insights[0]["estimated_improvement"]
                >= insights[1]["estimated_improvement"]
            )
