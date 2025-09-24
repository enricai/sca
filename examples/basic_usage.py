#!/usr/bin/env python3
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

"""
Basic usage examples for the Multi-Dimensional Code Analyzer.

This script demonstrates how to use the enhanced code analysis package for
analyzing code quality patterns, architectural adherence, and best practices
in Git commits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the package to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from semantic_code_analyzer import EnhancedScorerConfig, MultiDimensionalScorer


def example_basic_analysis() -> None:
    """Basic example of analyzing a single commit."""
    console = Console()
    console.print("🔍 Basic Multi-Dimensional Analysis Example")
    console.print("=" * 50)

    # Configuration with default weights
    config = EnhancedScorerConfig()

    console.print("📊 Using configuration:")
    console.print(f"   Architectural: {config.architectural_weight}")
    console.print(f"   Quality: {config.quality_weight}")
    console.print(f"   TypeScript: {config.typescript_weight}")
    console.print(f"   Framework: {config.framework_weight}")

    # Initialize scorer
    scorer = MultiDimensionalScorer(config, repo_path=".")

    # Get latest commit
    try:
        import git

        repo = git.Repo(".")
        latest_commit = str(repo.head.commit)
        console.print(f"\n🔍 Analyzing latest commit: {latest_commit[:8]}")

        # Perform analysis
        results = scorer.analyze_commit(latest_commit)

        # Display results
        console.print("\n📊 Results:")
        console.print(f"   Overall adherence: {results['overall_adherence']:.3f}")
        console.print(f"   Confidence: {results['confidence']:.3f}")

        # Show dimensional breakdown
        dimensional_scores = results.get("dimensional_scores", {})
        console.print("\n📈 Dimensional Scores:")
        for dimension, score in dimensional_scores.items():
            console.print(f"   {dimension:12}: {score:.3f}")

        # Show pattern summary
        pattern_analysis = results.get("pattern_analysis", {})
        if pattern_analysis:
            console.print("\n🔍 Pattern Analysis:")
            console.print(
                f"   Total patterns: {pattern_analysis.get('total_patterns_found', 0)}"
            )
            console.print(
                f"   Avg confidence: {pattern_analysis.get('pattern_confidence_avg', 0):.3f}"
            )

        # Show top recommendations
        feedback = results.get("actionable_feedback", [])
        if feedback:
            console.print("\n💡 Top Recommendations:")
            for i, rec in enumerate(feedback[:3], 1):
                console.print(f"   {i}. [{rec['severity'].upper()}] {rec['message']}")

    except Exception as e:
        console.print(f"❌ Analysis failed: {e}")


def example_custom_weights() -> None:
    """Example with custom scoring weights."""
    console = Console()
    console.print("\n\n⚖️  Custom Weights Analysis Example")
    console.print("=" * 50)

    # Custom configuration emphasizing code quality
    config = EnhancedScorerConfig(
        architectural_weight=0.20,
        quality_weight=0.40,  # Emphasize quality
        typescript_weight=0.30,  # Emphasize type safety
        framework_weight=0.10,
    )

    console.print("📊 Custom configuration (quality-focused):")
    console.print(f"   Quality: {config.quality_weight} (emphasized)")
    console.print(f"   TypeScript: {config.typescript_weight} (emphasized)")
    console.print(f"   Architectural: {config.architectural_weight}")
    console.print(f"   Framework: {config.framework_weight}")

    # Example with mock files
    mock_files = {
        "src/components/QualityExample.tsx": """
'use client';

import React, { useState, useCallback, useMemo } from 'react';

interface QualityExampleProps {
    data: Array<{ id: string; name: string; }>;
    onSelect: (id: string) => void;
}

const QualityExample: React.FC<QualityExampleProps> = ({ data, onSelect }) => {
    const [selectedId, setSelectedId] = useState<string | null>(null);

    const handleSelect = useCallback((id: string) => {
        setSelectedId(id);
        onSelect(id);
    }, [onSelect]);

    const sortedData = useMemo(() => {
        return [...data].sort((a, b) => a.name.localeCompare(b.name));
    }, [data]);

    return (
        <div role="listbox" aria-label="Data selection">
            {sortedData.map((item) => (
                <button
                    key={item.id}
                    onClick={() => handleSelect(item.id)}
                    aria-selected={selectedId === item.id}
                    className={selectedId === item.id ? 'selected' : ''}
                >
                    {item.name}
                </button>
            ))}
        </div>
    );
};

export default QualityExample;
"""
    }

    scorer = MultiDimensionalScorer(config, repo_path=".")
    results = scorer.analyze_files(mock_files)

    console.print("\n📊 Results with custom weights:")
    console.print(f"   Overall adherence: {results['overall_adherence']:.3f}")

    dimensional_scores = results.get("dimensional_scores", {})
    for dimension, score in dimensional_scores.items():
        console.print(f"   {dimension:12}: {score:.3f}")


def example_file_analysis() -> None:
    """Example of analyzing files without git context."""
    console = Console()
    console.print("\n\n📁 File Analysis Example")
    console.print("=" * 50)

    # Files with different quality levels
    files_to_analyze = {
        "good_component.tsx": """
import React, { useState } from 'react';

interface GoodComponentProps {
    title: string;
}

const GoodComponent: React.FC<GoodComponentProps> = ({ title }) => {
    const [count, setCount] = useState<number>(0);

    return (
        <div>
            <h1>{title}</h1>
            <button onClick={() => setCount(c => c + 1)}>
                Count: {count}
            </button>
        </div>
    );
};

export default GoodComponent;
""",
        "poor_component.js": """
import React from 'react';

function Thing(props) {
    const [stuff, setStuff] = React.useState();

    return <div onClick={() => setStuff(!stuff)}>{props.title}</div>;
}

export default Thing;
""",
    }

    config = EnhancedScorerConfig()
    scorer = MultiDimensionalScorer(config, repo_path=".")

    results = scorer.analyze_files(files_to_analyze)

    console.print("📊 File Analysis Results:")
    console.print(f"   Overall adherence: {results['overall_adherence']:.3f}")

    # Show file-level breakdown
    file_analysis = results.get("file_level_analysis", {})
    for file_path, file_data in file_analysis.items():
        console.print(f"\n   📄 {file_path}:")
        scores = file_data.get("scores", {})
        for analyzer, score in scores.items():
            console.print(f"      {analyzer}: {score:.3f}")


if __name__ == "__main__":
    console = Console()
    try:
        example_basic_analysis()
        example_custom_weights()
        example_file_analysis()

        console.print("\n🎉 All examples completed successfully!")
        console.print("\n💡 Next steps:")
        console.print(
            "   • Run: python -m semantic_code_analyzer.cli analyze <commit_hash>"
        )
        console.print(
            "   • Compare: python -m semantic_code_analyzer.cli compare --base-commit <base> --compare-commits <others>"
        )

    except Exception as e:
        console.print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
