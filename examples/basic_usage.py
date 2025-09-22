#!/usr/bin/env python3
"""
Basic usage examples for the Semantic Code Analyzer.

This script demonstrates how to use the SCA package for analyzing
semantic similarity of Git commits against a codebase.
"""

import os
import sys
from pathlib import Path

# Add the package to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_code_analyzer import SemanticScorer, ScorerConfig


def example_basic_analysis():
    """Basic example of analyzing a single commit."""
    print("🔍 Basic Commit Analysis Example")
    print("=" * 50)

    # Configuration for the analyzer
    config = ScorerConfig(
        model_name="microsoft/graphcodebert-base",
        distance_metric="euclidean",
        max_files=20,
        use_mps=True,  # Enable Apple M3 acceleration
        cache_embeddings=True,
        detailed_output=True
    )

    # Initialize the scorer with current repository
    repo_path = "."  # Current directory
    scorer = SemanticScorer(repo_path, config)

    # Get recent commits to analyze
    print("📋 Getting recent commits...")
    recent_commits = scorer.commit_extractor.get_commit_list(max_count=5)

    if not recent_commits:
        print("❌ No commits found in repository")
        return

    # Analyze the most recent commit
    latest_commit = recent_commits[0]
    print(f"📊 Analyzing commit: {latest_commit.hash}")
    print(f"   Message: {latest_commit.message}")
    print(f"   Author: {latest_commit.author}")
    print(f"   Files changed: {len(latest_commit.files_changed)}")

    # Perform the analysis
    try:
        result = scorer.score_commit_similarity(latest_commit.hash, language="python")

        # Display results
        print("\n✅ Analysis Results:")
        print(f"   Max Similarity: {result.aggregate_scores['max_similarity']:.3f}")
        print(f"   Mean Similarity: {result.aggregate_scores['mean_similarity']:.3f}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print(f"   Files Analyzed: {len(result.file_results)}")

        # Show per-file results
        if result.file_results:
            print("\n📄 Per-File Results:")
            for file_path, file_result in result.file_results.items():
                similarity = file_result['overall_similarity']['max_similarity']
                print(f"   {file_path}: {similarity:.3f}")

        # Interpretation
        max_sim = result.aggregate_scores['max_similarity']
        if max_sim >= 0.8:
            interpretation = "Very High - Follows existing patterns closely"
        elif max_sim >= 0.6:
            interpretation = "Good - Reasonably consistent with codebase"
        elif max_sim >= 0.4:
            interpretation = "Moderate - Some alignment but notable differences"
        elif max_sim >= 0.2:
            interpretation = "Low - Different patterns from existing code"
        else:
            interpretation = "Very Low - Significantly different approach"

        print(f"\n🎯 Interpretation: {interpretation}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")


def example_batch_analysis():
    """Example of analyzing multiple commits in batch."""
    print("\n🔄 Batch Analysis Example")
    print("=" * 50)

    config = ScorerConfig(
        distance_metric="euclidean",
        max_files=15,
        cache_embeddings=True
    )

    scorer = SemanticScorer(".", config)

    try:
        # Analyze last 5 commits
        print("📊 Analyzing last 5 commits...")
        results = scorer.get_recent_commits_analysis(max_commits=5, language="python")

        if not results:
            print("❌ No results found")
            return

        print(f"✅ Analyzed {len(results)} commits:")
        print()

        # Create a summary table
        print("| Commit  | Author      | Max Sim | Mean Sim | Files |")
        print("|---------|-------------|---------|----------|-------|")

        for result in results:
            commit_info = result.commit_info
            aggregate = result.aggregate_scores

            print(f"| {commit_info.hash} | {commit_info.author[:10]:10} | "
                  f"{aggregate['max_similarity']:.3f}   | "
                  f"{aggregate['mean_similarity']:.3f}    | "
                  f"{len(result.file_results):5} |")

        # Find the most and least similar commits
        similarities = [r.aggregate_scores['max_similarity'] for r in results]
        max_idx = similarities.index(max(similarities))
        min_idx = similarities.index(min(similarities))

        print(f"\n🎯 Most similar commit: {results[max_idx].commit_info.hash} "
              f"({similarities[max_idx]:.3f})")
        print(f"🎯 Least similar commit: {results[min_idx].commit_info.hash} "
              f"({similarities[min_idx]:.3f})")

    except Exception as e:
        print(f"❌ Error during batch analysis: {e}")


def example_commit_comparison():
    """Example of comparing two commits."""
    print("\n⚖️ Commit Comparison Example")
    print("=" * 50)

    config = ScorerConfig(
        distance_metric="euclidean",
        cache_embeddings=True
    )

    scorer = SemanticScorer(".", config)

    try:
        # Get recent commits
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=3)

        if len(recent_commits) < 2:
            print("❌ Need at least 2 commits for comparison")
            return

        commit_a = recent_commits[0]
        commit_b = recent_commits[1]

        print(f"📊 Comparing commits:")
        print(f"   Commit A: {commit_a.hash} - {commit_a.message}")
        print(f"   Commit B: {commit_b.hash} - {commit_b.message}")

        # Perform comparison
        comparison = scorer.compare_commits(commit_a.hash, commit_b.hash, language="python")

        # Display results
        result_a = comparison['commit_a']
        result_b = comparison['commit_b']
        cross_sim = comparison['cross_similarity']
        sim_diff = comparison['similarity_difference']

        print("\n✅ Comparison Results:")
        print(f"   Commit A Max Similarity: {result_a.aggregate_scores['max_similarity']:.3f}")
        print(f"   Commit B Max Similarity: {result_b.aggregate_scores['max_similarity']:.3f}")
        print(f"   Cross-Similarity: {cross_sim['max_similarity']:.3f}")
        print(f"   Similarity Difference: {sim_diff['max_similarity_diff']:+.3f}")

        if sim_diff['max_similarity_diff'] > 0:
            print("🎯 Commit B is more similar to the codebase")
        elif sim_diff['max_similarity_diff'] < 0:
            print("🎯 Commit A is more similar to the codebase")
        else:
            print("🎯 Both commits have similar similarity scores")

    except Exception as e:
        print(f"❌ Error during comparison: {e}")


def example_different_languages():
    """Example of analyzing different programming languages."""
    print("\n🌐 Multi-Language Analysis Example")
    print("=" * 50)

    config = ScorerConfig(
        distance_metric="euclidean",
        max_files=10
    )

    scorer = SemanticScorer(".", config)

    # Try different languages
    languages = ["python", "javascript", "java"]

    try:
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=1)
        if not recent_commits:
            print("❌ No commits found")
            return

        commit_hash = recent_commits[0].hash

        print(f"📊 Analyzing commit {commit_hash} with different language settings:")

        for language in languages:
            try:
                result = scorer.score_commit_similarity(commit_hash, language=language)
                max_sim = result.aggregate_scores['max_similarity']
                print(f"   {language:10}: {max_sim:.3f} (files: {len(result.file_results)})")
            except Exception as e:
                print(f"   {language:10}: Error - {e}")

    except Exception as e:
        print(f"❌ Error during multi-language analysis: {e}")


def example_distance_metrics():
    """Example of comparing different distance metrics."""
    print("\n📏 Distance Metrics Comparison")
    print("=" * 50)

    metrics = ["euclidean", "cosine", "manhattan", "chebyshev"]

    try:
        # Get a recent commit
        scorer = SemanticScorer(".", ScorerConfig())
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=1)

        if not recent_commits:
            print("❌ No commits found")
            return

        commit_hash = recent_commits[0].hash
        print(f"📊 Comparing distance metrics for commit {commit_hash}:")

        results = {}
        for metric in metrics:
            try:
                config = ScorerConfig(
                    distance_metric=metric,
                    max_files=10,
                    cache_embeddings=False  # Disable cache to ensure fresh calculation
                )

                scorer = SemanticScorer(".", config)
                result = scorer.score_commit_similarity(commit_hash, language="python")
                results[metric] = result.aggregate_scores['max_similarity']

                print(f"   {metric:10}: {results[metric]:.3f}")

            except Exception as e:
                print(f"   {metric:10}: Error - {e}")

        # Find best metric for this commit
        if results:
            best_metric = max(results, key=results.get)
            print(f"\n🎯 Best metric for this commit: {best_metric} ({results[best_metric]:.3f})")

    except Exception as e:
        print(f"❌ Error during distance metrics comparison: {e}")


def example_scorer_info():
    """Example of getting scorer information."""
    print("\n📋 Scorer Information Example")
    print("=" * 50)

    try:
        config = ScorerConfig()
        scorer = SemanticScorer(".", config)

        info = scorer.get_scorer_info()

        print("🔧 Configuration:")
        config_info = info['config']
        print(f"   Model: {config_info['model_name']}")
        print(f"   Distance Metric: {config_info['distance_metric']}")
        print(f"   Max Files: {config_info['max_files'] or 'Unlimited'}")
        print(f"   Use MPS: {config_info['use_mps']}")
        print(f"   Cache Embeddings: {config_info['cache_embeddings']}")

        print("\n🤖 Model Information:")
        model_info = info['model_info']
        print(f"   Device: {model_info['device']}")
        print(f"   Embedding Dimension: {model_info['embedding_dim']}")
        print(f"   Max Length: {model_info['max_length']}")
        print(f"   Cache Size: {model_info['cache_size']} embeddings")
        print(f"   MPS Available: {model_info['mps_available']}")

        print("\n📁 Repository:")
        print(f"   Path: {info['repo_path']}")
        print(f"   Supported Languages: {', '.join(info['supported_languages'])}")

    except Exception as e:
        print(f"❌ Error getting scorer info: {e}")


def main():
    """Run all examples."""
    print("🚀 Semantic Code Analyzer - Usage Examples")
    print("=" * 60)

    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ This script must be run from within a Git repository")
        print("   Please navigate to a Git repository and try again")
        return

    try:
        # Run examples
        example_basic_analysis()
        example_batch_analysis()
        example_commit_comparison()
        example_different_languages()
        example_distance_metrics()
        example_scorer_info()

        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("   - Try different configuration options")
        print("   - Experiment with different distance metrics")
        print("   - Analyze commits from different time periods")
        print("   - Use the CLI interface: sca-analyze --help")

    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()