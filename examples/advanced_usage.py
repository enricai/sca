#!/usr/bin/env python3
"""
Advanced usage examples for the Semantic Code Analyzer.

This script demonstrates advanced features including custom configurations,
performance optimization, and integration with other tools.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

# Add the package to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_code_analyzer import SemanticScorer, ScorerConfig
from semantic_code_analyzer.similarity_calculator import DistanceMetric
from semantic_code_analyzer.code_embedder import EmbeddingConfig


def example_custom_configuration():
    """Example of creating custom configurations for different use cases."""
    print("🔧 Custom Configuration Examples")
    print("=" * 50)

    # Configuration for large codebases
    large_codebase_config = ScorerConfig(
        model_name="microsoft/graphcodebert-base",
        distance_metric="euclidean",
        max_files=500,  # Limit for performance
        use_mps=True,
        cache_embeddings=True,
        normalize_embeddings=True,
        detailed_output=False,  # Reduce output for large analysis
        save_results=True
    )

    # Configuration for detailed analysis
    detailed_analysis_config = ScorerConfig(
        model_name="microsoft/graphcodebert-base",
        distance_metric="cosine",
        max_files=50,
        include_functions=True,  # Analyze individual functions
        detailed_output=True,
        cache_embeddings=True
    )

    # Configuration for performance benchmarking
    benchmark_config = ScorerConfig(
        model_name="microsoft/graphcodebert-base",
        distance_metric="euclidean",
        max_files=100,
        cache_embeddings=False,  # Fresh calculations
        normalize_embeddings=True,
        use_mps=True
    )

    print("📋 Configuration Examples Created:")
    print(f"   Large Codebase Config: max_files={large_codebase_config.max_files}")
    print(f"   Detailed Analysis Config: detailed_output={detailed_analysis_config.detailed_output}")
    print(f"   Benchmark Config: cache_embeddings={benchmark_config.cache_embeddings}")

    return {
        'large_codebase': large_codebase_config,
        'detailed': detailed_analysis_config,
        'benchmark': benchmark_config
    }


def example_performance_optimization():
    """Example of optimizing performance for different scenarios."""
    print("\n⚡ Performance Optimization Examples")
    print("=" * 50)

    # Fast analysis configuration
    fast_config = ScorerConfig(
        distance_metric="euclidean",  # Fastest metric
        max_files=20,  # Limit file count
        cache_embeddings=True,  # Enable caching
        use_mps=True,  # Use Apple M3 acceleration
        normalize_embeddings=False,  # Skip normalization
        detailed_output=False  # Minimal output
    )

    scorer = SemanticScorer(".", fast_config)

    try:
        # Get a commit to analyze
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=3)
        if not recent_commits:
            print("❌ No commits found")
            return

        # Benchmark different configurations
        configs = {
            "Fast": fast_config,
            "Cached": ScorerConfig(cache_embeddings=True, max_files=20),
            "Uncached": ScorerConfig(cache_embeddings=False, max_files=20)
        }

        commit_hash = recent_commits[0].hash
        print(f"📊 Performance comparison for commit {commit_hash}:")

        results = {}
        for config_name, config in configs.items():
            try:
                scorer = SemanticScorer(".", config)

                start_time = time.time()
                result = scorer.score_commit_similarity(commit_hash, language="python")
                end_time = time.time()

                processing_time = end_time - start_time
                results[config_name] = {
                    'time': processing_time,
                    'similarity': result.aggregate_scores['max_similarity'],
                    'files': len(result.file_results)
                }

                print(f"   {config_name:10}: {processing_time:.2f}s | "
                      f"Similarity: {result.aggregate_scores['max_similarity']:.3f} | "
                      f"Files: {len(result.file_results)}")

            except Exception as e:
                print(f"   {config_name:10}: Error - {e}")

        # Performance insights
        if results:
            fastest = min(results, key=lambda x: results[x]['time'])
            print(f"\n🏃 Fastest configuration: {fastest} ({results[fastest]['time']:.2f}s)")

    except Exception as e:
        print(f"❌ Error during performance optimization: {e}")


def example_similarity_analysis_pipeline():
    """Example of a complete similarity analysis pipeline."""
    print("\n🔄 Similarity Analysis Pipeline")
    print("=" * 50)

    config = ScorerConfig(
        distance_metric="euclidean",
        max_files=30,
        cache_embeddings=True,
        detailed_output=True
    )

    scorer = SemanticScorer(".", config)

    try:
        # Step 1: Analyze recent commits
        print("📊 Step 1: Analyzing recent commits...")
        recent_results = scorer.get_recent_commits_analysis(max_commits=10, language="python")

        if not recent_results:
            print("❌ No commits found")
            return

        # Step 2: Calculate statistics
        print("📈 Step 2: Calculating statistics...")
        similarities = [r.aggregate_scores['max_similarity'] for r in recent_results]

        stats = {
            'mean': sum(similarities) / len(similarities),
            'max': max(similarities),
            'min': min(similarities),
            'std': (sum((x - sum(similarities) / len(similarities)) ** 2 for x in similarities) / len(similarities)) ** 0.5
        }

        print(f"   Mean similarity: {stats['mean']:.3f}")
        print(f"   Max similarity: {stats['max']:.3f}")
        print(f"   Min similarity: {stats['min']:.3f}")
        print(f"   Std deviation: {stats['std']:.3f}")

        # Step 3: Identify outliers
        print("🎯 Step 3: Identifying outliers...")
        threshold = stats['mean'] - 2 * stats['std']
        outliers = []

        for result in recent_results:
            similarity = result.aggregate_scores['max_similarity']
            if similarity < threshold:
                outliers.append({
                    'commit': result.commit_info.hash,
                    'similarity': similarity,
                    'message': result.commit_info.message,
                    'author': result.commit_info.author
                })

        if outliers:
            print(f"   Found {len(outliers)} outlier commits:")
            for outlier in outliers:
                print(f"     {outlier['commit']}: {outlier['similarity']:.3f} - {outlier['message'][:50]}...")
        else:
            print("   No outliers found")

        # Step 4: Generate recommendations
        print("💡 Step 4: Generating recommendations...")
        if stats['mean'] < 0.4:
            print("   ⚠️  Low average similarity - consider code review process improvements")
        elif stats['mean'] > 0.8:
            print("   ✅ High average similarity - good consistency with codebase")
        else:
            print("   📊 Moderate similarity - room for improvement")

        if stats['std'] > 0.3:
            print("   ⚠️  High variability - inconsistent coding patterns")
        else:
            print("   ✅ Low variability - consistent coding patterns")

        # Step 5: Export results
        print("💾 Step 5: Exporting results...")
        export_data = {
            'analysis_timestamp': time.time(),
            'repository_path': str(scorer.repo_path),
            'configuration': {
                'model_name': config.model_name,
                'distance_metric': config.distance_metric,
                'max_files': config.max_files
            },
            'statistics': stats,
            'outliers': outliers,
            'commits_analyzed': len(recent_results),
            'detailed_results': [
                {
                    'commit_hash': r.commit_info.hash,
                    'similarity': r.aggregate_scores['max_similarity'],
                    'files_count': len(r.file_results),
                    'processing_time': r.processing_time
                }
                for r in recent_results
            ]
        }

        # Save to file
        output_file = Path("similarity_analysis_pipeline.json")
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"   Results exported to: {output_file}")

    except Exception as e:
        print(f"❌ Error in analysis pipeline: {e}")


def example_function_level_analysis():
    """Example of analyzing individual functions within commits."""
    print("\n🔍 Function-Level Analysis")
    print("=" * 50)

    config = ScorerConfig(
        include_functions=True,
        detailed_output=True,
        max_files=10
    )

    scorer = SemanticScorer(".", config)

    try:
        # Get recent Python commits
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=1)
        if not recent_commits:
            print("❌ No commits found")
            return

        commit_hash = recent_commits[0].hash
        print(f"📊 Analyzing functions in commit {commit_hash}...")

        # Extract commit changes
        commit_changes = scorer.commit_extractor.extract_commit_changes(commit_hash)

        if not commit_changes:
            print("❌ No code changes found in commit")
            return

        # Analyze each file's functions
        for file_path, code in commit_changes.items():
            if not file_path.endswith('.py'):
                continue

            print(f"\n📄 Analyzing file: {file_path}")

            # Extract functions from the code
            try:
                functions = scorer.code_embedder.get_function_embeddings(code, "python")

                if not functions:
                    print("   No functions found")
                    continue

                print(f"   Found {len(functions)} functions:")

                for func in functions:
                    print(f"     - {func.name} (lines {func.line_start}-{func.line_end})")

                    # Get similarity for this function
                    if func.embedding is not None:
                        # Compare against existing codebase
                        existing_codebase = scorer.commit_extractor.get_existing_codebase(
                            exclude_files=[file_path], max_files=20
                        )

                        if existing_codebase:
                            codebase_embeddings = []
                            for cb_code in existing_codebase.values():
                                try:
                                    embedding = scorer.code_embedder.get_code_embedding(cb_code, "python")
                                    codebase_embeddings.append(embedding)
                                except:
                                    continue

                            if codebase_embeddings:
                                similarity_result = scorer.similarity_calculator.calculate_similarity_score(
                                    func.embedding, codebase_embeddings, return_details=True
                                )
                                print(f"       Similarity: {similarity_result.max_similarity:.3f}")

            except Exception as e:
                print(f"   Error analyzing functions: {e}")

    except Exception as e:
        print(f"❌ Error in function-level analysis: {e}")


def example_cross_repository_comparison():
    """Example of comparing commits across different repositories."""
    print("\n🔄 Cross-Repository Comparison")
    print("=" * 50)

    # This example would require access to multiple repositories
    # For demo purposes, we'll show the structure

    config = ScorerConfig(
        distance_metric="euclidean",
        max_files=20,
        cache_embeddings=True
    )

    print("📋 Cross-repository comparison structure:")
    print("   1. Initialize scorers for each repository")
    print("   2. Extract embeddings from commits in each repo")
    print("   3. Calculate cross-repository similarity matrix")
    print("   4. Identify similar patterns across projects")

    # Example structure (would require actual repositories)
    repository_paths = [
        ".",  # Current repository
        # "/path/to/repo2",
        # "/path/to/repo3"
    ]

    print(f"\n📊 Analyzing {len(repository_paths)} repositories:")

    for i, repo_path in enumerate(repository_paths):
        try:
            scorer = SemanticScorer(repo_path, config)
            recent_commits = scorer.commit_extractor.get_commit_list(max_count=3)

            print(f"   Repository {i+1}: {repo_path}")
            print(f"     Recent commits: {len(recent_commits)}")

            if recent_commits:
                # Analyze latest commit
                latest = recent_commits[0]
                result = scorer.score_commit_similarity(latest.hash, language="python")
                print(f"     Latest commit similarity: {result.aggregate_scores['max_similarity']:.3f}")

        except Exception as e:
            print(f"   Repository {i+1}: Error - {e}")

    print("\n💡 Note: Full cross-repository comparison requires multiple repos")


def example_integration_with_ci_cd():
    """Example of integrating SCA with CI/CD pipelines."""
    print("\n🔗 CI/CD Integration Example")
    print("=" * 50)

    def check_commit_quality(commit_hash: str, threshold: float = 0.3) -> Dict:
        """
        Check if a commit meets quality standards.

        Args:
            commit_hash: Hash of commit to check
            threshold: Minimum similarity threshold

        Returns:
            Dictionary with check results
        """
        config = ScorerConfig(
            distance_metric="euclidean",
            max_files=50,
            cache_embeddings=True,
            detailed_output=False
        )

        try:
            scorer = SemanticScorer(".", config)
            result = scorer.score_commit_similarity(commit_hash, language="python")

            similarity = result.aggregate_scores['max_similarity']
            passed = similarity >= threshold

            return {
                'commit_hash': commit_hash,
                'similarity_score': similarity,
                'threshold': threshold,
                'passed': passed,
                'files_analyzed': len(result.file_results),
                'processing_time': result.processing_time,
                'recommendation': _get_recommendation(similarity)
            }

        except Exception as e:
            return {
                'commit_hash': commit_hash,
                'error': str(e),
                'passed': False
            }

    def _get_recommendation(similarity: float) -> str:
        """Get recommendation based on similarity score."""
        if similarity >= 0.8:
            return "Excellent - Code follows established patterns"
        elif similarity >= 0.6:
            return "Good - Minor style variations"
        elif similarity >= 0.4:
            return "Acceptable - Consider code review"
        elif similarity >= 0.2:
            return "Warning - Significant style differences"
        else:
            return "Alert - Major style inconsistencies"

    # Example CI/CD check
    try:
        scorer = SemanticScorer(".", ScorerConfig())
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=1)

        if recent_commits:
            commit_hash = recent_commits[0].hash
            print(f"🔍 Running CI/CD quality check for commit {commit_hash}...")

            check_result = check_commit_quality(commit_hash, threshold=0.3)

            print("\n📊 Quality Check Results:")
            if 'error' in check_result:
                print(f"   ❌ Error: {check_result['error']}")
                print("   Exit code: 1")
            else:
                status = "✅ PASSED" if check_result['passed'] else "❌ FAILED"
                print(f"   Status: {status}")
                print(f"   Similarity Score: {check_result['similarity_score']:.3f}")
                print(f"   Threshold: {check_result['threshold']}")
                print(f"   Files Analyzed: {check_result['files_analyzed']}")
                print(f"   Processing Time: {check_result['processing_time']:.2f}s")
                print(f"   Recommendation: {check_result['recommendation']}")

                exit_code = 0 if check_result['passed'] else 1
                print(f"   Exit code: {exit_code}")

                # Example of saving results for CI/CD systems
                ci_output = {
                    'semantic_analysis': check_result,
                    'ci_metadata': {
                        'pipeline_id': 'example-pipeline-123',
                        'build_number': 456,
                        'timestamp': time.time()
                    }
                }

                with open('ci_semantic_analysis.json', 'w') as f:
                    json.dump(ci_output, f, indent=2)

                print("   Results saved to: ci_semantic_analysis.json")

        else:
            print("❌ No commits found for CI/CD check")

    except Exception as e:
        print(f"❌ Error in CI/CD integration: {e}")


def example_custom_analysis_workflow():
    """Example of creating a custom analysis workflow."""
    print("\n🎯 Custom Analysis Workflow")
    print("=" * 50)

    class CustomAnalysisWorkflow:
        """Custom workflow for specialized analysis needs."""

        def __init__(self, repo_path: str):
            self.repo_path = repo_path
            self.config = ScorerConfig(
                distance_metric="euclidean",
                max_files=100,
                cache_embeddings=True
            )
            self.scorer = SemanticScorer(repo_path, self.config)

        def analyze_commits_by_author(self, author: str, max_commits: int = 10):
            """Analyze commits by a specific author."""
            print(f"👤 Analyzing commits by {author}...")

            # Get all recent commits
            all_commits = self.scorer.commit_extractor.get_commit_list(max_count=50)

            # Filter by author
            author_commits = [c for c in all_commits if author.lower() in c.author.lower()][:max_commits]

            if not author_commits:
                print(f"   No commits found for author: {author}")
                return []

            print(f"   Found {len(author_commits)} commits by {author}")

            # Analyze each commit
            results = []
            for commit in author_commits:
                try:
                    result = self.scorer.score_commit_similarity(commit.hash, language="python")
                    results.append({
                        'commit': commit,
                        'result': result
                    })
                except Exception as e:
                    print(f"   Error analyzing {commit.hash}: {e}")

            return results

        def compare_time_periods(self, days_ago_1: int = 30, days_ago_2: int = 90):
            """Compare similarity patterns between different time periods."""
            print(f"⏰ Comparing commits from {days_ago_1} vs {days_ago_2} days ago...")

            import datetime

            # Get recent commits
            all_commits = self.scorer.commit_extractor.get_commit_list(max_count=100)

            now = datetime.datetime.now()
            period_1_cutoff = now - datetime.timedelta(days=days_ago_1)
            period_2_cutoff = now - datetime.timedelta(days=days_ago_2)

            # Note: This is simplified - actual implementation would parse commit timestamps
            period_1_commits = all_commits[:10]  # Recent commits
            period_2_commits = all_commits[10:20]  # Older commits

            print(f"   Period 1 (recent): {len(period_1_commits)} commits")
            print(f"   Period 2 (older): {len(period_2_commits)} commits")

            # Analyze both periods
            results = {}
            for period_name, commits in [("recent", period_1_commits), ("older", period_2_commits)]:
                similarities = []
                for commit in commits[:5]:  # Limit for demo
                    try:
                        result = self.scorer.score_commit_similarity(commit.hash, language="python")
                        similarities.append(result.aggregate_scores['max_similarity'])
                    except:
                        continue

                if similarities:
                    results[period_name] = {
                        'mean': sum(similarities) / len(similarities),
                        'max': max(similarities),
                        'min': min(similarities),
                        'count': len(similarities)
                    }

            # Compare results
            if len(results) == 2:
                recent_mean = results['recent']['mean']
                older_mean = results['older']['mean']
                improvement = recent_mean - older_mean

                print(f"   Recent period mean similarity: {recent_mean:.3f}")
                print(f"   Older period mean similarity: {older_mean:.3f}")
                print(f"   Improvement: {improvement:+.3f}")

                if improvement > 0.1:
                    print("   📈 Significant improvement in code consistency")
                elif improvement < -0.1:
                    print("   📉 Decline in code consistency")
                else:
                    print("   📊 Stable code consistency")

    # Run custom workflow
    try:
        workflow = CustomAnalysisWorkflow(".")

        # Example: Analyze commits by author
        scorer = SemanticScorer(".", ScorerConfig())
        recent_commits = scorer.commit_extractor.get_commit_list(max_count=5)

        if recent_commits:
            # Get the most recent author
            recent_author = recent_commits[0].author
            author_results = workflow.analyze_commits_by_author(recent_author, max_commits=3)

            if author_results:
                similarities = [r['result'].aggregate_scores['max_similarity'] for r in author_results]
                avg_similarity = sum(similarities) / len(similarities)
                print(f"   Average similarity for {recent_author}: {avg_similarity:.3f}")

        # Example: Compare time periods
        workflow.compare_time_periods()

    except Exception as e:
        print(f"❌ Error in custom workflow: {e}")


def main():
    """Run all advanced examples."""
    print("🚀 Semantic Code Analyzer - Advanced Usage Examples")
    print("=" * 70)

    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ This script must be run from within a Git repository")
        return

    try:
        # Run advanced examples
        example_custom_configuration()
        example_performance_optimization()
        example_similarity_analysis_pipeline()
        example_function_level_analysis()
        example_cross_repository_comparison()
        example_integration_with_ci_cd()
        example_custom_analysis_workflow()

        print("\n🎉 All advanced examples completed!")
        print("\nAdvanced Features Demonstrated:")
        print("   ✅ Custom configurations for different use cases")
        print("   ✅ Performance optimization techniques")
        print("   ✅ Complete analysis pipelines")
        print("   ✅ Function-level analysis")
        print("   ✅ CI/CD integration patterns")
        print("   ✅ Custom workflow development")

    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()