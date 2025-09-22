"""
Command-line interface for the Semantic Code Analyzer.

Provides easy-to-use CLI commands for analyzing semantic similarity of Git commits
with rich output formatting and progress tracking.
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import print as rprint

from .semantic_scorer import SemanticScorer, ScorerConfig
from .similarity_calculator import DistanceMetric

# Initialize Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool, quiet: bool):
    """Setup logging configuration based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


@click.group()
@click.version_option(version="0.1.0", package_name="semantic-code-analyzer")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Enable quiet mode (errors only)')
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """
    Semantic Code Analyzer - Analyze semantic similarity of Git commits.

    A tool for analyzing how semantically similar a Git commit is to the existing
    codebase using state-of-the-art code embeddings and optimized for Apple M3 hardware.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    setup_logging(verbose, quiet)


@cli.command()
@click.argument('commit_hash')
@click.option('--repo-path', '-r', default='.',
              help='Path to Git repository (default: current directory)')
@click.option('--language', '-l', default='python',
              type=click.Choice(['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go']),
              help='Programming language of the code')
@click.option('--model', '-m', default='microsoft/graphcodebert-base',
              help='Model to use for code embeddings')
@click.option('--distance-metric', '-d', default='euclidean',
              type=click.Choice(['euclidean', 'cosine', 'manhattan', 'chebyshev']),
              help='Distance metric for similarity calculation')
@click.option('--max-files', type=int, help='Maximum number of files to analyze')
@click.option('--output', '-o', type=click.Path(), help='Save results to JSON file')
@click.option('--detailed', is_flag=True, help='Show detailed per-file analysis')
@click.option('--no-cache', is_flag=True, help='Disable embedding caching')
@click.pass_context
def analyze(ctx, commit_hash: str, repo_path: str, language: str, model: str,
           distance_metric: str, max_files: Optional[int], output: Optional[str],
           detailed: bool, no_cache: bool):
    """
    Analyze semantic similarity of a specific commit against the codebase.

    COMMIT_HASH: The Git commit hash to analyze
    """
    try:
        # Create configuration
        config = ScorerConfig(
            model_name=model,
            distance_metric=distance_metric,
            max_files=max_files,
            detailed_output=detailed,
            cache_embeddings=not no_cache,
            save_results=bool(output)
        )

        # Display analysis start
        console.print(Panel.fit(
            f"[bold blue]Semantic Code Analysis[/bold blue]\n"
            f"Repository: {repo_path}\n"
            f"Commit: {commit_hash}\n"
            f"Language: {language}\n"
            f"Model: {model}",
            title="Starting Analysis"
        ))

        # Initialize scorer with progress indication
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:

            init_task = progress.add_task("Initializing semantic scorer...", total=None)
            scorer = SemanticScorer(repo_path, config)
            progress.update(init_task, completed=True)

            # Perform analysis
            analysis_task = progress.add_task("Analyzing commit similarity...", total=None)
            result = scorer.score_commit_similarity(commit_hash, language)
            progress.update(analysis_task, completed=True)

        # Display results
        _display_analysis_results(result, detailed)

        # Save results if requested
        if output:
            _save_results_to_file(result, output)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--repo-path', '-r', default='.',
              help='Path to Git repository (default: current directory)')
@click.option('--count', '-c', default=10, type=int,
              help='Number of recent commits to analyze')
@click.option('--branch', '-b', default='HEAD',
              help='Branch to analyze commits from')
@click.option('--language', '-l', default='python',
              type=click.Choice(['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go']),
              help='Programming language of the code')
@click.option('--model', '-m', default='microsoft/graphcodebert-base',
              help='Model to use for code embeddings')
@click.option('--output', '-o', type=click.Path(), help='Save results to JSON file')
@click.pass_context
def batch(ctx, repo_path: str, count: int, branch: str, language: str,
         model: str, output: Optional[str]):
    """
    Analyze semantic similarity for multiple recent commits.
    """
    try:
        config = ScorerConfig(
            model_name=model,
            save_results=bool(output)
        )

        console.print(Panel.fit(
            f"[bold blue]Batch Semantic Analysis[/bold blue]\n"
            f"Repository: {repo_path}\n"
            f"Branch: {branch}\n"
            f"Count: {count} commits\n"
            f"Language: {language}",
            title="Starting Batch Analysis"
        ))

        # Initialize scorer
        with console.status("[bold green]Initializing..."):
            scorer = SemanticScorer(repo_path, config)

        # Analyze recent commits
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task("Analyzing commits...", total=count)
            results = scorer.get_recent_commits_analysis(count, branch, language)

            for i, _ in enumerate(results):
                progress.update(task, advance=1)

        # Display batch results
        _display_batch_results(results)

        # Save results if requested
        if output:
            _save_batch_results_to_file(results, output)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('commit_a')
@click.argument('commit_b')
@click.option('--repo-path', '-r', default='.',
              help='Path to Git repository (default: current directory)')
@click.option('--language', '-l', default='python',
              type=click.Choice(['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go']),
              help='Programming language of the code')
@click.option('--output', '-o', type=click.Path(), help='Save results to JSON file')
@click.pass_context
def compare(ctx, commit_a: str, commit_b: str, repo_path: str,
           language: str, output: Optional[str]):
    """
    Compare semantic similarity between two commits.

    COMMIT_A: First commit hash
    COMMIT_B: Second commit hash
    """
    try:
        config = ScorerConfig(save_results=bool(output))

        console.print(Panel.fit(
            f"[bold blue]Commit Comparison[/bold blue]\n"
            f"Repository: {repo_path}\n"
            f"Commit A: {commit_a}\n"
            f"Commit B: {commit_b}\n"
            f"Language: {language}",
            title="Starting Comparison"
        ))

        # Initialize scorer
        with console.status("[bold green]Initializing..."):
            scorer = SemanticScorer(repo_path, config)

        # Compare commits
        with console.status("[bold green]Comparing commits..."):
            comparison_result = scorer.compare_commits(commit_a, commit_b, language)

        # Display comparison results
        _display_comparison_results(comparison_result)

        # Save results if requested
        if output:
            _save_results_to_file(comparison_result, output)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--repo-path', '-r', default='.',
              help='Path to Git repository (default: current directory)')
def info(repo_path: str):
    """
    Display information about the repository and scorer configuration.
    """
    try:
        config = ScorerConfig()

        with console.status("[bold green]Gathering information..."):
            scorer = SemanticScorer(repo_path, config)
            info_data = scorer.get_scorer_info()

        # Display information
        _display_info(info_data)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def _display_analysis_results(result, detailed: bool = False):
    """Display analysis results in a formatted table."""
    # Main results table
    table = Table(title="Similarity Analysis Results")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")
    table.add_column("Interpretation", style="green")

    # Add main scores
    max_sim = result.aggregate_scores.get('max_similarity', 0)
    mean_sim = result.aggregate_scores.get('mean_similarity', 0)

    table.add_row("Max Similarity", f"{max_sim:.3f}", _interpret_score(max_sim))
    table.add_row("Mean Similarity", f"{mean_sim:.3f}", _interpret_score(mean_sim))
    table.add_row("Processing Time", f"{result.processing_time:.2f}s", "")
    table.add_row("Files Analyzed", str(len(result.file_results)), "")

    console.print(table)

    # Commit information
    commit_info = result.commit_info
    console.print(Panel(
        f"[bold]Commit Information[/bold]\n"
        f"Hash: {commit_info.hash}\n"
        f"Author: {commit_info.author}\n"
        f"Message: {commit_info.message[:100]}...\n"
        f"Files Changed: {len(commit_info.files_changed)}\n"
        f"Insertions: +{commit_info.insertions}\n"
        f"Deletions: -{commit_info.deletions}",
        title="Commit Details"
    ))

    # Detailed file results
    if detailed and result.file_results:
        _display_detailed_file_results(result.file_results)


def _display_detailed_file_results(file_results):
    """Display detailed per-file analysis results."""
    for file_path, file_result in file_results.items():
        overall_sim = file_result['overall_similarity']
        similar_files = file_result['most_similar_files']

        # File similarity table
        file_table = Table(title=f"Analysis for {file_path}")
        file_table.add_column("Metric", style="cyan")
        file_table.add_column("Value", style="magenta")

        file_table.add_row("Max Similarity", f"{overall_sim['max_similarity']:.3f}")
        file_table.add_row("Mean Similarity", f"{overall_sim['mean_similarity']:.3f}")
        file_table.add_row("Std Similarity", f"{overall_sim['std_similarity']:.3f}")

        console.print(file_table)

        # Most similar files
        if similar_files:
            similar_table = Table(title="Most Similar Files")
            similar_table.add_column("Rank", style="yellow")
            similar_table.add_column("File", style="blue")
            similar_table.add_column("Similarity", style="green")

            for sim_file in similar_files[:5]:  # Top 5
                similar_table.add_row(
                    str(sim_file['rank']),
                    sim_file['file_path'],
                    f"{sim_file['similarity_score']:.3f}"
                )

            console.print(similar_table)
        console.print()


def _display_batch_results(results):
    """Display batch analysis results."""
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Summary table
    table = Table(title="Batch Analysis Results")
    table.add_column("Commit", style="cyan", no_wrap=True)
    table.add_column("Files", style="blue")
    table.add_column("Max Similarity", style="green")
    table.add_column("Mean Similarity", style="yellow")
    table.add_column("Author", style="magenta")

    for result in results:
        commit_info = result.commit_info
        aggregate = result.aggregate_scores

        table.add_row(
            commit_info.hash,
            str(len(result.file_results)),
            f"{aggregate.get('max_similarity', 0):.3f}",
            f"{aggregate.get('mean_similarity', 0):.3f}",
            commit_info.author[:20]
        )

    console.print(table)

    # Statistics
    max_similarities = [r.aggregate_scores.get('max_similarity', 0) for r in results]
    if max_similarities:
        avg_similarity = sum(max_similarities) / len(max_similarities)
        max_overall = max(max_similarities)
        min_overall = min(max_similarities)

        console.print(Panel(
            f"[bold]Batch Statistics[/bold]\n"
            f"Total Commits: {len(results)}\n"
            f"Average Max Similarity: {avg_similarity:.3f}\n"
            f"Highest Similarity: {max_overall:.3f}\n"
            f"Lowest Similarity: {min_overall:.3f}",
            title="Summary"
        ))


def _display_comparison_results(comparison_result):
    """Display commit comparison results."""
    result_a = comparison_result['commit_a']
    result_b = comparison_result['commit_b']
    similarity_diff = comparison_result['similarity_difference']

    # Comparison table
    table = Table(title="Commit Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Commit A", style="blue")
    table.add_column("Commit B", style="green")
    table.add_column("Difference", style="yellow")

    max_sim_a = result_a.aggregate_scores.get('max_similarity', 0)
    max_sim_b = result_b.aggregate_scores.get('max_similarity', 0)
    mean_sim_a = result_a.aggregate_scores.get('mean_similarity', 0)
    mean_sim_b = result_b.aggregate_scores.get('mean_similarity', 0)

    table.add_row(
        "Max Similarity",
        f"{max_sim_a:.3f}",
        f"{max_sim_b:.3f}",
        f"{similarity_diff['max_similarity_diff']:+.3f}"
    )
    table.add_row(
        "Mean Similarity",
        f"{mean_sim_a:.3f}",
        f"{mean_sim_b:.3f}",
        f"{similarity_diff['mean_similarity_diff']:+.3f}"
    )

    console.print(table)

    # Cross-similarity info
    cross_sim = comparison_result['cross_similarity']
    console.print(Panel(
        f"[bold]Cross-Similarity Analysis[/bold]\n"
        f"Matrix Shape: {cross_sim['shape']}\n"
        f"Max Cross-Similarity: {cross_sim['max_similarity']:.3f}\n"
        f"Mean Cross-Similarity: {cross_sim['mean_similarity']:.3f}",
        title="Direct Comparison"
    ))


def _display_info(info_data):
    """Display scorer and repository information."""
    # Repository info
    console.print(Panel(
        f"[bold]Repository Information[/bold]\n"
        f"Path: {info_data['repo_path']}\n"
        f"Supported Languages: {', '.join(info_data['supported_languages'])}",
        title="Repository"
    ))

    # Model info
    model_info = info_data['model_info']
    console.print(Panel(
        f"[bold]Model Configuration[/bold]\n"
        f"Model: {model_info['model_name']}\n"
        f"Device: {model_info['device']}\n"
        f"Max Length: {model_info['max_length']}\n"
        f"Embedding Dimension: {model_info['embedding_dim']}\n"
        f"Cache Size: {model_info['cache_size']} embeddings\n"
        f"MPS Available: {model_info['mps_available']}\n"
        f"CUDA Available: {model_info['cuda_available']}",
        title="Model"
    ))

    # Configuration
    config = info_data['config']
    console.print(Panel(
        f"[bold]Scorer Configuration[/bold]\n"
        f"Distance Metric: {config['distance_metric']}\n"
        f"Max Files: {config['max_files'] or 'Unlimited'}\n"
        f"Cache Embeddings: {config['cache_embeddings']}\n"
        f"Normalize Embeddings: {config['normalize_embeddings']}",
        title="Configuration"
    ))


def _interpret_score(score: float) -> str:
    """Interpret similarity score into human-readable text."""
    if score >= 0.8:
        return "Very High - Follows existing patterns closely"
    elif score >= 0.6:
        return "Good - Reasonably consistent with codebase"
    elif score >= 0.4:
        return "Moderate - Some alignment with existing code"
    elif score >= 0.2:
        return "Low - Different patterns from existing code"
    else:
        return "Very Low - Significantly different approach"


def _save_results_to_file(result, output_path: str):
    """Save results to JSON file."""
    from dataclasses import asdict

    # Convert to dictionary format
    if hasattr(result, '__dict__'):
        result_dict = asdict(result)
    else:
        result_dict = result

    # Make JSON serializable
    result_dict = _make_json_serializable(result_dict)

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)


def _save_batch_results_to_file(results, output_path: str):
    """Save batch results to JSON file."""
    from dataclasses import asdict

    results_list = []
    for result in results:
        result_dict = asdict(result)
        result_dict = _make_json_serializable(result_dict)
        results_list.append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)


def _make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()