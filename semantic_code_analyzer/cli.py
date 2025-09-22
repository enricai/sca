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

"""CLI for multi-dimensional code analysis.

This module provides a command-line interface for comprehensive code quality
analysis through multi-dimensional pattern recognition.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from .scorers import EnhancedScorerConfig, MultiDimensionalScorer

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """Multi-Dimensional Code Analyzer for comprehensive code quality analysis."""
    # Set up logging
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("commit_hash")
@click.option(
    "--repo-path",
    "-r",
    default=".",
    help="Path to repository (default: current directory)",
)
@click.option("--output", "-o", help="Output file for results (JSON format)")
@click.option(
    "--architectural-weight",
    default=0.25,
    type=float,
    help="Weight for architectural patterns",
)
@click.option(
    "--quality-weight", default=0.25, type=float, help="Weight for code quality"
)
@click.option(
    "--framework-weight", default=0.20, type=float, help="Weight for framework patterns"
)
@click.option(
    "--disable-architectural", is_flag=True, help="Disable architectural analysis"
)
@click.option("--disable-quality", is_flag=True, help="Disable quality analysis")
@click.option("--disable-typescript", is_flag=True, help="Disable TypeScript analysis")
@click.option("--disable-framework", is_flag=True, help="Disable framework analysis")
@click.option(
    "--disable-domain-adherence",
    is_flag=True,
    help="Disable domain-aware adherence analysis",
)
@click.option(
    "--domain-adherence-weight",
    default=0.15,
    type=float,
    help="Weight for domain adherence analysis",
)
@click.option(
    "--similarity-threshold",
    default=0.3,
    type=float,
    help="Minimum similarity threshold for pattern matching",
)
@click.option(
    "--max-similar-patterns",
    default=10,
    type=int,
    help="Maximum number of similar patterns to consider",
)
@click.option(
    "--disable-pattern-indices",
    is_flag=True,
    help="Disable automatic pattern index building",
)
@click.option(
    "--max-recommendations",
    default=10,
    type=int,
    help="Maximum recommendations per file",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    commit_hash: str,
    repo_path: str,
    output: str | None,
    architectural_weight: float,
    quality_weight: float,
    framework_weight: float,
    domain_adherence_weight: float,
    similarity_threshold: float,
    max_similar_patterns: int,
    disable_architectural: bool,
    disable_quality: bool,
    disable_typescript: bool,
    disable_framework: bool,
    disable_domain_adherence: bool,
    disable_pattern_indices: bool,
    max_recommendations: int,
) -> None:
    """Perform multi-dimensional analysis on a commit."""
    console.print(f"[bold blue]Analyzing commit: {commit_hash}[/bold blue]")

    # Calculate TypeScript weight as remainder
    typescript_weight = 1.0 - (
        architectural_weight
        + quality_weight
        + framework_weight
        + domain_adherence_weight
    )

    # Validate all weights are positive and sum to 1.0
    if typescript_weight < 0:
        console.print(
            "[red]Error: Weights sum to more than 1.0. Reduce other weights.[/red]"
        )
        console.print(f"   Architectural: {architectural_weight}")
        console.print(f"   Quality: {quality_weight}")
        console.print(f"   Framework: {framework_weight}")
        console.print(f"   Domain Adherence: {domain_adherence_weight}")
        console.print(
            f"   Would leave {typescript_weight:.3f} for TypeScript (must be positive)"
        )
        sys.exit(1)

    if typescript_weight < 0.05:
        console.print(
            f"[yellow]Warning: TypeScript weight very low ({typescript_weight:.3f})[/yellow]"
        )

    # Create configuration
    config = EnhancedScorerConfig(
        architectural_weight=architectural_weight,
        quality_weight=quality_weight,
        framework_weight=framework_weight,
        typescript_weight=typescript_weight,
        domain_adherence_weight=domain_adherence_weight,
        enable_architectural_analysis=not disable_architectural,
        enable_quality_analysis=not disable_quality,
        enable_typescript_analysis=not disable_typescript,
        enable_framework_analysis=not disable_framework,
        enable_domain_adherence_analysis=not disable_domain_adherence,
        similarity_threshold=similarity_threshold,
        max_similar_patterns=max_similar_patterns,
        build_pattern_indices=not disable_pattern_indices,
        max_recommendations_per_file=max_recommendations,
        include_actionable_feedback=True,
        include_pattern_details=ctx.obj["verbose"],
    )

    try:
        # Initialize scorer
        with console.status("[bold green]Initializing analyzers..."):
            scorer = MultiDimensionalScorer(config, repo_path)

        # Perform analysis
        console.print("[bold green]Running multi-dimensional analysis...[/bold green]")
        start_time = time.time()

        results = scorer.analyze_commit(commit_hash)

        analysis_time = time.time() - start_time
        console.print(f"[green]Analysis completed in {analysis_time:.2f}s[/green]")

        # Display results
        _display_results(results, ctx.obj["verbose"])

        # Save results if requested
        if output:
            _save_results(results, output, scorer)
            console.print(f"[green]Results saved to {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        if ctx.obj["debug"]:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--base-commit", required=True, help="Base commit hash (your implementation)"
)
@click.option(
    "--compare-commits",
    required=True,
    help="Comma-separated list of commits to compare",
)
@click.option("--repo-path", "-r", default=".", help="Path to repository")
@click.option("--output", "-o", help="Output file for comparison results")
@click.pass_context
def compare(
    ctx: click.Context,
    base_commit: str,
    compare_commits: str,
    repo_path: str,
    output: str | None,
) -> None:
    """Compare multiple commits against a base implementation."""
    commits_to_compare = [c.strip() for c in compare_commits.split(",")]

    console.print(
        f"[bold blue]Comparing {len(commits_to_compare)} commits against base: {base_commit}[/bold blue]"
    )

    # Default configuration for comparison
    config = EnhancedScorerConfig()
    scorer = MultiDimensionalScorer(config, repo_path)

    # Analyze all commits
    all_results = {}

    # Analyze base commit
    console.print(f"[green]Analyzing base commit: {base_commit}[/green]")
    try:
        all_results["base"] = scorer.analyze_commit(base_commit)
    except Exception as e:
        console.print(f"[red]Error analyzing base commit: {e}[/red]")
        sys.exit(1)

    # Analyze comparison commits
    for commit in track(commits_to_compare, description="Analyzing commits..."):
        try:
            all_results[commit] = scorer.analyze_commit(commit)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to analyze {commit}: {e}[/yellow]")

    # Display comparison
    _display_comparison(all_results, base_commit)

    # Save comparison results
    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        console.print(f"[green]Comparison results saved to {output}[/green]")


def _display_results(results: dict[str, Any], verbose: bool = False) -> None:
    """Display analysis results in a formatted way."""
    # Overall score panel
    overall_score = results.get("overall_adherence", 0)
    score_color = _get_score_color(overall_score)

    console.print(
        Panel(
            f"[{score_color}]Overall Adherence: {overall_score:.3f}[/{score_color}]",
            title="Analysis Results",
            border_style=score_color,
        )
    )

    # Dimensional scores table
    dimensional_scores = results.get("dimensional_scores", {})
    if dimensional_scores:
        table = Table(title="Dimensional Scores")
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Interpretation", style="dim")

        for dimension, score in dimensional_scores.items():
            score_color = _get_score_color(score)
            interpretation = _get_score_interpretation(score)
            table.add_row(
                dimension.title(),
                f"[{score_color}]{score:.3f}[/{score_color}]",
                interpretation,
            )

        console.print(table)

    # Pattern analysis summary
    pattern_analysis = results.get("pattern_analysis", {})
    if pattern_analysis:
        console.print("\n[bold]Pattern Analysis:[/bold]")
        console.print(
            f"  Total patterns found: {pattern_analysis.get('total_patterns_found', 0)}"
        )
        console.print(
            f"  Average confidence: {pattern_analysis.get('pattern_confidence_avg', 0):.3f}"
        )

        patterns_by_type = pattern_analysis.get("patterns_by_type", {})
        if patterns_by_type:
            console.print("  Patterns by type:")
            for pattern_type, count in patterns_by_type.items():
                console.print(f"    {pattern_type}: {count}")

    # Actionable feedback
    feedback = results.get("actionable_feedback", [])
    if feedback:
        console.print("\n[bold red]Top Recommendations:[/bold red]")
        for i, rec in enumerate(feedback[:5], 1):  # Show top 5
            severity_color = _get_severity_color(rec["severity"])
            console.print(
                f"  {i}. [{severity_color}]{rec['severity'].upper()}[/{severity_color}]: {rec['message']}"
            )
            if rec.get("suggested_fix"):
                console.print(f"     [dim]Fix: {rec['suggested_fix']}[/dim]")

    # File-level analysis if verbose
    if verbose:
        file_analysis = results.get("file_level_analysis", {})
        if file_analysis:
            console.print("\n[bold]File-Level Analysis:[/bold]")
            for file_path, file_data in list(file_analysis.items())[
                :3
            ]:  # Show top 3 files
                console.print(f"  {file_path}:")
                scores = file_data.get("scores", {})
                for analyzer, score in scores.items():
                    score_color = _get_score_color(score)
                    console.print(
                        f"    {analyzer}: [{score_color}]{score:.3f}[/{score_color}]"
                    )


def _display_comparison(
    all_results: dict[str, dict[str, Any]], base_commit: str
) -> None:
    """Display comparison results between multiple commits."""
    # Comparison table
    table = Table(title="Commit Comparison")
    table.add_column("Commit", style="cyan")
    table.add_column("Overall Score", justify="right")
    table.add_column("Architectural", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("TypeScript", justify="right")
    table.add_column("Framework", justify="right")
    table.add_column("Domain Adherence", justify="right")

    # Add base commit first
    if base_commit in all_results:
        base_result = all_results[base_commit]
        _add_comparison_row(table, "BASE", base_result, is_base=True)

    # Add other commits
    for commit_hash, result in all_results.items():
        if commit_hash != "base" and commit_hash != base_commit:
            _add_comparison_row(table, commit_hash[:8], result)

    console.print(table)


def _add_comparison_row(
    table: Table, commit_label: str, result: dict[str, Any], is_base: bool = False
) -> None:
    """Add a row to the comparison table."""
    overall = result.get("overall_adherence", 0)
    dimensional = result.get("dimensional_scores", {})

    style = "bold green" if is_base else ""

    table.add_row(
        f"[{style}]{commit_label}[/{style}]",
        f"[{_get_score_color(overall)}]{overall:.3f}[/{_get_score_color(overall)}]",
        f"{dimensional.get('architectural', 0):.3f}",
        f"{dimensional.get('quality', 0):.3f}",
        f"{dimensional.get('typescript', 0):.3f}",
        f"{dimensional.get('framework', 0):.3f}",
        f"{dimensional.get('domain_adherence', 0):.3f}",
    )


def _save_results(
    results: dict[str, Any], output_path: str, scorer: MultiDimensionalScorer
) -> str:
    """Save results using the scorer's save method."""
    try:
        saved_path = scorer.save_results(results, output_path)
        return saved_path
    except Exception:
        # Fallback to simple JSON save
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return output_path


def _get_score_color(score: float) -> str:
    """Get color for score display."""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    else:
        return "red"


def _get_score_interpretation(score: float) -> str:
    """Get human-readable interpretation of score."""
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "Very Good"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.6:
        return "Fair"
    elif score >= 0.5:
        return "Needs Improvement"
    else:
        return "Poor"


def _get_severity_color(severity: str) -> str:
    """Get color for severity display."""
    severity_colors = {
        "critical": "bright_red",
        "error": "red",
        "warning": "yellow",
        "info": "blue",
    }
    return severity_colors.get(severity.lower(), "white")


if __name__ == "__main__":
    cli()
