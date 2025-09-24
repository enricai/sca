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
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, track
from rich.table import Table

from .hardware import DeviceManager, DeviceType
from .scorers import EnhancedScorerConfig, MultiDimensionalScorer

console = Console()
logger = logging.getLogger(__name__)


def _run_pre_analysis_health_check(
    device_preference: str, console: Console
) -> tuple[bool, str]:
    """Run pre-analysis health checks and get user confirmation if needed.

    Args:
        device_preference: Preferred device type (auto, cpu, mps, cuda)
        console: Rich console for user interaction

    Returns:
        Tuple of (continue_analysis, device_to_use) where:
        - continue_analysis: True if analysis should continue, False if user wants to abort
        - device_to_use: Device preference, potentially modified based on user choice
    """
    if device_preference != "auto" and device_preference != "mps":
        return True, device_preference  # Skip health check for non-MPS devices

    # Check if we're on macOS with potential MPS issues
    import platform

    import torch

    if platform.system() != "Darwin":
        return True, device_preference  # Skip MPS checks on non-macOS

    console.print("[dim]🔍 Running pre-analysis hardware health check...[/dim]")

    issues_found = []
    warnings = []

    # Check PyTorch MPS availability
    try:
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            issues_found.append("MPS not available in current PyTorch installation")
        else:
            # Test basic MPS operations
            try:
                test_tensor = torch.randn(2, 2, device="mps")
                torch.mm(test_tensor, test_tensor)
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception as e:
                issues_found.append(f"MPS operations test failed: {str(e)[:100]}")
    except Exception as e:
        issues_found.append(f"MPS compatibility check failed: {str(e)[:100]}")

    # Check PyTorch version for known MPS issues
    try:
        torch_version = torch.__version__
        if torch_version.startswith("1."):
            warnings.append(
                f"PyTorch {torch_version} has limited MPS support - consider upgrading"
            )
    except Exception:
        warnings.append("Could not determine PyTorch version")

    if not issues_found and not warnings:
        console.print("[dim]✅ Hardware health check passed[/dim]")
        return True, device_preference

    # Display issues to user
    if issues_found:
        console.print("[yellow]⚠️  Hardware acceleration issues detected:[/yellow]")
        for issue in issues_found:
            console.print(f"[dim]  • {issue}[/dim]")

    if warnings:
        console.print("[yellow]⚠️  Hardware warnings:[/yellow]")
        for warning in warnings:
            console.print(f"[dim]  • {warning}[/dim]")

    # Ask user how to proceed
    console.print("")
    console.print("[dim]Options:[/dim]")
    console.print("[dim]  1. Continue with CPU fallback (slower but stable)[/dim]")
    console.print(
        "[dim]  2. Continue with MPS acceleration (may fail during analysis)[/dim]"
    )
    console.print("[dim]  3. Abort and fix issues first[/dim]")

    while True:
        choice = input("How would you like to proceed? (1/2/3): ").strip()
        if choice == "1":
            console.print("[dim]Using CPU fallback for stability[/dim]")
            return True, "cpu"
        elif choice == "2":
            console.print(
                "[yellow]⚠️  Continuing with MPS - may experience failures[/yellow]"
            )
            return True, device_preference
        elif choice == "3":
            console.print(
                "[dim]Analysis aborted. Please address the issues above.[/dim]"
            )
            return False, device_preference
        else:
            console.print("[red]Please enter 1, 2, or 3[/red]")


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
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda"]),
    default="auto",
    help="Hardware device preference for AI model acceleration (auto, cpu, mps, cuda)",
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
    device: str,
) -> None:
    """Perform multi-dimensional analysis on a commit."""
    console.print(f"[bold blue]Analyzing commit: {commit_hash}[/bold blue]")

    logger.info("=== STARTING COMMIT ANALYSIS ===")
    logger.info(f"Target commit: {commit_hash}")
    logger.info(f"Repository path: {repo_path}")
    logger.info(f"Output file: {output}")
    logger.info(f"Device preference: {device}")
    logger.info(f"Debug mode: {ctx.obj['debug']}")
    logger.info(f"Verbose mode: {ctx.obj['verbose']}")

    # Calculate TypeScript weight as remainder
    logger.info("=== WEIGHT CALCULATION ===")
    logger.info(f"Architectural weight: {architectural_weight}")
    logger.info(f"Quality weight: {quality_weight}")
    logger.info(f"Framework weight: {framework_weight}")
    logger.info(f"Domain adherence weight: {domain_adherence_weight}")

    typescript_weight = 1.0 - (
        architectural_weight
        + quality_weight
        + framework_weight
        + domain_adherence_weight
    )
    logger.info(f"Calculated TypeScript weight: {typescript_weight}")

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
    logger.info("=== CREATING CONFIGURATION ===")
    logger.info("Creating EnhancedScorerConfig...")

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
    logger.info("EnhancedScorerConfig created successfully")

    try:
        # Initialize DeviceManager for hardware acceleration
        logger.info("=== DEVICE MANAGER INITIALIZATION ===")
        device_preference = None if device == "auto" else DeviceType(device.lower())
        logger.info(f"Device preference: {device_preference}")

        # Run pre-analysis health check for MPS devices
        should_continue, final_device = _run_pre_analysis_health_check(device, console)
        if not should_continue:
            console.print("[yellow]Analysis aborted by user[/yellow]")
            sys.exit(0)

        # Use the device choice from health check (may have changed to CPU)
        device = final_device
        device_preference = None if device == "auto" else DeviceType(device.lower())

        logger.info("Starting DeviceManager initialization...")

        with console.status("[bold green]Initializing hardware acceleration..."):
            try:
                device_manager = DeviceManager(prefer_device=device_preference)
            except Exception as e:
                logger.error(f"CRITICAL: DeviceManager initialization failed: {e}")
                console.print(
                    f"[red]❌ Hardware acceleration initialization failed: {e}[/red]"
                )
                if "mps" in str(e).lower():
                    console.print(
                        "[yellow]⚠️  MPS acceleration unavailable - analysis will be slower[/yellow]"
                    )
                    console.print(
                        "[dim]   Suggestion: Check PyTorch installation or use --device cpu[/dim]"
                    )
                raise

        logger.info("DeviceManager initialized successfully")

        # Display hardware information with enhanced status reporting
        logger.info("=== HARDWARE INFORMATION ===")
        hardware_info = device_manager.hardware_info
        device_status = device_manager.get_device_status_report()

        logger.info(f"Device name: {hardware_info.device_name}")
        logger.info(f"Device type: {hardware_info.device_type}")
        logger.info(f"Platform: {hardware_info.platform}")
        logger.info(f"Architecture: {hardware_info.architecture}")
        logger.info(f"Memory GB: {hardware_info.memory_gb}")
        logger.info(f"Apple Silicon: {hardware_info.is_apple_silicon}")
        logger.info(f"Chip generation: {hardware_info.chip_generation}")
        logger.info(f"Supports MPS: {hardware_info.supports_mps}")
        logger.info(f"Supports CUDA: {hardware_info.supports_cuda}")

        # Enhanced user-facing device status display
        device_icon = (
            "⚡"
            if hardware_info.device_type == DeviceType.MPS
            else "🖥️" if hardware_info.device_type == DeviceType.CUDA else "💻"
        )
        console.print(
            f"[dim]{device_icon} Using: {hardware_info.device_name} "
            f"({hardware_info.memory_gb:.1f}GB memory)[/dim]"
        )

        # Display warnings prominently to users
        if device_status["warnings"]:
            for warning in device_status["warnings"]:
                if warning["level"] == "warning":
                    console.print(f"[yellow]⚠️  {warning['message']}[/yellow]")
                    console.print(f"[dim]   Impact: {warning['impact']}[/dim]")
                    console.print(f"[dim]   Suggestion: {warning['suggestion']}[/dim]")

        # Display performance expectations
        if device_status["performance_notes"]:
            performance_note = device_status["performance_notes"][
                0
            ]  # Show primary note
            console.print(f"[dim]📊 Performance: {performance_note}[/dim]")

        # Initialize scorer with DeviceManager using progress tracking
        logger.info("=== MULTIDIMENSIONAL SCORER INITIALIZATION ===")
        logger.info("Starting MultiDimensionalScorer initialization...")

        # Create progress bar for analyzer initialization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Add main initialization task
            init_task = progress.add_task(
                "[bold green]Initializing analyzers...", total=None
            )

            def progress_callback(message: str) -> None:
                """Progress callback to update the task description."""
                progress.update(init_task, description=f"[bold green]{message}")
                logger.info(f"Progress: {message}")

            scorer = MultiDimensionalScorer(
                config, repo_path, device_manager, progress_callback=progress_callback
            )

        logger.info("MultiDimensionalScorer initialized successfully")

        # Perform analysis
        start_time = time.time()

        results = scorer.analyze_commit(
            commit_hash, progress_callback=progress_callback
        )

        analysis_time = time.time() - start_time
        console.print(f"[green]Analysis completed in {analysis_time:.2f}s[/green]")

        # Display results
        _display_results(results, ctx.obj["verbose"])

        # Report hardware fallbacks to user
        fallback_report = scorer.get_hardware_fallback_report()
        if fallback_report["has_any_fallbacks"]:
            console.print("")  # Add spacing
            console.print("[yellow]⚠️  Hardware Acceleration Issues Detected[/yellow]")
            console.print(f"[dim]{fallback_report['summary_message']}[/dim]")
            if fallback_report["performance_impact"]:
                console.print(
                    f"[dim]Performance impact: {fallback_report['performance_impact']}[/dim]"
                )

            if fallback_report["suggestions"]:
                console.print("[dim]Suggestions:[/dim]")
                for suggestion in fallback_report["suggestions"][
                    :2
                ]:  # Show top 2 suggestions
                    console.print(f"[dim]  • {suggestion}[/dim]")

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
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda"]),
    default="auto",
    help="Hardware device preference for AI model acceleration (auto, cpu, mps, cuda)",
)
@click.pass_context
def compare(
    ctx: click.Context,
    base_commit: str,
    compare_commits: str,
    repo_path: str,
    output: str | None,
    device: str,
) -> None:
    """Compare multiple commits against a base implementation."""
    commits_to_compare = [c.strip() for c in compare_commits.split(",")]

    console.print(
        f"[bold blue]Comparing {len(commits_to_compare)} commits against base: {base_commit}[/bold blue]"
    )

    # Run pre-analysis health check for MPS devices
    should_continue, final_device = _run_pre_analysis_health_check(device, console)
    if not should_continue:
        console.print("[yellow]Comparison aborted by user[/yellow]")
        sys.exit(0)

    # Use the device choice from health check (may have changed to CPU)
    device = final_device

    # Initialize DeviceManager for hardware acceleration
    device_preference = None if device == "auto" else DeviceType(device.lower())
    with console.status("[bold green]Initializing hardware acceleration..."):
        try:
            device_manager = DeviceManager(prefer_device=device_preference)
        except Exception as e:
            console.print(
                f"[red]❌ Hardware acceleration initialization failed: {e}[/red]"
            )
            if "mps" in str(e).lower():
                console.print(
                    "[yellow]⚠️  MPS acceleration unavailable - analysis will be slower[/yellow]"
                )
                console.print(
                    "[dim]   Suggestion: Check PyTorch installation or use --device cpu[/dim]"
                )
            raise

    # Display hardware information with enhanced status reporting
    hardware_info = device_manager.hardware_info
    device_status = device_manager.get_device_status_report()

    device_icon = (
        "⚡"
        if hardware_info.device_type == DeviceType.MPS
        else "🖥️" if hardware_info.device_type == DeviceType.CUDA else "💻"
    )
    console.print(
        f"[dim]{device_icon} Using: {hardware_info.device_name} "
        f"({hardware_info.memory_gb:.1f}GB memory)[/dim]"
    )

    # Display warnings prominently to users
    if device_status["warnings"]:
        for warning in device_status["warnings"]:
            if warning["level"] == "warning":
                console.print(f"[yellow]⚠️  {warning['message']}[/yellow]")
                console.print(f"[dim]   Impact: {warning['impact']}[/dim]")

    # Default configuration for comparison
    config = EnhancedScorerConfig()
    scorer = MultiDimensionalScorer(config, repo_path, device_manager)

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

    # Report hardware fallbacks to user
    fallback_report = scorer.get_hardware_fallback_report()
    if fallback_report["has_any_fallbacks"]:
        console.print("")  # Add spacing
        console.print("[yellow]⚠️  Hardware Acceleration Issues Detected[/yellow]")
        console.print(f"[dim]{fallback_report['summary_message']}[/dim]")
        if fallback_report["performance_impact"]:
            console.print(
                f"[dim]Performance impact: {fallback_report['performance_impact']}[/dim]"
            )

        if fallback_report["suggestions"]:
            console.print("[dim]Suggestions:[/dim]")
            for suggestion in fallback_report["suggestions"][
                :2
            ]:  # Show top 2 suggestions
                console.print(f"[dim]  • {suggestion}[/dim]")

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
