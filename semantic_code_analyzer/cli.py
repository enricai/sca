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
from .training import CodeStyleTrainer, FineTuningConfig

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
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Multi-Dimensional Code Analyzer for comprehensive code quality analysis."""
    # Set up logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


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
@click.option(
    "--pattern-index-commit",
    default="parent",
    help="Git commit to use for building pattern indices (default: 'parent' - the commit before the one being analyzed, or specify commit hash like 'HEAD', 'main', etc.)",
)
@click.option(
    "--enable-regex-analyzers",
    is_flag=True,
    help="Enable regex-based pattern analyzers (architectural, quality, TypeScript, framework) in addition to semantic embeddings. Default uses only embeddings for pure style matching.",
)
@click.option(
    "--fine-tuned-model",
    help="Use a fine-tuned code embedding model (commit hash like 'abc123d' or HuggingFace Hub ID like 'username/model-name')",
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
    pattern_index_commit: str,
    enable_regex_analyzers: bool,
    fine_tuned_model: str | None,
) -> None:
    """Perform multi-dimensional analysis on a commit."""
    console.print(f"[bold blue]Analyzing commit: {commit_hash}[/bold blue]")

    logger.info("=== STARTING COMMIT ANALYSIS ===")
    logger.info(f"Target commit: {commit_hash}")
    logger.info(f"Repository path: {repo_path}")
    logger.info(f"Output file: {output}")
    logger.info(f"Device preference: {device}")
    logger.info(f"Verbose mode: {ctx.obj['verbose']}")

    # Default to embeddings-only mode (disable regex analyzers)
    if not enable_regex_analyzers:
        logger.info("=== EMBEDDINGS-ONLY MODE (DEFAULT) ===")
        console.print(
            "[dim]📊 Using semantic embeddings for analysis (add --enable-regex-analyzers for multi-dimensional mode)[/dim]"
        )

        # Disable all regex analyzers
        disable_architectural = True
        disable_quality = True
        disable_typescript = True
        disable_framework = True

        # Use 100% domain adherence
        architectural_weight = 0.0
        quality_weight = 0.0
        framework_weight = 0.0
        typescript_weight = 0.0
        domain_adherence_weight = 1.0

        logger.info("Using embeddings-only mode (default)")
        logger.info("Domain adherence weight set to 1.0 (100%)")
    else:
        logger.info("=== MULTI-DIMENSIONAL MODE (REGEX ANALYZERS ENABLED) ===")
        console.print("[dim]📊 Using multi-dimensional mode with regex analyzers[/dim]")

    # Calculate TypeScript weight as remainder (if using regex analyzers)
    logger.info("=== WEIGHT CALCULATION ===")
    logger.info(f"Architectural weight: {architectural_weight}")
    logger.info(f"Quality weight: {quality_weight}")
    logger.info(f"Framework weight: {framework_weight}")
    logger.info(f"Domain adherence weight: {domain_adherence_weight}")

    if enable_regex_analyzers:
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

    if enable_regex_analyzers and typescript_weight < 0.05:
        console.print(
            f"[yellow]Warning: TypeScript weight very low ({typescript_weight:.3f})[/yellow]"
        )

    # Display fine-tuned model info if being used
    if fine_tuned_model:
        console.print(f"[dim]🎯 Using fine-tuned model: {fine_tuned_model}[/dim]")
        logger.info(f"Fine-tuned model requested: {fine_tuned_model}")

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
        fine_tuned_model_commit=fine_tuned_model,
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

        # Create progress bar for analyzer initialization and analysis
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Add main task
            analysis_task = progress.add_task(
                "[bold green]Initializing analyzers...", total=None
            )

            def progress_callback(message: str) -> None:
                """Progress callback to update the task description."""
                progress.update(analysis_task, description=f"[bold green]{message}")
                logger.info(f"Progress: {message}")

            # Initialize scorer
            scorer = MultiDimensionalScorer(
                config, repo_path, device_manager, progress_callback=progress_callback
            )

            logger.info("MultiDimensionalScorer initialized successfully")

            # Perform analysis (keep inside Progress context)
            start_time = time.time()

            results = scorer.analyze_commit(
                commit_hash,
                pattern_index_commit=pattern_index_commit,
                progress_callback=progress_callback,
            )

            analysis_time = time.time() - start_time

        # Display completion message after progress bar closes
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
        if ctx.obj["verbose"]:
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
    # Overall score panel with code-focused score
    overall_score = results.get("overall_adherence", 0)
    code_focused_score = results.get("code_focused_score", overall_score)
    score_color = _get_score_color(code_focused_score)

    # Show both scores if they're different
    if abs(overall_score - code_focused_score) > 0.01:
        score_text = (
            f"[{score_color}]Code-Focused Score: {code_focused_score:.3f}[/{score_color}] "
            f"[dim](Overall: {overall_score:.3f})[/dim]"
        )
    else:
        score_text = (
            f"[{score_color}]Overall Adherence: {overall_score:.3f}[/{score_color}]"
        )

    console.print(
        Panel(
            score_text,
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

    # Domain breakdown table
    domain_breakdown = results.get("domain_breakdown", {})
    if domain_breakdown:
        # Import DOMAIN_WEIGHTS for display
        from .scorers.multi_dimensional_scorer import DOMAIN_WEIGHTS

        domain_table = Table(title="Domain Breakdown")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Files", justify="right")
        domain_table.add_column("Avg Score", justify="right")
        domain_table.add_column("Range", justify="right", style="dim")
        domain_table.add_column("Weight", justify="center", style="dim")
        domain_table.add_column("Interpretation")

        # Sort domains: code domains first (backend, frontend, testing, database)
        # then infrastructure, then docs/config
        code_domains = ["backend", "frontend", "testing", "database"]
        other_domains = [d for d in domain_breakdown.keys() if d not in code_domains]

        for domain in code_domains + other_domains:
            if domain in domain_breakdown:
                stats = domain_breakdown[domain]
                avg_score = stats["avg_score"]
                min_score = stats["min_score"]
                max_score = stats["max_score"]
                file_count = stats["file_count"]
                weight = DOMAIN_WEIGHTS.get(domain, 0.5)

                # Color-coded weight indicator
                if weight >= 0.8:
                    weight_icon = "🟢"
                elif weight >= 0.5:
                    weight_icon = "🟡"
                else:
                    weight_icon = "🔴"

                score_color = _get_score_color(avg_score)
                interpretation = _get_score_interpretation(avg_score)

                # Add note for low-weight domains
                if weight < 0.5:
                    interpretation = f"{interpretation} (low weight)"

                domain_table.add_row(
                    domain.title(),
                    str(file_count),
                    f"[{score_color}]{avg_score:.3f}[/{score_color}]",
                    f"{min_score:.2f}-{max_score:.2f}",
                    weight_icon,
                    interpretation,
                )

        console.print("\n")
        console.print(domain_table)

        # Show top files per domain (code domains only)
        console.print("\n[bold]File Details by Domain:[/bold]")

        code_domains = ["backend", "frontend", "testing", "database"]

        for domain in code_domains:
            if domain in domain_breakdown:
                stats = domain_breakdown[domain]
                files = stats.get("files", [])
                scores = stats.get("scores", [])

                if not files:
                    continue

                # Pair files with scores and sort by score (descending)
                file_score_pairs = list(zip(files, scores, strict=True))
                file_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Show top 20 (or all if fewer)
                display_count = min(20, len(file_score_pairs))

                console.print(
                    f"\n  [{domain.title()}] - Top {display_count}/{len(files)} Files"
                )

                for i, (file_path, score) in enumerate(
                    file_score_pairs[:display_count], 1
                ):
                    score_color = _get_score_color(score)
                    # Truncate long paths for readability
                    display_path = (
                        file_path if len(file_path) < 60 else "..." + file_path[-57:]
                    )
                    console.print(
                        f"    {i}. [{score_color}]{score:.3f}[/{score_color}] - {display_path}"
                    )

                # If there are more files and lowest score is concerning, show it
                if (
                    len(file_score_pairs) > display_count
                    and file_score_pairs[-1][1] < 0.5
                ):
                    lowest_file = file_score_pairs[-1][0]
                    lowest_score = file_score_pairs[-1][1]
                    display_lowest = (
                        lowest_file
                        if len(lowest_file) < 50
                        else "..." + lowest_file[-47:]
                    )
                    console.print(
                        f"    [dim]... (lowest: {display_lowest} = {lowest_score:.3f})[/dim]"
                    )

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

    # Actionable feedback - grouped by domain for code review focus
    feedback = results.get("actionable_feedback", [])
    domain_breakdown = results.get("domain_breakdown", {})

    if feedback and domain_breakdown:
        # Helper function to extract domain from file path using domain_breakdown
        def get_file_domain(file_path: str) -> str:
            """Get domain for a file from domain breakdown."""
            for domain, stats in domain_breakdown.items():
                if file_path in stats.get("files", []):
                    return domain
            return "unknown"

        # Show top issues per code domain (skip docs/config)
        code_domains = ["backend", "frontend", "testing", "database"]
        console.print("\n[bold]Top Issues by Domain:[/bold]")

        shown_count = 0
        for domain in code_domains:
            # Get recommendations for this domain
            domain_recs = [
                rec
                for rec in feedback
                if get_file_domain(rec["file"]) == domain
                and rec["severity"]
                in ["error", "warning"]  # Focus on actionable issues
            ]

            if domain_recs and shown_count < 3:  # Show up to 3 domains
                console.print(f"\n  [{domain.title()}]")
                for rec in domain_recs[:2]:  # Top 2 per domain
                    severity_color = _get_severity_color(rec["severity"])
                    console.print(
                        f"    • [{severity_color}]{rec['file']}[/{severity_color}]: {rec['message'][:80]}..."
                    )
                shown_count += 1

        # If no code domain issues, show generic top recommendations
        if shown_count == 0:
            console.print("\n[bold]Top Recommendations:[/bold]")
            for i, rec in enumerate(feedback[:5], 1):  # Show top 5
                severity_color = _get_severity_color(rec["severity"])
                console.print(
                    f"  {i}. [{severity_color}]{rec['severity'].upper()}[/{severity_color}]: {rec['message']}"
                )
                if rec.get("suggested_fix"):
                    console.print(f"     [dim]Fix: {rec['suggested_fix']}[/dim]")
    elif feedback:
        # Fallback to original display if no domain breakdown
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


@cli.command()
@click.argument("commit_hash")
@click.option(
    "--repo-path",
    "-r",
    default=".",
    help="Path to repository (default: current directory)",
)
@click.option(
    "--epochs",
    default=3,
    type=int,
    help="Number of training epochs (default: 3)",
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Training batch size (default: 8)",
)
@click.option(
    "--learning-rate",
    default=5e-5,
    type=float,
    help="Learning rate (default: 5e-5)",
)
@click.option(
    "--max-files",
    default=1000,
    type=int,
    help="Maximum files to use for training (default: 1000)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda"]),
    default="auto",
    help="Hardware device preference (auto, cpu, mps, cuda)",
)
@click.option(
    "--output-name",
    help="Custom name for fine-tuned model (default: commit hash)",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="Push fine-tuned model to HuggingFace Hub (requires authentication)",
)
@click.option(
    "--hub-model-id",
    help="HuggingFace Hub model ID (e.g., 'username/model-name'). Auto-generated if not provided.",
)
@click.option(
    "--private",
    is_flag=True,
    help="Create private repository on HuggingFace Hub (requires --push-to-hub)",
)
@click.pass_context
def fine_tune(
    ctx: click.Context,
    commit_hash: str,
    repo_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_files: int,
    device: str,
    output_name: str | None,
    push_to_hub: bool,
    hub_model_id: str | None,
    private: bool,
) -> None:
    """Fine-tune code embedding model on a specific commit to learn code style patterns.

    This command trains a specialized version of the code embedding model on your codebase
    at the specified commit. The fine-tuned model learns your code style, naming
    conventions, and patterns, which improves Domain Adherence scores when analyzing
    similar code.

    Example:
        sca-analyze fine-tune dbc9a23 --repo-path ~/src/enric/web
    """
    console.print(
        f"[bold blue]Fine-tuning code embedding model on commit: {commit_hash}[/bold blue]"
    )

    logger.info("=== STARTING FINE-TUNING ===")
    logger.info(f"Target commit: {commit_hash}")
    logger.info(f"Repository path: {repo_path}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Max files: {max_files}")
    logger.info(f"Device preference: {device}")

    try:
        # Run pre-training health check
        should_continue, final_device = _run_pre_analysis_health_check(device, console)
        if not should_continue:
            console.print("[yellow]Fine-tuning aborted by user[/yellow]")
            sys.exit(0)

        device = final_device

        # Initialize device manager
        device_preference = None if device == "auto" else DeviceType(device.lower())

        with console.status("[bold green]Initializing hardware acceleration..."):
            device_manager = DeviceManager(prefer_device=device_preference)

        # Display hardware info
        hardware_info = device_manager.hardware_info
        device_icon = (
            "⚡"
            if hardware_info.device_type == DeviceType.MPS
            else "🖥️" if hardware_info.device_type == DeviceType.CUDA else "💻"
        )
        console.print(
            f"[dim]{device_icon} Using: {hardware_info.device_name} "
            f"({hardware_info.memory_gb:.1f}GB memory)[/dim]"
        )

        # Display M3-specific hardware metrics if on M3
        if (
            hardware_info.is_apple_silicon
            and hardware_info.chip_generation
            and hardware_info.chip_generation.value.startswith("m3")
        ):
            try:
                m3_metrics = device_manager.get_m3_performance_metrics()
                console.print("[dim]📊 M3 Hardware Details:[/dim]")
                console.print(
                    f"[dim]  • GPU Cores: {m3_metrics.get('gpu_cores', 'N/A')}[/dim]"
                )
                console.print(
                    f"[dim]  • Neural Engine: {m3_metrics.get('neural_engine_cores', 'N/A')} cores[/dim]"
                )
                console.print(
                    f"[dim]  • Memory Bandwidth: {m3_metrics.get('memory_bandwidth_gbps', 'N/A')} GB/s[/dim]"
                )

                mps_accel = m3_metrics.get("mps_acceleration", {})
                mps_active = mps_accel.get("active", False)
                mps_status = "✅ Active" if mps_active else "❌ Not Active"
                console.print(f"[dim]  • MPS Acceleration: {mps_status}[/dim]")

                if mps_active:
                    console.print(
                        f"[dim]  • Optimized Batch Size: {mps_accel.get('optimized_batch_size', 'N/A')}[/dim]"
                    )
            except Exception as e:
                logger.debug(f"Could not retrieve M3 metrics: {e}")

        console.print("[dim]This will take approximately 30-45 minutes on M3...[/dim]")

        # Create training configuration
        train_config = FineTuningConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_files=max_files,
            device_preference=device,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            hub_private=private,
        )

        # Initialize trainer
        from pathlib import Path

        cache_dir = Path.cwd() / ".sca_cache"

        trainer = CodeStyleTrainer(
            config=train_config,
            repo_path=repo_path,
            cache_dir=cache_dir,
            device_manager=device_manager,
        )

        # Start fine-tuning
        start_time = time.time()

        model_path = trainer.fine_tune_on_commit(commit_hash, output_name)

        training_time = time.time() - start_time

        # Display success message
        console.print(
            f"[green]Fine-tuning completed in {training_time / 60:.1f} minutes![/green]"
        )
        console.print(f"[green]Model saved to: {model_path}[/green]")
        console.print("")

        # Display final hardware utilization report
        console.print("[bold]✅ Training Complete - Hardware Summary:[/bold]")
        console.print(f"[dim]  • Device used: {hardware_info.device_type.value}[/dim]")

        if hardware_info.is_apple_silicon and hardware_info.chip_generation:
            console.print(
                f"[dim]  • Chip: {hardware_info.chip_generation.value.upper()}[/dim]"
            )

        if hardware_info.device_type == DeviceType.MPS:
            console.print(
                "[dim]  • MPS acceleration: ✅ Active throughout training[/dim]"
            )
            avg_batch_speed = (
                len(trainer.training_stats.get("train_losses", [])) * 3 / training_time
            )  # Rough estimate
            if avg_batch_speed > 0:
                console.print(
                    f"[dim]  • Average speed: ~{avg_batch_speed:.1f} epochs/minute[/dim]"
                )
        else:
            console.print(
                f"[dim]  • Device: {hardware_info.device_type.value} (CPU or CUDA)[/dim]"
            )

        console.print("")
        console.print("[bold]To use this fine-tuned model:[/bold]")

        # Show appropriate command based on whether model was pushed to Hub
        if push_to_hub:
            model_id = hub_model_id or f"sca-finetuned-{commit_hash[:7]}"
            console.print(
                f"  sca-analyze analyze <commit> --fine-tuned-model {model_id}"
            )
        else:
            console.print(
                f"  sca-analyze analyze <commit> --fine-tuned-model {commit_hash[:7]}"
            )

    except Exception as e:
        console.print(f"[red]Error during fine-tuning: {e}[/red]")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("file_path")
@click.option(
    "--from-results",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Path to analysis results JSON file (from analyze --output)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed technical analysis including embedding dimensions",
)
@click.pass_context
def explain(
    ctx: click.Context,
    file_path: str,
    from_results: str,
    verbose: bool,
) -> None:
    """Explain embedding divergence for a file using cached analysis results.

    Shows embedding similarity metrics comparing your file to reference patterns.

    Example:
        sca-analyze analyze HEAD --output results.json
        sca-analyze explain src/components/UserProfile.tsx -r results.json
    """
    console.print(f"[bold blue]Explaining analysis for: {file_path}[/bold blue]")

    logger.info("=== STARTING EXPLAIN COMMAND ===")
    logger.info(f"File: {file_path}")
    logger.info(f"Results file: {from_results}")
    logger.info(f"Verbose: {verbose}")

    try:
        # Load results JSON
        with open(from_results, encoding="utf-8") as f:
            results = json.load(f)

        # Find file in results
        file_level_analysis = results.get("file_level_analysis", {})

        if file_path not in file_level_analysis:
            console.print(f"[red]Error: File '{file_path}' not found in results[/red]")
            console.print("\n[dim]Available files:[/dim]")
            for available_file in list(file_level_analysis.keys())[:10]:
                console.print(f"[dim]  - {available_file}[/dim]")
            if len(file_level_analysis) > 10:
                console.print(
                    f"[dim]  ... and {len(file_level_analysis) - 10} more[/dim]"
                )
            sys.exit(1)

        file_data = file_level_analysis[file_path]

        # Check if embedding data is available
        if "embedding_data" not in file_data:
            console.print("[yellow]⚠️  No embedding data found for this file[/yellow]")
            console.print(
                "[dim]This file may not have had similar patterns, or was analyzed "
                "without domain adherence enabled.[/dim]"
            )
            console.print("\n[dim]Showing available analysis data instead:[/dim]")
            _display_file_summary(file_path, file_data)
            sys.exit(0)

        embedding_data = file_data["embedding_data"]

        # Reconstruct embeddings from JSON
        import numpy as np

        query_embedding = np.array(embedding_data["query_embedding"])
        reference_embeddings = [
            np.array(pattern["embedding"])
            for pattern in embedding_data["similar_patterns"]
        ]
        # Perform embedding divergence analysis (only embedding-derived metrics)
        divergence_analysis = _analyze_embedding_divergence(
            query_embedding=query_embedding,
            reference_embeddings=reference_embeddings,
        )

        # Display explanation
        _display_embedding_explanation(
            file_path=file_path,
            file_data=file_data,
            embedding_data=embedding_data,
            divergence_analysis=divergence_analysis,
            verbose=verbose,
        )

    except Exception as e:
        console.print(f"[red]Error explaining file: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _analyze_embedding_divergence(
    query_embedding: Any,
    reference_embeddings: list[Any],
) -> dict[str, Any]:
    """Analyze embedding divergence using only embedding-derived metrics.

    Args:
        query_embedding: NumPy array of query embedding
        reference_embeddings: List of NumPy arrays for reference embeddings

    Returns:
        Embedding-derived divergence analysis (no assumptions or interpretations)
    """
    import numpy as np

    # Calculate divergence score
    if reference_embeddings:
        avg_ref_embedding = np.mean(reference_embeddings, axis=0)
        diff_vector = np.abs(query_embedding - avg_ref_embedding)
        divergence_score = float(np.mean(diff_vector))

        # Top divergent dimensions
        top_n = 20
        top_divergent_dims = np.argsort(diff_vector)[-top_n:][::-1]

        dimension_divergence = [
            {
                "dimension": int(dim),
                "divergence": float(diff_vector[dim]),
                "query_value": float(query_embedding[dim]),
                "ref_value": float(avg_ref_embedding[dim]),
            }
            for dim in top_divergent_dims
        ]

        # Calculate similarity distribution
        similarity_distribution = []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        for ref_emb in reference_embeddings:
            ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-9)
            similarity = float(np.dot(query_norm, ref_norm))
            similarity_distribution.append(similarity)

        avg_similarity = float(np.mean(similarity_distribution))
    else:
        divergence_score = 1.0
        dimension_divergence = []
        similarity_distribution = []
        avg_similarity = 0.0

    return {
        "divergence_score": divergence_score,
        "embedding_analysis": {
            "dimension_divergence": dimension_divergence[:10],
            "similarity_distribution": similarity_distribution,
            "avg_similarity": avg_similarity,
        },
    }


def _display_file_summary(file_path: str, file_data: dict[str, Any]) -> None:
    """Display basic file analysis summary.

    Args:
        file_path: Path to the file
        file_data: File analysis data from results
    """
    scores = file_data.get("scores", {})

    if scores:
        console.print("\n[bold]Analyzer Scores:[/bold]")
        for analyzer, score in scores.items():
            score_color = _get_score_color(score)
            console.print(f"  {analyzer}: [{score_color}]{score:.3f}[/{score_color}]")


def _display_embedding_explanation(
    file_path: str,
    file_data: dict[str, Any],
    embedding_data: dict[str, Any],
    divergence_analysis: dict[str, Any],
    verbose: bool,
) -> None:
    """Display embedding divergence explanation using Rich formatting.

    Args:
        file_path: Path to the file being explained
        file_data: File analysis data from results
        embedding_data: Embedding data (query + references)
        divergence_analysis: Output from divergence analysis
        verbose: Show detailed technical information
    """
    from pathlib import Path

    # Get scores
    scores = file_data.get("scores", {})
    domain_score = scores.get("domain_adherence", 0.0)
    domain = embedding_data.get("domain", "unknown")

    # Header panel
    score_color = _get_score_color(domain_score)
    divergence_score = divergence_analysis.get("divergence_score", 0.0)
    div_level = (
        "HIGH"
        if divergence_score > 0.15
        else "MEDIUM" if divergence_score > 0.08 else "LOW"
    )

    header_text = f"""File: {Path(file_path).name}
Full Path: {file_path}
Domain: {domain}
Score: [{score_color}]{domain_score:.3f}[/{score_color}]
Divergence: {divergence_score:.3f} ({div_level})"""

    console.print(
        Panel(
            header_text,
            title="Embedding Divergence Analysis",
            border_style=score_color,
        )
    )

    # Similar patterns comparison
    similar_patterns = embedding_data.get("similar_patterns", [])
    if similar_patterns:
        console.print("\n[bold]Reference Patterns (Similar Files):[/bold]")
        ref_table = Table(show_header=True)
        ref_table.add_column("File", style="cyan")
        ref_table.add_column("Similarity", justify="right")

        for pattern in similar_patterns[:5]:
            sim_score = pattern["similarity_score"]
            sim_color = _get_score_color(sim_score)
            ref_table.add_row(
                pattern["file_path"],
                f"[{sim_color}]{sim_score:.3f}[/{sim_color}]",
            )

        console.print(ref_table)

    # Embedding analysis
    emb_analysis = divergence_analysis.get("embedding_analysis", {})
    avg_similarity = emb_analysis.get("avg_similarity", 0.0)

    console.print(f"\n[bold]Average Embedding Similarity:[/bold] {avg_similarity:.3f}")
    console.print(
        "[dim]Cosine similarity between your file's embedding and reference patterns[/dim]"
    )

    # Similarity distribution
    sim_dist = emb_analysis.get("similarity_distribution", [])
    if sim_dist:
        console.print("\n[bold]Similarity to Each Reference:[/bold]")
        for i, sim in enumerate(sim_dist, 1):
            sim_color = _get_score_color(sim)
            console.print(f"  {i}. [{sim_color}]{sim:.3f}[/{sim_color}]")

    # Verbose: Embedding dimensions
    if verbose and emb_analysis.get("dimension_divergence"):
        console.print("\n[bold]Top Divergent Embedding Dimensions:[/bold]")
        console.print(
            "[dim]These dimensions differ most from reference patterns:[/dim]\n"
        )

        dim_table = Table(show_header=True)
        dim_table.add_column("Dim", justify="right", style="cyan")
        dim_table.add_column("Query", justify="right")
        dim_table.add_column("Reference", justify="right")
        dim_table.add_column("Divergence", justify="right")

        for dim_data in emb_analysis["dimension_divergence"][:10]:
            dim_table.add_row(
                str(dim_data["dimension"]),
                f"{dim_data['query_value']:.4f}",
                f"{dim_data['ref_value']:.4f}",
                f"{dim_data['divergence']:.4f}",
            )

        console.print(dim_table)

    # Footer
    console.print("\n[bold]Interpretation:[/bold]")
    console.print(
        "[dim]The divergence score and dimension differences show which parts of the "
        "1536-dimensional embedding space differ most from reference patterns.[/dim]"
    )
    console.print(
        "[dim]Higher divergence = more semantically different from established code patterns.[/dim]"
    )
    console.print(
        "\n[dim]To improve: Examine reference files to understand established patterns.[/dim]"
    )


if __name__ == "__main__":
    cli()
