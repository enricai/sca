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

"""Enhanced progress reporting for multi-dimensional code analysis.

This module provides comprehensive progress reporting capabilities that work
with both Rich and tqdm libraries, offering thread-safe, nested progress
tracking for long-running analysis operations.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressLevel(Enum):
    """Progress reporting verbosity levels."""

    MINIMAL = "minimal"  # Only show major phases
    NORMAL = "normal"  # Show phases and file processing
    DETAILED = "detailed"  # Show all progress including sub-operations


class ProgressBackend(Enum):
    """Progress reporting backend options."""

    RICH = "rich"  # Use Rich progress bars (default)
    TQDM = "tqdm"  # Use tqdm progress bars
    AUTO = "auto"  # Auto-select based on environment


@dataclass
class ProgressConfig:
    """Configuration for progress reporting."""

    enabled: bool = True
    level: ProgressLevel = ProgressLevel.NORMAL
    backend: ProgressBackend = ProgressBackend.RICH
    show_time_remaining: bool = True
    show_speed: bool = True
    refresh_per_second: int = 10
    console_width: int | None = None


class AnalysisPhase(Enum):
    """Major phases in the analysis pipeline."""

    INITIALIZATION = "Initialization"
    REPOSITORY_SETUP = "Repository Setup"
    HARDWARE_INIT = "Hardware Initialization"
    ANALYZER_INIT = "Analyzer Initialization"
    FILE_EXTRACTION = "File Extraction"
    PATTERN_BUILDING = "Pattern Index Building"
    ANALYSIS = "Multi-Dimensional Analysis"
    AGGREGATION = "Results Aggregation"
    FINALIZATION = "Finalization"


class ProgressManager:
    """Thread-safe progress manager for multi-dimensional code analysis.

    Provides unified interface for progress reporting using either Rich or tqdm,
    with support for nested progress tracking and phase-based reporting.
    """

    def __init__(self, config: ProgressConfig | None = None):
        """Initialize the progress manager.

        Args:
            config: Progress configuration (uses defaults if None)
        """
        self.config = config or ProgressConfig()
        self.console = Console()
        self._lock = threading.Lock()
        self._active_progress: Progress | None = None
        self._active_tasks: dict[str, TaskID] = {}
        self._phase_start_times: dict[str, float] = {}
        self._nested_callbacks: dict[str, list[Callable[[str], None]]] = {}

    def is_enabled(self) -> bool:
        """Check if progress reporting is enabled."""
        return self.config.enabled

    @contextmanager
    def create_progress_context(
        self, total_phases: int = len(AnalysisPhase)
    ) -> Generator[ProgressTracker]:
        """Create a progress context for the entire analysis pipeline.

        Args:
            total_phases: Total number of phases to track

        Yields:
            ProgressTracker instance for managing progress
        """
        if not self.config.enabled:
            yield DummyProgressTracker()
            return

        with self._lock:
            if self.config.backend == ProgressBackend.RICH:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    (
                        TimeRemainingColumn()
                        if self.config.show_time_remaining
                        else TextColumn("")
                    ),
                    console=self.console,
                    refresh_per_second=self.config.refresh_per_second,
                )
                progress.start()

                # Create main pipeline task
                pipeline_task = progress.add_task(
                    "[bold green]Analysis Pipeline", total=total_phases
                )

                self._active_progress = progress
                self._active_tasks["pipeline"] = pipeline_task

                try:
                    yield RichProgressTracker(self, progress, pipeline_task)
                finally:
                    progress.stop()
                    self._active_progress = None
                    self._active_tasks.clear()
            else:
                # Fallback to tqdm or console output
                yield TqdmProgressTracker(self)

    def create_nested_callback(
        self, parent_phase: str, description_prefix: str = ""
    ) -> Callable[[str], None]:
        """Create a nested progress callback for sub-operations.

        Args:
            parent_phase: Parent phase name for nesting
            description_prefix: Optional prefix for nested descriptions

        Returns:
            Progress callback function
        """
        if not self.config.enabled:
            return lambda msg: None

        def nested_callback(message: str) -> None:
            """Nested progress callback that updates parent context."""
            with self._lock:
                if self._active_progress and "pipeline" in self._active_tasks:
                    full_message = (
                        f"{description_prefix}{message}"
                        if description_prefix
                        else message
                    )
                    # Update the current phase task with nested message
                    if parent_phase in self._active_tasks:
                        self._active_progress.update(
                            self._active_tasks[parent_phase],
                            description=f"[bold blue]{parent_phase}:[/bold blue] {full_message}",
                        )
                    else:
                        # Update pipeline task if no specific phase task
                        self._active_progress.update(
                            self._active_tasks["pipeline"],
                            description=f"[bold green]{parent_phase}:[/bold green] {full_message}",
                        )

        return nested_callback


class ProgressTracker:
    """Abstract base class for progress tracking implementations."""

    def start_phase(self, phase: AnalysisPhase, description: str | None = None) -> None:
        """Start a new analysis phase."""
        pass

    def update_phase(self, phase: AnalysisPhase, message: str) -> None:
        """Update the current phase with a message."""
        pass

    def complete_phase(self, phase: AnalysisPhase) -> None:
        """Mark a phase as completed."""
        pass

    def create_file_progress(
        self, total_files: int, description: str = "Processing files"
    ) -> FileProgressContext:
        """Create a file processing progress context."""
        return DummyFileProgressContext()

    def create_heartbeat_context(
        self,
        phase: AnalysisPhase,
        base_message: str,
        heartbeat_interval: float = 5.0,
    ) -> HeartbeatProgressContext:
        """Create a heartbeat progress context for long-running operations."""
        return HeartbeatProgressContext(self, phase, base_message, heartbeat_interval)

    def create_download_context(
        self, phase: AnalysisPhase, model_name: str
    ) -> DownloadProgressContext:
        """Create a download progress context for model downloads."""
        return DownloadProgressContext(self, phase, model_name)

    def log_message(self, message: str, level: str = "info") -> None:
        """Log a message alongside progress."""
        pass


class RichProgressTracker(ProgressTracker):
    """Rich-based progress tracker implementation."""

    def __init__(
        self, manager: ProgressManager, progress: Progress, pipeline_task: TaskID
    ):
        """Initialize Rich progress tracker.

        Args:
            manager: Parent progress manager
            progress: Rich Progress instance
            pipeline_task: Main pipeline task ID
        """
        self.manager = manager
        self.progress = progress
        self.pipeline_task = pipeline_task
        self.current_phase: AnalysisPhase | None = None
        self.phase_task: TaskID | None = None

    def start_phase(self, phase: AnalysisPhase, description: str | None = None) -> None:
        """Start a new analysis phase with Rich progress."""
        self.current_phase = phase
        phase_desc = description or phase.value

        # Complete previous phase task if exists
        if self.phase_task is not None:
            self.progress.update(self.phase_task, completed=True)

        # Create new phase task
        self.phase_task = self.progress.add_task(
            f"[bold blue]{phase_desc}[/bold blue]", total=None
        )

        # Store phase start time
        self.manager._phase_start_times[phase.value] = time.time()

        logger.debug(f"Started phase: {phase.value}")

    def update_phase(self, phase: AnalysisPhase, message: str) -> None:
        """Update phase progress with a message."""
        if self.phase_task is not None:
            self.progress.update(
                self.phase_task,
                description=f"[bold blue]{phase.value}:[/bold blue] {message}",
            )

    def complete_phase(self, phase: AnalysisPhase) -> None:
        """Complete a phase and update pipeline progress."""
        if self.phase_task is not None:
            self.progress.update(self.phase_task, completed=True)

        # Update pipeline progress
        self.progress.advance(self.pipeline_task, 1)

        # Log phase completion time
        if phase.value in self.manager._phase_start_times:
            duration = time.time() - self.manager._phase_start_times[phase.value]
            logger.debug(f"Completed phase {phase.value} in {duration:.2f}s")

    def create_file_progress(
        self, total_files: int, description: str = "Processing files"
    ) -> FileProgressContext:
        """Create file processing progress with Rich."""
        file_task = self.progress.add_task(
            f"[dim]{description}[/dim]", total=total_files
        )
        return RichFileProgressContext(self.progress, file_task)

    def log_message(self, message: str, level: str = "info") -> None:
        """Log message with Rich console."""
        if level == "error":
            self.manager.console.print(f"[red]Error:[/red] {message}")
        elif level == "warning":
            self.manager.console.print(f"[yellow]Warning:[/yellow] {message}")
        else:
            self.manager.console.print(f"[dim]{message}[/dim]")


class TqdmProgressTracker(ProgressTracker):
    """Tqdm-based progress tracker implementation."""

    def __init__(self, manager: ProgressManager):
        """Initialize tqdm progress tracker.

        Args:
            manager: Parent progress manager
        """
        self.manager = manager
        self.current_phase: AnalysisPhase | None = None
        self.phase_progress: tqdm[Any] | None = None

    def start_phase(self, phase: AnalysisPhase, description: str | None = None) -> None:
        """Start phase with tqdm."""
        if self.phase_progress:
            self.phase_progress.close()

        phase_desc = description or phase.value
        logger.info("🔍 %s", phase_desc)
        self.current_phase = phase

    def update_phase(self, phase: AnalysisPhase, message: str) -> None:
        """Update phase with message."""
        logger.info("   %s", message)

    def complete_phase(self, phase: AnalysisPhase) -> None:
        """Complete phase."""
        logger.info("✅ %s completed", phase.value)

    def create_file_progress(
        self, total_files: int, description: str = "Processing files"
    ) -> FileProgressContext:
        """Create file progress with tqdm."""
        return TqdmFileProgressContext(total_files, description)


class DummyProgressTracker(ProgressTracker):
    """No-op progress tracker for when progress is disabled."""

    def log_message(self, message: str, level: str = "info") -> None:
        """Log to logger instead of progress display."""
        getattr(logger, level, logger.info)(message)


class FileProgressContext:
    """Base class for file processing progress contexts."""

    def __enter__(self) -> FileProgressContext:
        """Enter file progress context."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit file progress context."""
        pass

    def update(self, file_path: str, completed: int | None = None) -> None:
        """Update file progress."""
        pass

    def advance(self, amount: int = 1) -> None:
        """Advance progress by amount."""
        pass


class RichFileProgressContext(FileProgressContext):
    """Rich-based file progress context."""

    def __init__(self, progress: Progress, task_id: TaskID):
        """Initialize Rich file progress context.

        Args:
            progress: Rich Progress instance
            task_id: Task ID for this file progress
        """
        self.progress = progress
        self.task_id = task_id

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Clean up Rich file progress."""
        self.progress.update(self.task_id, completed=True)

    def update(self, file_path: str, completed: int | None = None) -> None:
        """Update file progress with current file."""
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        self.progress.update(
            self.task_id,
            description=f"[dim]Processing:[/dim] {file_name}",
            completed=completed,
        )

    def advance(self, amount: int = 1) -> None:
        """Advance Rich file progress."""
        self.progress.advance(self.task_id, amount)


class TqdmFileProgressContext(FileProgressContext):
    """Tqdm-based file progress context."""

    def __init__(self, total_files: int, description: str):
        """Initialize tqdm file progress context.

        Args:
            total_files: Total number of files to process
            description: Description for the progress bar
        """
        self.progress_bar = tqdm(
            total=total_files, desc=description, unit="files", ncols=100
        )

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Clean up tqdm progress bar."""
        self.progress_bar.close()

    def update(self, file_path: str, completed: int | None = None) -> None:
        """Update tqdm progress with current file."""
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        self.progress_bar.set_postfix_str(f"Processing: {file_name}")
        if completed is not None:
            self.progress_bar.n = completed
            self.progress_bar.refresh()

    def advance(self, amount: int = 1) -> None:
        """Advance tqdm progress."""
        self.progress_bar.update(amount)


class DummyFileProgressContext(FileProgressContext):
    """No-op file progress context."""

    pass


class HeartbeatProgressContext:
    """Context manager for long-running operations that need periodic heartbeat updates.

    This is used for operations like model downloads where we can't show percentage
    but want to indicate the system is still working.
    """

    def __init__(
        self,
        progress_tracker: ProgressTracker,
        phase: AnalysisPhase,
        base_message: str,
        heartbeat_interval: float = 5.0,
    ):
        """Initialize heartbeat progress context.

        Args:
            progress_tracker: Progress tracker to update
            phase: Analysis phase being updated
            base_message: Base message to show
            heartbeat_interval: Interval between heartbeat updates in seconds
        """
        self.progress_tracker = progress_tracker
        self.phase = phase
        self.base_message = base_message
        self.heartbeat_interval = heartbeat_interval
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._start_time = time.time()

    def __enter__(self) -> HeartbeatProgressContext:
        """Start heartbeat updates."""
        self._start_heartbeat()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Stop heartbeat updates."""
        self._stop_heartbeat()

    def _start_heartbeat(self) -> None:
        """Start the heartbeat thread."""
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker, daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)

    def _heartbeat_worker(self) -> None:
        """Worker function for heartbeat updates."""
        count = 0
        while not self._stop_event.wait(self.heartbeat_interval):
            count += 1
            elapsed = time.time() - self._start_time

            # Create animated indicator
            indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            indicator = indicators[count % len(indicators)]

            # Show elapsed time
            elapsed_str = f"{elapsed:.0f}s"
            message = f"{self.base_message} {indicator} ({elapsed_str} elapsed)"

            self.progress_tracker.update_phase(self.phase, message)

    def update_message(self, new_message: str) -> None:
        """Update the base message during the operation."""
        self.base_message = new_message


class DownloadProgressContext:
    """Context manager for tracking model download progress.

    Hooks into HuggingFace transformers download progress when possible.
    """

    def __init__(
        self,
        progress_tracker: ProgressTracker,
        phase: AnalysisPhase,
        model_name: str,
    ):
        """Initialize download progress context.

        Args:
            progress_tracker: Progress tracker to update
            phase: Analysis phase being updated
            model_name: Name of model being downloaded
        """
        self.progress_tracker = progress_tracker
        self.phase = phase
        self.model_name = model_name
        self._download_started = False
        self._total_size: int | None = None
        self._downloaded_size = 0

    def __enter__(self) -> DownloadProgressContext:
        """Start download progress tracking."""
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Complete download progress tracking."""
        if self._download_started:
            self.progress_tracker.update_phase(
                self.phase, f"Download completed for {self.model_name}"
            )

    def start_download(self, total_size: int | None = None) -> None:
        """Signal that download has started.

        Args:
            total_size: Total download size in bytes if known
        """
        self._download_started = True
        self._total_size = total_size

        if total_size:
            size_mb = total_size / (1024 * 1024)
            self.progress_tracker.update_phase(
                self.phase, f"Downloading {self.model_name} ({size_mb:.1f}MB)..."
            )
        else:
            self.progress_tracker.update_phase(
                self.phase, f"Downloading {self.model_name}..."
            )

    def update_progress(self, downloaded_size: int) -> None:
        """Update download progress.

        Args:
            downloaded_size: Number of bytes downloaded so far
        """
        self._downloaded_size = downloaded_size

        if self._total_size:
            percent = (downloaded_size / self._total_size) * 100
            downloaded_mb = downloaded_size / (1024 * 1024)
            total_mb = self._total_size / (1024 * 1024)

            self.progress_tracker.update_phase(
                self.phase,
                f"Downloading {self.model_name} {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f}MB)",
            )
        else:
            downloaded_mb = downloaded_size / (1024 * 1024)
            self.progress_tracker.update_phase(
                self.phase, f"Downloaded {downloaded_mb:.1f}MB of {self.model_name}..."
            )


# Convenience functions for common use cases
def create_default_progress_manager() -> ProgressManager:
    """Create a progress manager with default configuration."""
    return ProgressManager()


def create_minimal_progress_manager() -> ProgressManager:
    """Create a progress manager with minimal output."""
    config = ProgressConfig(level=ProgressLevel.MINIMAL)
    return ProgressManager(config)


def create_detailed_progress_manager() -> ProgressManager:
    """Create a progress manager with detailed output."""
    config = ProgressConfig(level=ProgressLevel.DETAILED)
    return ProgressManager(config)
