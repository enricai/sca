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

"""Pattern indexing system using code embeddings and FAISS similarity search.

This module provides functionality to build domain-specific code pattern indices
using state-of-the-art code embeddings for fast similarity search and pattern
matching in domain-aware adherence analysis.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import faiss  # type: ignore[import-untyped]
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..hardware.device_manager import DeviceManager, DeviceType
from ..hardware.exceptions import FallbackError, ModelLoadingError
from ..parsing import FunctionExtractor
from ..parsing.data_compressor import DataCompressionConfig

logger = logging.getLogger(__name__)

# Type variable for retry mechanism
T = TypeVar("T")

# Global lock to prevent concurrent model loading
_model_loading_lock = threading.Lock()

# Suppress warnings from transformers and torch
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


@dataclass
class SimilarityMatch:
    """Represents a similarity match from pattern search."""

    file_path: str
    similarity_score: float  # 0.0 to 1.0
    code_snippet: str
    domain: str
    context: dict[str, Any]


@dataclass
class PatternIndex:
    """Container for a domain-specific pattern index."""

    domain: str
    index: faiss.IndexFlatIP
    file_paths: list[str]
    code_snippets: list[str]
    embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]]
    metadata: dict[str, Any]
    function_names: list[str] | None = None  # Function names for each chunk
    is_methods: list[bool] | None = None  # Whether each chunk is a method


class PatternIndexer:
    """Build and manage domain-specific FAISS similarity indices using code embeddings.

    This class provides functionality to:
    - Extract code embeddings using Qodo-Embed
    - Build FAISS indices for fast similarity search
    - Search for similar patterns within domain contexts
    - Cache embeddings for performance optimization
    """

    def __init__(
        self,
        model_name: str = "Qodo/Qodo-Embed-1-1.5B",
        model_revision: str = "main",  # pragma: allowlist secret
        cache_dir: str | None = None,
        device_manager: DeviceManager | None = None,
        enable_optimizations: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        fine_tuned_model_commit: str | None = None,
        compression_config: DataCompressionConfig | None = None,
    ):
        """Initialize the PatternIndexer with Qodo-Embed model.

        Args:
            model_name: Name of the code embedding model to use
            model_revision: Model revision/commit hash for reproducible downloads (security)
            cache_dir: Directory for caching models and indices
            device_manager: Device manager for hardware acceleration (auto-created if None)
            enable_optimizations: Enable hardware-specific optimizations
            progress_callback: Optional callback to report initialization progress
            fine_tuned_model_commit: Commit hash of fine-tuned model to load (optional)
            compression_config: Configuration for data compression (uses defaults if None)
        """
        logger.info("=== PATTERN INDEXER INIT START ===")
        logger.info("Starting PatternIndexer initialization")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Cache dir: {cache_dir}")
        logger.info(f"Enable optimizations: {enable_optimizations}")

        # Store progress callback for reporting
        self.progress_callback = progress_callback

        # Helper function to report progress
        def report_progress(message: str) -> None:
            """Report progress if callback is available."""
            if self.progress_callback:
                self.progress_callback(message)

        self.model_name = model_name
        self.model_revision = model_revision
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".sca_cache"
        self.fine_tuned_model_commit = fine_tuned_model_commit
        self.is_fine_tuned = fine_tuned_model_commit is not None
        logger.info(f"Using cache directory: {self.cache_dir}")

        if self.is_fine_tuned:
            logger.info(f"Fine-tuned model requested: {fine_tuned_model_commit}")

        report_progress("Setting up cache directory...")
        try:
            logger.info("Creating cache directory...")
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Cache directory created successfully")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise

        # Initialize device manager for hardware acceleration
        report_progress("Initializing hardware acceleration...")
        if device_manager is None:
            logger.info("Creating new DeviceManager for PatternIndexer...")
            self.device_manager = DeviceManager()
            logger.info("New DeviceManager created for PatternIndexer")
        else:
            logger.info("Using provided DeviceManager")
            self.device_manager = device_manager

        self.enable_optimizations = enable_optimizations

        # Initialize fallback event tracking for user reporting
        self.fallback_events: dict[str, Any] = {
            "mps_device_failures": 0,
            "mps_inference_failures": 0,
            "original_device": None,
            "current_device": None,
            "performance_degradation_factor": 1.0,
            "failure_reasons": [],
        }

        # Get hardware-optimized settings
        report_progress("Configuring hardware optimizations...")
        logger.info("Getting hardware optimization settings...")
        try:
            self.optimization_settings = (
                self.device_manager.optimize_for_model_loading()
            )
            logger.info(f"Optimization settings: {self.optimization_settings}")
        except Exception as e:
            logger.error(f"Failed to get optimization settings: {e}")
            raise

        try:
            self.device_recommendations = (
                self.device_manager.get_device_recommendations()
            )
            logger.info(f"Device recommendations: {self.device_recommendations}")
        except Exception as e:
            logger.error(f"Failed to get device recommendations: {e}")
            raise

        # Validate MPS compatibility before attempting to use it
        report_progress("Validating device compatibility...")
        logger.info("Validating MPS compatibility...")
        try:
            self._validate_mps_compatibility()
            logger.info("MPS compatibility validation completed")
        except Exception as e:
            logger.error(f"MPS compatibility validation failed: {e}")
            raise

        # Initialize model and tokenizer with hardware optimization
        hardware_info = self.device_manager.hardware_info
        logger.info("=== MODEL LOADING START ===")
        logger.info(
            f"Loading code embedding model: {model_name} on {hardware_info.device_name}"
        )
        logger.info(f"Target device type: {hardware_info.device_type}")
        logger.info(f"Available memory: {hardware_info.memory_gb:.1f}GB")

        try:
            # Check if we should load a fine-tuned model instead
            if self.is_fine_tuned:
                report_progress(
                    f"Loading fine-tuned model for commit {fine_tuned_model_commit}..."
                )
                logger.info("=== FINE-TUNED MODEL LOADING ===")
                fine_tuned_path = self._get_fine_tuned_model_path(
                    fine_tuned_model_commit
                )

                if not fine_tuned_path.exists():
                    raise ValueError(
                        f"Fine-tuned model not found for commit {fine_tuned_model_commit}. "
                        f"Expected at: {fine_tuned_path}"
                    )

                logger.info(f"Loading fine-tuned model from: {fine_tuned_path}")

                # Load tokenizer from fine-tuned model directory
                # Loading from local trusted fine-tuned directory, not remote Hub
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fine_tuned_path
                )  # nosec B615
                logger.info("Tokenizer loaded from fine-tuned model")

                # Load fine-tuned model
                # Loading from local trusted fine-tuned directory, not remote Hub
                self.model = AutoModel.from_pretrained(
                    fine_tuned_path,
                    low_cpu_mem_usage=self.optimization_settings["low_cpu_mem_usage"],
                )  # nosec B615
                logger.info("Fine-tuned model loaded successfully")

                # Load metadata
                metadata_path = fine_tuned_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, encoding="utf-8") as f:
                        fine_tuned_metadata = json.load(f)
                        logger.info(f"Fine-tuned model metadata: {fine_tuned_metadata}")
                        # fine_tuned_model_commit is guaranteed to be not None here (checked earlier)
                        if fine_tuned_model_commit:
                            self.model_name = (
                                f"fine-tuned-{fine_tuned_model_commit[:7]}"
                            )
            else:
                # Load base code embedding model (original behavior)
                # Load tokenizer with retry mechanism
                report_progress("Loading model tokenizer...")
                logger.info("=== TOKENIZER LOADING ===")
                logger.info("Loading AutoTokenizer...")
                logger.info(f"Tokenizer model: {model_name}")
                logger.info(f"Tokenizer cache dir: {str(self.cache_dir)}")

                self.tokenizer = self._load_with_retry(
                    lambda: AutoTokenizer.from_pretrained(
                        model_name,
                        revision=self.model_revision,
                        cache_dir=str(self.cache_dir),  # nosec B615
                    ),
                    "tokenizer",
                )
                logger.info("Tokenizer loaded successfully!")

                # Load model with hardware-specific optimizations and retry mechanism
                report_progress("Loading embedding model (this is the slow step)...")
                logger.info("=== MODEL LOADING ===")
                logger.info("Preparing model loading parameters...")

                # Note: AutoModel.from_pretrained() doesn't support dtype parameter directly
                # The dtype should be applied after loading
                model_kwargs = {
                    "cache_dir": str(self.cache_dir),
                    "low_cpu_mem_usage": self.optimization_settings[
                        "low_cpu_mem_usage"
                    ],
                    "use_safetensors": True,  # Use safetensors to bypass PyTorch 2.6 requirement
                }
                logger.info(f"Model kwargs: {model_kwargs}")
                logger.info(
                    f"Target dtype (to be applied after loading): {self.optimization_settings['dtype']}"
                )

                logger.info("Loading AutoModel - THIS IS WHERE SEGFAULTS OFTEN OCCUR!")
                logger.info("About to call AutoModel.from_pretrained()...")

                self.model = self._load_with_retry(
                    lambda: AutoModel.from_pretrained(
                        model_name, revision=self.model_revision, **model_kwargs
                    ),  # nosec B615
                    "model",
                )
                logger.info("AutoModel loaded successfully!")

            report_progress("Configuring model settings...")
            logger.info("Setting model to evaluation mode...")
            self.model.eval()
            logger.info("Model set to evaluation mode")

            # Configure device with intelligent selection
            report_progress("Setting up device configuration...")
            logger.info("=== DEVICE CONFIGURATION ===")
            self.device = self.device_manager.torch_device
            logger.info(f"Target device: {self.device}")
            logger.info(f"Device type: {self.device.type}")

            # Track original device for fallback reporting
            self.fallback_events["original_device"] = str(self.device)
            self.fallback_events["current_device"] = str(self.device)

            # Apply M3-specific optimizations
            if self.enable_optimizations:
                report_progress("Applying hardware optimizations...")
                logger.info("Applying hardware-specific optimizations...")
                try:
                    self._apply_hardware_optimizations()
                    logger.info("Hardware optimizations applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply hardware optimizations: {e}")
                    raise

            # Move model to optimized device with MPS segmentation fault protection
            report_progress("Moving model to accelerated device...")
            logger.info("=== MODEL DEVICE MOVEMENT ===")
            logger.info("CRITICAL: About to move model to device - high segfault risk!")
            logger.info(f"Moving model to device: {self.device}")

            try:
                logger.info("Calling self.model.to(self.device)...")
                self.model.to(self.device)
                logger.info("Model moved to device successfully!")

                # Apply dtype if needed (after moving to device)
                target_dtype = self.optimization_settings["dtype"]
                if (
                    target_dtype != torch.float32
                ):  # Only convert if different from default
                    logger.info(f"Converting model to dtype: {target_dtype}")
                    try:
                        self.model = self.model.to(dtype=target_dtype)
                        logger.info("Model dtype conversion completed successfully")
                    except Exception as dtype_error:
                        logger.warning(f"Failed to convert model dtype: {dtype_error}")
                        logger.info("Continuing with default dtype")

                # Test MPS operations to ensure stability
                if self.device.type == "mps":
                    logger.info("=== MPS OPERATIONS VALIDATION ===")
                    logger.info("Device is MPS - running additional validation...")
                    self._validate_mps_operations()
                    logger.info("MPS operations validation passed!")

            except (RuntimeError, OSError, SystemError) as e:
                logger.error(
                    f"CRITICAL: Device movement failed with error: {e}. "
                    "This may indicate MPS segmentation fault issues."
                )
                logger.exception("Full traceback for device movement failure:")
                if self.device.type == "mps":
                    logger.info("Attempting CPU fallback due to MPS device failure")
                    # Track device movement failure
                    self.fallback_events["mps_device_failures"] = (
                        int(self.fallback_events["mps_device_failures"]) + 1
                    )
                    failure_reasons = list(self.fallback_events["failure_reasons"])
                    failure_reasons.append(f"Device movement: {str(e)[:100]}")
                    self.fallback_events["failure_reasons"] = failure_reasons
                    self._fallback_to_cpu()
                else:
                    raise

            # Log hardware configuration
            logger.info("=== FINAL CONFIGURATION ===")
            memory_info = self.device_manager.get_memory_info()
            logger.info(
                f"Model loaded on {hardware_info.device_name} "
                f"({memory_info['system_memory_gb']:.1f}GB memory available)"
            )
            logger.info(f"Final model device: {next(self.model.parameters()).device}")
            logger.info("=== MODEL LOADING COMPLETE ===")
        except Exception as e:
            model_error = ModelLoadingError(
                f"Failed to load code embedding model on {hardware_info.device_name}: {e}",
                device_type=hardware_info.device_type.value,
                model_name=self.model_name,
            )
            logger.error(str(model_error))

            # Attempt fallback to CPU if enabled
            if (
                self.device_manager.device_config.fallback_enabled
                and hardware_info.device_type != DeviceType.CPU
            ):
                logger.warning("Attempting CPU fallback for model loading")
                try:
                    self._fallback_to_cpu()
                    logger.info("Successfully loaded model on CPU fallback")
                except Exception as fallback_error:
                    logger.error(f"CPU fallback also failed: {fallback_error}")
                    raise FallbackError(
                        f"Could not initialize code embedding model on any device: {e}",
                        original_error=model_error,
                        fallback_device="cpu",
                    ) from model_error
            else:
                raise model_error from None

        # Storage for domain indices
        self.domain_indices: dict[str, PatternIndex] = {}
        self.embedding_cache: dict[str, np.ndarray[Any, np.dtype[np.floating[Any]]]] = (
            {}
        )

        # Store compression configuration
        self.compression_config = compression_config or DataCompressionConfig()

        # Initialize function extractor for AST-based chunking with compression support
        report_progress("Initializing function extractor...")
        self.function_extractor = FunctionExtractor(
            compression_config=self.compression_config
        )
        logger.info("FunctionExtractor initialized with data compression support")

        # Performance metrics tracking
        self._performance_metrics = {
            "total_embeddings_generated": 0,
            "average_embedding_time": 0.0,
            "cache_hit_rate": 0.0,
            "memory_usage_peak": 0.0,
        }

        report_progress("Pattern indexer ready!")
        logger.info("=== PATTERN INDEXER INIT COMPLETE ===")

    def _load_with_retry(self, load_func: Callable[[], T], component_name: str) -> T:
        """Load model components with retry mechanism and concurrent loading protection.

        Args:
            load_func: Function to load the component
            component_name: Name of component for logging

        Returns:
            Loaded component

        Raises:
            ModelLoadingError: If loading fails after all retries
        """
        logger.info(f"=== LOADING {component_name.upper()} WITH RETRY ===")
        logger.info(f"Starting {component_name} loading with retry mechanism")
        # Check if model loading is disabled for testing (but not when models are mocked)
        if os.getenv("SCA_DISABLE_MODEL_LOADING", "0") == "1":
            # Check if we're in a test environment with mocked models
            # Allow mocked tests to proceed by trying the load_func first
            try:
                # Try to load - if it's a mock, it should work
                result = load_func()
                # If it's a real mock object, return it
                if (
                    hasattr(result, "_mock_name")
                    or str(type(result)).startswith("<MagicMock")
                    or str(type(result)).startswith("<Mock")
                ):
                    return result
            except Exception as e:
                logger.debug(
                    "Mock detection failed for %s, proceeding with disabled loading: %s",
                    component_name,
                    str(e),
                )

            # If we get here, it's not a mock, so disable loading
            raise ModelLoadingError(
                f"Model loading disabled for testing. "
                f"Set SCA_DISABLE_MODEL_LOADING=0 to enable {component_name} loading."
            )

        max_retries = 3
        base_delay = 2.0

        logger.info(f"Acquiring model loading lock for {component_name}...")
        with _model_loading_lock:
            logger.info(f"Model loading lock acquired for {component_name}")

            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Loading {component_name} (attempt {attempt + 1}/{max_retries})"
                    )

                    # Add small random delay to prevent thundering herd
                    if attempt > 0:
                        secure_random = secrets.SystemRandom()
                        delay = base_delay * (
                            2 ** (attempt - 1)
                        ) + secure_random.uniform(0, 1)
                        logger.info(f"Waiting {delay:.1f}s before retry...")
                        time.sleep(delay)

                    logger.info(
                        f"CRITICAL: About to execute load_func() for {component_name}"
                    )
                    logger.info("This is where segmentation faults typically occur!")

                    result = load_func()

                    logger.info(f"SUCCESS: {component_name} loaded successfully!")
                    logger.info(f"Loaded {component_name} type: {type(result)}")
                    return result

                except Exception as e:
                    logger.error(
                        f"FAILED: {component_name} loading attempt {attempt + 1} failed: {e}"
                    )
                    logger.exception(
                        f"Full traceback for {component_name} loading failure:"
                    )

                    if attempt == max_retries - 1:
                        logger.error(
                            f"CRITICAL: All {max_retries} attempts to load {component_name} failed"
                        )
                        raise ModelLoadingError(
                            f"Failed to load {component_name} after {max_retries} attempts. "
                            f"Last error: {e}"
                        ) from e

                    # Force garbage collection to free memory
                    logger.info("Performing garbage collection before retry...")
                    import gc

                    gc.collect()
                    if torch.backends.mps.is_available():
                        logger.info("Emptying MPS cache...")
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        logger.info("Emptying CUDA cache...")
                        torch.cuda.empty_cache()
                    logger.info("Memory cleanup completed")

        # This should never be reached, but for type safety
        raise ModelLoadingError(f"Unexpected error loading {component_name}")

    def _apply_hardware_optimizations(self) -> None:
        """Apply hardware-specific optimizations based on detected device."""
        hardware_info = self.device_manager.hardware_info

        if hardware_info.device_type == DeviceType.MPS:
            self._apply_mps_optimizations()
        elif hardware_info.device_type == DeviceType.CUDA:
            self._apply_cuda_optimizations()
        elif hardware_info.device_type == DeviceType.ROCM:
            self._apply_rocm_optimizations()
        elif hardware_info.device_type == DeviceType.XPU:
            self._apply_xpu_optimizations()

        # Memory optimizations for unified memory architectures
        if hardware_info.unified_memory:
            self._apply_unified_memory_optimizations()

        logger.info(f"Applied {hardware_info.device_type.value} optimizations")

    def _apply_mps_optimizations(self) -> None:
        """Apply Metal Performance Shaders optimizations for Apple Silicon."""
        try:
            # Enable MPS memory fraction if available
            if hasattr(torch.mps, "set_per_process_memory_fraction"):
                max_fraction = self.device_manager.device_config.max_memory_fraction
                torch.mps.set_per_process_memory_fraction(max_fraction)

            # Configure for inference
            if self.device_manager.device_config.optimize_for_inference:
                if hasattr(torch.backends.mps, "allow_tf32"):
                    torch.backends.mps.allow_tf32 = True

            logger.debug("Applied MPS-specific optimizations")

        except Exception as e:
            logger.warning(f"Could not apply all MPS optimizations: {e}")

    def _apply_cuda_optimizations(self) -> None:
        """Apply CUDA-specific optimizations."""
        try:
            # Enable TensorFloat-32 for better performance on compatible hardware
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable memory pooling
            if self.device_manager.device_config.enable_memory_pooling:
                torch.cuda.empty_cache()

            logger.debug("Applied CUDA-specific optimizations")

        except Exception as e:
            logger.warning(f"Could not apply all CUDA optimizations: {e}")

    def _apply_rocm_optimizations(self) -> None:
        """Apply ROCm-specific optimizations (AMD GPUs)."""
        try:
            # ROCm uses CUDA API compatibility (HIP), so apply similar optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable memory pooling
            if self.device_manager.device_config.enable_memory_pooling:
                torch.cuda.empty_cache()

            logger.debug("Applied ROCm-specific optimizations")

        except Exception as e:
            logger.warning(f"Could not apply all ROCm optimizations: {e}")

    def _apply_xpu_optimizations(self) -> None:
        """Apply Intel XPU-specific optimizations."""
        try:
            # Enable memory pooling if available
            if self.device_manager.device_config.enable_memory_pooling:
                if hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
                    torch.xpu.empty_cache()

            logger.debug("Applied Intel XPU-specific optimizations")

        except Exception as e:
            logger.warning(f"Could not apply all XPU optimizations: {e}")

    def _apply_unified_memory_optimizations(self) -> None:
        """Apply optimizations for unified memory architectures (Apple Silicon)."""
        # Adjust batch processing for unified memory efficiency
        recommended_batch = self.device_manager.device_config.batch_size

        logger.debug(
            f"Configured for unified memory with batch size: {recommended_batch}"
        )

    def _validate_mps_compatibility(self) -> None:
        """Validate MPS compatibility and warn about potential issues."""
        hardware_info = self.device_manager.hardware_info

        if hardware_info.device_type == DeviceType.MPS:
            try:
                # Test basic MPS operations to detect potential issues
                test_tensor = torch.randn(2, 2, device="mps")
                torch.mm(test_tensor, test_tensor)  # Test operation
                torch.mps.synchronize()

                logger.debug("MPS compatibility validated successfully")
            except Exception as e:
                logger.warning(
                    f"MPS compatibility issues detected: {e}. "
                    "CPU fallback will be used if model loading fails."
                )

    def _validate_mps_operations(self) -> None:
        """Validate MPS operations with model to detect segmentation fault issues."""
        try:
            # Test model inference with MPS to catch segmentation faults early
            dummy_input = torch.randint(0, 1000, (1, 512), device=self.device)

            with torch.no_grad():
                # Test model forward pass
                _ = self.model(dummy_input)

            # Synchronize to ensure operations complete
            torch.mps.synchronize()

            # Memory management test
            torch.mps.empty_cache()

            logger.debug("MPS model operations validated successfully")

        except (RuntimeError, OSError, SystemError, MemoryError) as e:
            logger.error(f"MPS operation validation failed: {e}")
            raise RuntimeError(
                f"MPS segmentation fault detected during model operations: {e}"
            ) from e

    def _fallback_to_cpu(self) -> None:
        """Fallback to CPU when primary device fails."""
        logger.info("Falling back to CPU device")

        try:
            # Clear any device memory before fallback
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug(f"Error during device memory cleanup: {e}")

        # Reload model on CPU
        model_kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": torch.float32,  # Use full precision for CPU
            "low_cpu_mem_usage": True,
        }

        self.model = AutoModel.from_pretrained(
            self.model_name, revision=self.model_revision, **model_kwargs
        )  # nosec B615
        self.model.eval()

        # Update device manager to CPU
        self.device_manager = DeviceManager(prefer_device=DeviceType.CPU)
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Track fallback event and performance impact
        self.fallback_events["current_device"] = str(self.device)
        self.fallback_events["performance_degradation_factor"] = (
            3.5  # Estimate 3.5x slower on CPU
        )
        logger.info(
            f"Fallback complete: {self.fallback_events['original_device']} -> {self.fallback_events['current_device']}"
        )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the pattern indexer.

        Returns:
            Dictionary with performance metrics and hardware utilization
        """
        hardware_info = self.device_manager.hardware_info
        memory_info = self.device_manager.get_memory_info()

        metrics = {
            "device_info": {
                "device_type": hardware_info.device_type.value,
                "device_name": hardware_info.device_name,
                "is_apple_silicon": hardware_info.is_apple_silicon,
                "chip_generation": (
                    hardware_info.chip_generation.value
                    if hardware_info.chip_generation
                    else None
                ),
            },
            "memory_usage": memory_info,
            "performance": self._performance_metrics,
            "configuration": {
                "batch_size": self.device_manager.device_config.batch_size,
                "mixed_precision": self.device_manager.device_config.enable_mixed_precision,
                "memory_pooling": self.device_manager.device_config.enable_memory_pooling,
            },
            "cache_statistics": {
                "domain_indices": len(self.domain_indices),
                "embedding_cache_size": len(self.embedding_cache),
            },
        }

        return metrics

    def get_fallback_report(self) -> dict[str, Any]:
        """Get fallback events report for user notification.

        Returns:
            Dictionary with fallback statistics and user-facing information.
        """
        device_failures = int(self.fallback_events["mps_device_failures"])
        inference_failures = int(self.fallback_events["mps_inference_failures"])
        total_failures = device_failures + inference_failures

        report = {
            "has_fallbacks": total_failures > 0,
            "total_failures": total_failures,
            "device_failures": device_failures,
            "inference_failures": inference_failures,
            "original_device": self.fallback_events["original_device"],
            "current_device": self.fallback_events["current_device"],
            "performance_impact": None,
            "user_message": None,
            "suggestions": [],
        }

        if report["has_fallbacks"]:
            degradation = self.fallback_events["performance_degradation_factor"]
            report["performance_impact"] = f"{degradation:.1f}x slower than expected"
            report["user_message"] = (
                f"Hardware acceleration failed ({total_failures} events) - "
                f"using CPU fallback (~{degradation:.1f}x slower)"
            )

            if "mps" in str(self.fallback_events["original_device"]).lower():
                report["suggestions"] = [
                    "Update PyTorch: pip install --upgrade torch torchvision",
                    "Check macOS compatibility with your PyTorch version",
                    "Use --device cpu flag to avoid MPS issues",
                    "Monitor system memory usage during analysis",
                ]

        return report

    def benchmark_embedding_performance(
        self, test_code: str = "def hello(): pass", iterations: int = 10
    ) -> dict[str, Any]:
        """Benchmark embedding generation performance on current hardware.

        Args:
            test_code: Code snippet to use for benchmarking
            iterations: Number of iterations to run

        Returns:
            Dictionary with benchmark results
        """
        import time

        logger.info(f"Running embedding benchmark with {iterations} iterations")

        # Warm up
        for _ in range(3):
            self._extract_code_embeddings(test_code)

        # Synchronize device before timing
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":  # Includes ROCm (uses CUDA API)
            torch.cuda.synchronize()
        elif self.device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

        # Run benchmark
        start_time = time.time()
        for _ in range(iterations):
            self._extract_code_embeddings(test_code)

        # Synchronize device after timing
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":  # Includes ROCm (uses CUDA API)
            torch.cuda.synchronize()
        elif self.device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

        elapsed_time = time.time() - start_time
        avg_time_per_embedding = elapsed_time / iterations

        benchmark_results = {
            "device": self.device.type,
            "iterations": iterations,
            "total_time_seconds": elapsed_time,
            "average_time_per_embedding": avg_time_per_embedding,
            "embeddings_per_second": 1.0 / avg_time_per_embedding,
            "hardware_info": self.device_manager.hardware_info.device_name,
            "test_code_length": len(test_code),
        }

        logger.info(
            f"Benchmark completed: {benchmark_results['embeddings_per_second']:.2f} embeddings/sec "
            f"on {self.device.type}"
        )

        return benchmark_results

    def _numpy_to_json_serializable(
        self, embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> dict[str, Any]:
        """Convert NumPy array to JSON-serializable format.

        Args:
            embeddings: NumPy array to convert

        Returns:
            Dictionary with shape, dtype, and data as nested lists
        """
        return {
            "shape": embeddings.shape,
            "dtype": str(embeddings.dtype),
            "data": embeddings.tolist(),
        }

    def _json_to_numpy_array(
        self, json_data: dict[str, Any]
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Convert JSON data back to NumPy array.

        Args:
            json_data: Dictionary with shape, dtype, and data

        Returns:
            Reconstructed NumPy array
        """
        array = np.array(json_data["data"], dtype=json_data["dtype"])
        return array.reshape(json_data["shape"])

    def build_domain_index(
        self,
        domain: str,
        codebase_files: dict[str, str],
        max_files: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """Build a FAISS similarity index for a specific domain.

        Args:
            domain: Domain name (e.g., 'frontend', 'backend')
            codebase_files: Dictionary mapping file paths to their content
            max_files: Maximum number of files to process (for performance)
            chunk_size: Maximum token chunk size for processing (auto-optimized if None)
        """
        # Use hardware-optimized chunk size if not specified
        if chunk_size is None:
            if "m3_optimizations" in self.device_recommendations:
                chunk_size = self.device_recommendations["m3_optimizations"][
                    "recommended_chunk_size"
                ]
            else:
                chunk_size = 2048  # Default fallback - allow larger functions

        logger.info(
            f"Building pattern index for domain: {domain} "
            f"(chunk_size: {chunk_size}, device: {self.device_manager.hardware_info.device_name})"
        )

        if not codebase_files:
            logger.warning(f"No files provided for domain {domain}")
            return

        # Limit files if specified
        files_to_process = (
            dict(list(codebase_files.items())[:max_files])
            if max_files
            else codebase_files
        )

        # Extract embeddings for all files using function-level chunking
        embeddings_list = []
        file_paths = []
        code_snippets = []
        function_names = []
        is_methods = []

        for file_path, content in files_to_process.items():
            try:
                # Extract functions using tree-sitter
                function_chunks = self.function_extractor.extract_functions(
                    file_path, content
                )

                # Get language name for this file
                file_ext = Path(file_path).suffix.lower()
                lang_config = (
                    self.function_extractor.language_registry.get_language_for_extension(
                        file_ext
                    )
                    if self.function_extractor.language_registry
                    else None
                )
                language_name = lang_config.name if lang_config else None

                for func_chunk in function_chunks:
                    # Apply data compression if function is data-heavy
                    func_chunk = self.function_extractor.apply_data_compression(
                        func_chunk, language=language_name
                    )

                    # Check token length
                    chunk_code = func_chunk.code
                    tokens = self.tokenizer.encode(chunk_code, truncation=False)

                    # If too long, try function without imports
                    if len(tokens) > chunk_size:
                        logger.debug(
                            f"Function {func_chunk.function_name} in {file_path} "
                            f"exceeds {chunk_size} tokens ({len(tokens)}), trying without imports"
                        )
                        # Try just the function code without imports
                        func_code_only = func_chunk.function_code
                        tokens_without_imports = self.tokenizer.encode(
                            func_code_only, truncation=False
                        )

                        if len(tokens_without_imports) > chunk_size:
                            logger.warning(
                                f"Skipping function {func_chunk.function_name} in {file_path}: "
                                f"still too long ({len(tokens_without_imports)} tokens)"
                            )
                            continue
                        else:
                            chunk_code = func_code_only

                    # Generate cache key
                    cache_key = f"{domain}:{file_path}:{func_chunk.function_name}:{hash(chunk_code)}"

                    # Get or generate embedding
                    if cache_key in self.embedding_cache:
                        embedding = self.embedding_cache[cache_key]
                    else:
                        embedding = self._extract_code_embeddings(chunk_code)
                        self.embedding_cache[cache_key] = embedding

                    # Store function metadata
                    embeddings_list.append(embedding)
                    file_paths.append(file_path)
                    code_snippets.append(chunk_code)
                    function_names.append(func_chunk.function_name)
                    is_methods.append(func_chunk.is_method)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        if not embeddings_list:
            logger.warning(f"No valid embeddings generated for domain {domain}")
            return

        # Create embeddings matrix
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)

        # Normalize embeddings for cosine similarity
        embeddings_matrix = self._normalize_embeddings(embeddings_matrix)

        # Build FAISS index
        dimension = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings_matrix)

        # Store the pattern index with function metadata
        pattern_index = PatternIndex(
            domain=domain,
            index=index,
            file_paths=file_paths,
            code_snippets=code_snippets,
            embeddings=embeddings_matrix,
            metadata={
                "num_patterns": len(embeddings_list),
                "embedding_dimension": dimension,
                "model_name": self.model_name,
                "chunk_size": chunk_size,
                "chunking_strategy": "function-level",
            },
            function_names=function_names,
            is_methods=is_methods,
        )

        self.domain_indices[domain] = pattern_index

        logger.info(
            f"Built index for domain {domain}: {len(embeddings_list)} patterns, "
            f"dimension {dimension}"
        )

        # Optionally save index to disk
        self._save_domain_index(pattern_index)

    def search_similar_patterns(
        self, query_code: str, domain: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> list[SimilarityMatch]:
        """Search for similar code patterns within a domain.

        Args:
            query_code: Code snippet to find similar patterns for
            domain: Domain to search within
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of SimilarityMatch objects ordered by similarity score
        """
        if domain not in self.domain_indices:
            logger.warning(f"No index found for domain {domain}")
            return []

        pattern_index = self.domain_indices[domain]

        try:
            # Log domain statistics before search
            logger.debug(f"=== PATTERN SEARCH INITIALIZATION (domain: {domain}) ===")
            logger.debug(f"Index contains {pattern_index.index.ntotal} patterns")
            logger.debug(f"Embedding dimension: {pattern_index.embeddings.shape[1]}")
            logger.debug(f"Unique files in index: {len(set(pattern_index.file_paths))}")

            # Extract query embedding
            query_embedding = self._extract_code_embeddings(query_code)
            query_embedding_normalized = self._normalize_embeddings(
                query_embedding.reshape(1, -1)
            )

            # Verify query embedding normalization
            query_norm = float(np.linalg.norm(query_embedding_normalized))
            logger.debug(f"Query embedding norm: {query_norm:.8f} (should be ~1.0)")
            logger.debug(
                f"Query code length: {len(query_code)} chars, first 50: {query_code[:50].replace(chr(10), ' ')}"
            )

            # Search in FAISS index - get more results for detailed logging
            search_k = max(top_k, 10)  # Search for at least 10 to see distribution
            scores, indices = pattern_index.index.search(
                query_embedding_normalized, search_k
            )

            # Debug logging for raw FAISS similarities
            logger.debug("=== FAISS SIMILARITY SEARCH RESULTS ===")
            logger.debug(
                f"Top {min(10, len(scores[0]))} raw similarities: {[f'{s:.6f}' for s in scores[0][:10]]}"
            )
            logger.debug(
                f"Mean of top {top_k} matches: {float(scores[0][:top_k].mean()):.6f}"
            )
            logger.debug(
                f"Max similarity: {float(scores[0][0]) if len(scores[0]) > 0 else 0:.6f}"
            )

            # Detailed logging for each match above threshold
            logger.debug("=== DETAILED MATCH ANALYSIS ===")
            for i, (score, idx) in enumerate(
                zip(scores[0][:search_k], indices[0][:search_k], strict=False)
            ):
                if idx == -1:
                    continue

                matched_file = pattern_index.file_paths[idx]
                logger.debug(
                    f"Match {i+1}: score={score:.8f}, file={matched_file}, idx={idx}"
                )

                # Enhanced logging for perfect/near-perfect matches
                if score >= 0.999:
                    logger.warning(
                        f"!!! PERFECT/NEAR-PERFECT MATCH DETECTED (score: {score:.10f}) !!!"
                    )
                    logger.warning(f"  Matched file: {matched_file}")

                    # Manual verification: calculate dot product
                    query_vec = query_embedding_normalized[0]
                    matched_vec = pattern_index.embeddings[idx]

                    # Verify matched embedding is normalized
                    matched_norm = float(np.linalg.norm(matched_vec))
                    logger.warning(
                        f"  Matched embedding norm: {matched_norm:.8f} (should be ~1.0)"
                    )

                    # Calculate dot product manually
                    manual_dot_product = float(np.dot(query_vec, matched_vec))
                    logger.warning(
                        f"  Manual dot product: {manual_dot_product:.10f} (vs FAISS: {score:.10f})"
                    )

                    # Check if embeddings are identical
                    embedding_diff = np.abs(query_vec - matched_vec)
                    max_diff = float(np.max(embedding_diff))
                    mean_diff = float(np.mean(embedding_diff))
                    logger.warning(
                        f"  Embedding difference: max={max_diff:.10f}, mean={mean_diff:.10f}"
                    )

                    # Log first 10 dimensions for comparison
                    logger.warning(
                        f"  Query embedding (first 10): {query_vec[:10].tolist()}"
                    )
                    logger.warning(
                        f"  Matched embedding (first 10): {matched_vec[:10].tolist()}"
                    )

                    # Log code snippet preview
                    matched_snippet = pattern_index.code_snippets[idx][:100].replace(
                        "\n", " "
                    )
                    logger.warning(f"  Matched code preview: {matched_snippet}...")

            # Create similarity matches (only for top_k and above threshold)
            similarity_matches = []
            for score, idx in zip(scores[0][:top_k], indices[0][:top_k], strict=False):
                if idx == -1 or score < min_similarity:
                    continue

                similarity_match = SimilarityMatch(
                    file_path=pattern_index.file_paths[idx],
                    similarity_score=float(score),
                    code_snippet=pattern_index.code_snippets[idx],
                    domain=domain,
                    context={
                        "index": int(idx),
                        "embedding_dimension": query_embedding_normalized.shape[1],
                        "search_parameters": {
                            "top_k": top_k,
                            "min_similarity": min_similarity,
                        },
                    },
                )
                similarity_matches.append(similarity_match)

            logger.debug(
                f"=== SEARCH COMPLETE: Found {len(similarity_matches)} similar patterns for domain {domain} ==="
            )
            return similarity_matches

        except Exception as e:
            logger.error(f"Error searching patterns in domain {domain}: {e}")
            return []

    def get_domain_statistics(self, domain: str) -> dict[str, Any]:
        """Get statistics for a domain index.

        Args:
            domain: Domain name

        Returns:
            Dictionary with index statistics
        """
        if domain not in self.domain_indices:
            return {}

        pattern_index = self.domain_indices[domain]
        return {
            "domain": domain,
            "num_patterns": pattern_index.index.ntotal,
            "embedding_dimension": pattern_index.embeddings.shape[1],
            "unique_files": len(set(pattern_index.file_paths)),
            "metadata": pattern_index.metadata,
        }

    def _last_token_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool embeddings using last token strategy (for Qodo-Embed).

        Args:
            last_hidden_state: Model output hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        # Get the position of the last non-padding token for each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

        if left_padding:
            # Left padding: last real token is at position sum(attention_mask) - 1
            return last_hidden_state[:, -1, :]
        else:
            # Right padding: find last non-padding token per sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]

    def _extract_code_embeddings(
        self, code_content: str
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Extract code embeddings for code content with hardware optimization.

        Args:
            code_content: Source code content

        Returns:
            Embedding vector
        """
        import time

        start_time = time.time()

        try:
            # Tokenize the code
            tokens = self.tokenizer(
                code_content,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            # Move tokens to device with error handling for MPS issues
            try:
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            except RuntimeError as e:
                if "mps" in str(e).lower():
                    logger.warning(
                        f"MPS error during token transfer: {e}. Retrying on CPU."
                    )
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                else:
                    raise e

            # Extract embeddings with mixed precision if enabled
            mixed_precision = self.device_manager.device_config.enable_mixed_precision
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=torch.float16)
                if mixed_precision and self.device.type in ["cuda", "mps"]
                else torch.no_grad()
            )

            with autocast_context:
                try:
                    outputs = self.model(**tokens)
                    # Use last token pooling for Qodo-Embed
                    pooled_output = self._last_token_pool(
                        outputs.last_hidden_state, tokens["attention_mask"]
                    )
                    embeddings = pooled_output.detach().cpu().numpy()
                except (RuntimeError, OSError, SystemError) as e:
                    if "mps" in str(e).lower() or any(
                        term in str(e).lower()
                        for term in ["segmentation", "memory", "device", "metal"]
                    ):
                        logger.warning(
                            f"MPS error during model inference (possible segmentation fault): {e}. "
                            "Falling back to CPU."
                        )
                        # Track inference failure
                        self.fallback_events["mps_inference_failures"] = (
                            int(self.fallback_events["mps_inference_failures"]) + 1
                        )
                        failure_reasons = list(self.fallback_events["failure_reasons"])
                        failure_reasons.append(f"Model inference: {str(e)[:100]}")
                        self.fallback_events["failure_reasons"] = failure_reasons

                        self._fallback_to_cpu()
                        # Retry on CPU
                        tokens = {k: v.to(self.device) for k, v in tokens.items()}
                        with torch.no_grad():
                            outputs = self.model(**tokens)
                            pooled_output = self._last_token_pool(
                                outputs.last_hidden_state, tokens["attention_mask"]
                            )
                            embeddings = pooled_output.detach().cpu().numpy()
                    else:
                        raise e

            # Update performance metrics
            elapsed_time = time.time() - start_time
            self._performance_metrics["total_embeddings_generated"] += 1

            # Update average embedding time with exponential moving average
            alpha = 0.1  # Smoothing factor
            current_avg = self._performance_metrics["average_embedding_time"]
            self._performance_metrics["average_embedding_time"] = (
                alpha * elapsed_time + (1 - alpha) * current_avg
            )

            return embeddings.squeeze()

        except Exception as e:
            logger.warning(f"Failed to extract embeddings for code snippet: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Qodo-Embed has 1536-dim embeddings

    def _normalize_embeddings(
        self, embeddings: np.ndarray[Any, np.dtype[np.floating[Any]]]
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Normalize embeddings for cosine similarity.

        Args:
            embeddings: Raw embeddings matrix

        Returns:
            Normalized embeddings matrix
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms

    def _save_domain_index(self, pattern_index: PatternIndex) -> None:
        """Save a domain index to disk for caching using JSON serialization.

        Args:
            pattern_index: PatternIndex to save
        """
        try:
            index_path = self.cache_dir / f"{pattern_index.domain}_index.json"

            # Save index data (excluding the FAISS index itself) with JSON serialization
            index_data = {
                "domain": pattern_index.domain,
                "file_paths": pattern_index.file_paths,
                "code_snippets": pattern_index.code_snippets,
                "embeddings": self._numpy_to_json_serializable(
                    pattern_index.embeddings
                ),
                "metadata": pattern_index.metadata,
                "function_names": pattern_index.function_names or [],
                "is_methods": pattern_index.is_methods or [],
                "format_version": "json_v2",  # Updated for function-level chunking
            }

            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            logger.debug(
                f"Saved domain index for {pattern_index.domain} to {index_path}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to save domain index for {pattern_index.domain}: {e}"
            )

    def load_domain_index(self, domain: str) -> bool:
        """Load a domain index from disk using JSON serialization.

        Args:
            domain: Domain name to load

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            index_path = self.cache_dir / f"{domain}_index.json"

            if not index_path.exists():
                logger.info(f"No cached index found for domain {domain}")
                return False

            with open(index_path, encoding="utf-8") as f:
                index_data = json.load(f)

            # Convert embeddings back to NumPy array
            if isinstance(index_data["embeddings"], dict):
                embeddings = self._json_to_numpy_array(index_data["embeddings"]).astype(
                    np.float32
                )
            else:
                # Handle legacy format if needed
                embeddings = np.array(index_data["embeddings"]).astype(np.float32)

            # Reconstruct FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

            # Load function metadata if available (new format)
            function_names = index_data.get("function_names")
            is_methods = index_data.get("is_methods")

            pattern_index = PatternIndex(
                domain=index_data["domain"],
                index=index,
                file_paths=index_data["file_paths"],
                code_snippets=index_data["code_snippets"],
                embeddings=embeddings,
                metadata=index_data["metadata"],
                function_names=function_names,
                is_methods=is_methods,
            )

            self.domain_indices[domain] = pattern_index
            logger.info(f"Loaded domain index for {domain} from JSON cache")
            return True

        except Exception as e:
            logger.warning(f"Failed to load domain index for {domain}: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear embedding cache to free memory."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "embedding_cache_size": len(self.embedding_cache),
            "domain_indices_count": len(self.domain_indices),
            "cache_directory": str(self.cache_dir),
            "model_device": str(self.device),
            "domains": list(self.domain_indices.keys()),
        }

    def analyze_embedding_divergence(
        self,
        query_embedding: np.ndarray[Any, np.dtype[np.floating[Any]]],
        reference_embeddings: list[np.ndarray[Any, np.dtype[np.floating[Any]]]],
    ) -> dict[str, Any]:
        """Analyze embedding divergence using only embedding-derived metrics.

        Args:
            query_embedding: Embedding vector for the file being explained
            reference_embeddings: Embeddings from similar patterns

        Returns:
            Dictionary with embedding-derived divergence metrics only
        """
        logger.info("Analyzing embedding divergence")

        # Calculate embedding divergence score
        if reference_embeddings:
            # Average reference embeddings
            avg_ref_embedding = np.mean(reference_embeddings, axis=0)

            # Calculate element-wise differences
            diff_vector = np.abs(query_embedding - avg_ref_embedding)

            # Overall divergence score (mean absolute difference)
            divergence_score = float(np.mean(diff_vector))

            # Find top divergent dimensions
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
            divergence_score = 1.0  # Max divergence if no references
            dimension_divergence = []
            similarity_distribution = []
            avg_similarity = 0.0

        result = {
            "divergence_score": divergence_score,
            "embedding_analysis": {
                "dimension_divergence": dimension_divergence[:10],  # Top 10
                "similarity_distribution": similarity_distribution,
                "avg_similarity": avg_similarity,
            },
        }

        logger.info(f"Divergence analysis complete: score={divergence_score:.3f}")

        return result

    def _get_fine_tuned_model_path(self, commit_hash: str | None = None) -> Path:
        """Get the path to a fine-tuned model (local or download from HuggingFace Hub).

        Args:
            commit_hash: Commit hash or HuggingFace Hub model ID (e.g., "username/model-name")

        Returns:
            Path to fine-tuned model directory
        """
        model_id = commit_hash or self.fine_tuned_model_commit
        if not model_id:
            raise ValueError("No model ID provided for fine-tuned model")

        # Check if this is a HuggingFace Hub model ID (contains '/')
        if "/" in model_id:
            return self._download_from_huggingface_hub(model_id)
        else:
            # Local model: use first 7 characters of commit hash as model name
            model_name = model_id[:7]
            return self.cache_dir / "fine_tuned_models" / model_name

    def _download_from_huggingface_hub(self, hub_model_id: str) -> Path:
        """Download a fine-tuned model from HuggingFace Hub.

        Args:
            hub_model_id: HuggingFace Hub model ID (e.g., "username/model-name")

        Returns:
            Path to downloaded model directory
        """
        try:
            from huggingface_hub import snapshot_download

            # Create cache directory for Hub models
            hub_cache_dir = self.cache_dir / "hub_models"
            hub_cache_dir.mkdir(exist_ok=True)

            # Sanitize model ID for directory name (replace / with --)
            safe_model_name = hub_model_id.replace("/", "--")
            local_model_dir = hub_cache_dir / safe_model_name

            # Download if not already cached
            if not local_model_dir.exists():
                logger.info(f"Downloading model from HuggingFace Hub: {hub_model_id}")
                # User-specified fine-tuned models cannot have pinned revisions
                # as they are custom models uploaded by users, not base models
                downloaded_path = snapshot_download(  # nosec B615
                    repo_id=hub_model_id,
                    cache_dir=str(hub_cache_dir),
                    local_dir=str(local_model_dir),
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Model downloaded to: {downloaded_path}")
            else:
                logger.info(f"Using cached Hub model from: {local_model_dir}")

            return local_model_dir

        except ImportError:
            raise ValueError(
                "huggingface_hub not installed. Install with: pip install huggingface-hub"
            ) from None
        except Exception as e:
            raise ValueError(
                f"Failed to download model from HuggingFace Hub ({hub_model_id}): {e}"
            ) from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "is_fine_tuned": self.is_fine_tuned,
            "fine_tuned_commit": self.fine_tuned_model_commit,
            "device": str(self.device),
            "device_type": self.device_manager.hardware_info.device_type.value,
        }
