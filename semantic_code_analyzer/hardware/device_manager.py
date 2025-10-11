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

"""Advanced device management and hardware acceleration for Apple Silicon and other platforms.

This module provides intelligent hardware detection, device optimization, and
acceleration support with specific optimizations for Apple M3 chips while
maintaining backward compatibility with other hardware configurations.
"""

from __future__ import annotations

import logging
import platform
import subprocess  # nosec B404 - Used for trusted macOS system_profiler hardware detection only
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for hardware acceleration."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Metal Performance Shaders (Apple Silicon)
    ROCM = "rocm"  # AMD ROCm (AMD GPUs)
    XPU = "xpu"  # Intel XPU (Intel GPUs via oneAPI)


class ChipGeneration(Enum):
    """Apple Silicon chip generations."""

    M1 = "m1"
    M1_PRO = "m1_pro"
    M1_MAX = "m1_max"
    M1_ULTRA = "m1_ultra"
    M2 = "m2"
    M2_PRO = "m2_pro"
    M2_MAX = "m2_max"
    M2_ULTRA = "m2_ultra"
    M3 = "m3"
    M3_PRO = "m3_pro"
    M3_MAX = "m3_max"
    M3_ULTRA = "m3_ultra"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """Comprehensive hardware information container."""

    platform: str
    architecture: str
    cpu_count: int
    memory_gb: float
    device_type: DeviceType
    device_name: str
    is_apple_silicon: bool
    chip_generation: ChipGeneration | None
    supports_mps: bool
    supports_cuda: bool
    supports_rocm: bool
    supports_xpu: bool
    neural_engine_cores: int | None
    gpu_cores: int | None
    memory_bandwidth_gbps: float | None
    unified_memory: bool


@dataclass
class DeviceConfig:
    """Configuration for device-specific optimizations."""

    device_type: DeviceType
    batch_size: int = 32
    max_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_memory_pooling: bool = True
    optimize_for_inference: bool = True
    fallback_enabled: bool = True


class DeviceManager:
    """Advanced device manager with Apple M3 optimizations and intelligent fallback.

    This class provides:
    - Automatic hardware detection with M3-specific optimizations
    - Intelligent device selection with performance prioritization
    - Memory management optimized for unified memory architectures
    - Graceful fallback mechanisms for maximum compatibility
    - Performance monitoring and optimization recommendations
    """

    def __init__(self, prefer_device: DeviceType | None = None):
        """Initialize device manager with hardware detection.

        Args:
            prefer_device: Preferred device type (None for auto-detection)
        """
        logger.info("=== DEVICE MANAGER INIT ===")
        logger.info(
            f"Starting DeviceManager initialization with prefer_device={prefer_device}"
        )

        self.prefer_device = prefer_device
        self._hardware_info: HardwareInfo | None = None
        self._device_config: DeviceConfig | None = None
        self._performance_metrics: dict[str, Any] = {}

        logger.info("Initializing device manager with hardware detection")
        try:
            logger.info("Starting hardware detection...")
            self._detect_hardware()
            logger.info("Hardware detection completed successfully")
        except Exception as e:
            logger.error(f"CRITICAL: Hardware detection failed: {e}")
            logger.exception("Full traceback for hardware detection failure:")
            raise

        try:
            logger.info("Starting device configuration...")
            self._configure_device()
            logger.info("Device configuration completed successfully")
        except Exception as e:
            logger.error(f"CRITICAL: Device configuration failed: {e}")
            logger.exception("Full traceback for device configuration failure:")
            raise

        logger.info("=== DEVICE MANAGER INIT COMPLETE ===")

    @property
    def hardware_info(self) -> HardwareInfo:
        """Get comprehensive hardware information."""
        if self._hardware_info is None:
            self._detect_hardware()
        assert self._hardware_info is not None  # For mypy
        return self._hardware_info

    @property
    def device_config(self) -> DeviceConfig:
        """Get optimized device configuration."""
        if self._device_config is None:
            self._configure_device()
        assert self._device_config is not None  # For mypy
        return self._device_config

    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device object.

        Returns:
            PyTorch device object with proper device string mapping
        """
        device_type = self.device_config.device_type

        # ROCm uses CUDA API compatibility layer (HIP)
        # PyTorch expects "cuda" device string for ROCm
        if device_type == DeviceType.ROCM:
            return torch.device("cuda")

        return torch.device(device_type.value)

    def _detect_hardware(self) -> None:
        """Perform comprehensive hardware detection with M3-specific logic."""
        logger.info("=== HARDWARE DETECTION START ===")
        logger.info("Performing comprehensive hardware detection")

        # Basic system information
        logger.info("Gathering basic system information...")
        system_platform = platform.system()
        architecture = platform.machine()
        cpu_count = psutil.cpu_count(logical=False) or 1  # Fallback to 1 if None
        memory_gb = psutil.virtual_memory().total / (1024**3)

        logger.info(f"System platform: {system_platform}")
        logger.info(f"Architecture: {architecture}")
        logger.info(f"CPU count: {cpu_count}")
        logger.info(f"Memory GB: {memory_gb:.1f}")

        # Apple Silicon detection
        logger.info("Detecting Apple Silicon...")
        try:
            is_apple_silicon = self._detect_apple_silicon()
            logger.info(f"Apple Silicon detected: {is_apple_silicon}")
        except Exception as e:
            logger.error(f"Error detecting Apple Silicon: {e}")
            is_apple_silicon = False

        chip_generation = None
        if is_apple_silicon:
            logger.info("Detecting Apple chip generation...")
            try:
                chip_generation = self._detect_apple_chip_generation()
                logger.info(f"Chip generation: {chip_generation}")
            except Exception as e:
                logger.error(f"Error detecting chip generation: {e}")

        # Device capability detection
        logger.info("=== DEVICE CAPABILITY DETECTION ===")
        logger.info("Checking MPS availability - CRITICAL POINT FOR SEGFAULTS")
        try:
            supports_mps = self._check_mps_availability()
            logger.info(f"MPS support detected: {supports_mps}")
        except Exception as e:
            logger.error(f"CRITICAL: MPS availability check failed: {e}")
            logger.exception("Full traceback for MPS detection failure:")
            supports_mps = False

        logger.info("Checking CUDA availability...")
        try:
            supports_cuda = torch.cuda.is_available()
            logger.info(f"CUDA support detected: {supports_cuda}")
        except Exception as e:
            logger.error(f"Error checking CUDA availability: {e}")
            supports_cuda = False

        logger.info("Checking ROCm availability...")
        try:
            supports_rocm = self._check_rocm_availability()
            logger.info(f"ROCm support detected: {supports_rocm}")
        except Exception as e:
            logger.error(f"Error checking ROCm availability: {e}")
            supports_rocm = False

        logger.info("Checking Intel XPU availability...")
        try:
            supports_xpu = self._check_xpu_availability()
            logger.info(f"Intel XPU support detected: {supports_xpu}")
        except Exception as e:
            logger.error(f"Error checking Intel XPU availability: {e}")
            supports_xpu = False

        # Apple Silicon specific details
        logger.info("Getting Apple Silicon specifications...")
        neural_engine_cores = None
        gpu_cores = None
        memory_bandwidth_gbps = None
        unified_memory = False

        if is_apple_silicon and chip_generation:
            try:
                (
                    neural_engine_cores,
                    gpu_cores,
                    memory_bandwidth_gbps,
                ) = self._get_apple_silicon_specs(chip_generation)
                unified_memory = True
                logger.info(f"Neural Engine cores: {neural_engine_cores}")
                logger.info(f"GPU cores: {gpu_cores}")
                logger.info(f"Memory bandwidth: {memory_bandwidth_gbps} GB/s")
                logger.info(f"Unified memory: {unified_memory}")
            except Exception as e:
                logger.error(f"Error getting Apple Silicon specs: {e}")

        # Device selection priority
        logger.info("=== DEVICE SELECTION ===")
        logger.info("Determining optimal device selection...")
        logger.info(f"Prefer device: {self.prefer_device}")
        logger.info(f"MPS available: {supports_mps}")
        logger.info(f"CUDA available: {supports_cuda}")
        logger.info(f"ROCm available: {supports_rocm}")
        logger.info(f"XPU available: {supports_xpu}")

        # Priority order (if no preference): MPS > CUDA > ROCm > XPU > CPU
        if supports_mps and (
            self.prefer_device is None or self.prefer_device == DeviceType.MPS
        ):
            device_type = DeviceType.MPS
            device_name = f"Apple Silicon MPS ({chip_generation.value if chip_generation else 'unknown'})"
            logger.info(f"Selected device: MPS - {device_name}")
        elif supports_cuda and (
            self.prefer_device is None or self.prefer_device == DeviceType.CUDA
        ):
            device_type = DeviceType.CUDA
            device_name = f"CUDA ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'})"
            logger.info(f"Selected device: CUDA - {device_name}")
        elif supports_rocm and (
            self.prefer_device is None or self.prefer_device == DeviceType.ROCM
        ):
            device_type = DeviceType.ROCM
            device_name = f"AMD ROCm ({self._get_rocm_device_name()})"
            logger.info(f"Selected device: ROCm - {device_name}")
        elif supports_xpu and (
            self.prefer_device is None or self.prefer_device == DeviceType.XPU
        ):
            device_type = DeviceType.XPU
            device_name = f"Intel XPU ({self._get_xpu_device_name()})"
            logger.info(f"Selected device: XPU - {device_name}")
        else:
            device_type = DeviceType.CPU
            device_name = f"CPU ({platform.processor() or 'unknown'})"
            logger.info(f"Selected device: CPU - {device_name}")

        logger.info("Creating HardwareInfo object...")
        self._hardware_info = HardwareInfo(
            platform=system_platform,
            architecture=architecture,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            device_type=device_type,
            device_name=device_name,
            is_apple_silicon=is_apple_silicon,
            chip_generation=chip_generation,
            supports_mps=supports_mps,
            supports_cuda=supports_cuda,
            supports_rocm=supports_rocm,
            supports_xpu=supports_xpu,
            neural_engine_cores=neural_engine_cores,
            gpu_cores=gpu_cores,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            unified_memory=unified_memory,
        )

        logger.info(
            f"=== HARDWARE DETECTION COMPLETE === Hardware detected: {device_name} with {memory_gb:.1f}GB memory"
        )

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        if platform.system() != "Darwin":
            return False

        try:
            # Check for Apple Silicon via system_profiler
            result = subprocess.run(  # nosec B607,B603
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout.lower()
                return any(
                    chip in output for chip in ["apple m1", "apple m2", "apple m3"]
                )

        except subprocess.TimeoutExpired:
            logger.debug(
                "system_profiler command timed out during Apple Silicon detection"
            )
        except FileNotFoundError:
            logger.debug(
                "system_profiler command not found, using fallback architecture detection"
            )

        # Fallback: check architecture
        return platform.machine() in ["arm64", "aarch64"]

    def _check_rocm_availability(self) -> bool:
        """Check if AMD ROCm is available for PyTorch.

        Returns:
            True if ROCm is available, False otherwise
        """
        try:
            # Check if PyTorch was built with ROCm support
            # ROCm uses CUDA API compatibility, so we check for 'hip' in version
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                # Check if ROCm device is actually available
                if torch.cuda.is_available():
                    return True
            return False
        except Exception as e:
            logger.debug(f"ROCm availability check failed: {e}")
            return False

    def _check_xpu_availability(self) -> bool:
        """Check if Intel XPU (oneAPI) is available for PyTorch.

        Returns:
            True if Intel XPU is available, False otherwise
        """
        try:
            # Check if PyTorch has XPU support (intel_extension_for_pytorch)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return True
            return False
        except Exception as e:
            logger.debug(f"Intel XPU availability check failed: {e}")
            return False

    def _get_rocm_device_name(self) -> str:
        """Get AMD GPU device name via ROCm.

        Returns:
            Device name string
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
            return "unknown AMD GPU"
        except Exception:
            return "unknown AMD GPU"

    def _get_xpu_device_name(self) -> str:
        """Get Intel XPU device name.

        Returns:
            Device name string
        """
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                # Try to get device name
                if hasattr(torch.xpu, "get_device_name"):
                    return torch.xpu.get_device_name(0)
                return "Intel XPU"
            return "unknown Intel XPU"
        except Exception:
            return "unknown Intel XPU"

    def _detect_apple_chip_generation(self) -> ChipGeneration | None:
        """Detect specific Apple Silicon chip generation."""
        if not self._detect_apple_silicon():
            return None

        try:
            result = subprocess.run(  # nosec B607,B603
                ["/usr/sbin/system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout.lower()

                # M3 variants (prioritized for latest features)
                if "apple m3 ultra" in output:
                    return ChipGeneration.M3_ULTRA
                elif "apple m3 max" in output:
                    return ChipGeneration.M3_MAX
                elif "apple m3 pro" in output:
                    return ChipGeneration.M3_PRO
                elif "apple m3" in output:
                    return ChipGeneration.M3

                # M2 variants
                elif "apple m2 ultra" in output:
                    return ChipGeneration.M2_ULTRA
                elif "apple m2 max" in output:
                    return ChipGeneration.M2_MAX
                elif "apple m2 pro" in output:
                    return ChipGeneration.M2_PRO
                elif "apple m2" in output:
                    return ChipGeneration.M2

                # M1 variants
                elif "apple m1 ultra" in output:
                    return ChipGeneration.M1_ULTRA
                elif "apple m1 max" in output:
                    return ChipGeneration.M1_MAX
                elif "apple m1 pro" in output:
                    return ChipGeneration.M1_PRO
                elif "apple m1" in output:
                    return ChipGeneration.M1

        except subprocess.TimeoutExpired:
            logger.debug(
                "system_profiler command timed out during chip generation detection"
            )
        except FileNotFoundError:
            logger.debug(
                "system_profiler command not found, returning unknown chip generation"
            )

        return ChipGeneration.UNKNOWN

    def _check_mps_availability(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available with enhanced validation."""
        logger.info("=== MPS AVAILABILITY CHECK START ===")
        logger.info("Starting comprehensive MPS availability check")

        try:
            # Basic availability check
            logger.info("Step 1: Checking basic MPS availability...")
            logger.info("Checking if torch.backends has 'mps' attribute...")

            if not hasattr(torch.backends, "mps"):
                logger.info("torch.backends.mps not available - MPS not supported")
                return False

            logger.info("torch.backends.mps found, checking is_available()...")

            if not torch.backends.mps.is_available():
                logger.info("torch.backends.mps.is_available() returned False")
                return False

            logger.info("torch.backends.mps.is_available() returned True")

            logger.info("Checking platform is Darwin...")
            if platform.system() != "Darwin":
                logger.info(
                    f"Platform is {platform.system()}, not Darwin - MPS not available"
                )
                return False

            logger.info("Platform check passed - running on Darwin")

            # Enhanced validation: Test actual MPS operations to prevent segfaults
            logger.info("=== MPS ENHANCED VALIDATION START ===")
            logger.info(
                "WARNING: About to test actual MPS operations - segfault risk area!"
            )

            try:
                logger.info("Step 2: Creating test tensor on MPS device...")
                logger.info("torch.randn(2, 2, device='mps', dtype=torch.float32)")

                test_tensor = torch.randn(2, 2, device="mps", dtype=torch.float32)
                logger.info("Test tensor created successfully on MPS")
                logger.info(f"Test tensor shape: {test_tensor.shape}")
                logger.info(f"Test tensor device: {test_tensor.device}")
                logger.info(f"Test tensor dtype: {test_tensor.dtype}")

                logger.info("Step 3: Performing matrix multiplication on MPS...")
                result = torch.mm(test_tensor, test_tensor)
                logger.info("Matrix multiplication completed successfully")
                logger.info(f"Result shape: {result.shape}")
                logger.info(f"Result device: {result.device}")

                logger.info("Step 4: Synchronizing MPS operations...")
                torch.mps.synchronize()
                logger.info("MPS synchronization completed successfully")

                logger.info("Step 5: Testing memory operations...")
                logger.info("Calling torch.mps.empty_cache()...")
                torch.mps.empty_cache()
                logger.info("MPS cache cleared successfully")

                logger.info("=== MPS ENHANCED VALIDATION PASSED ===")
                logger.debug("Enhanced MPS validation passed")
                return True

            except (RuntimeError, OSError, SystemError) as e:
                logger.error(f"CRITICAL: MPS enhanced validation failed: {e}")
                logger.error("This failure indicates MPS segmentation fault risk!")
                logger.warning(
                    f"MPS failed enhanced validation test: {e}. "
                    "Disabling MPS to prevent segmentation faults."
                )
                logger.exception("Full traceback for MPS validation failure:")
                return False

        except Exception as e:
            logger.error(f"CRITICAL: MPS availability check failed with exception: {e}")
            logger.exception("Full traceback for MPS availability check failure:")
            logger.debug(f"MPS availability check failed: {e}")
            return False

    def _get_apple_silicon_specs(
        self, chip: ChipGeneration
    ) -> tuple[int | None, int | None, float | None]:
        """Get Apple Silicon chip specifications.

        Returns:
            Tuple of (neural_engine_cores, gpu_cores, memory_bandwidth_gbps)
        """
        # Specifications for Apple Silicon chips
        specs = {
            ChipGeneration.M1: (16, 7, 68.25),
            ChipGeneration.M1_PRO: (16, 14, 200.0),
            ChipGeneration.M1_MAX: (16, 24, 400.0),
            ChipGeneration.M1_ULTRA: (32, 48, 800.0),
            ChipGeneration.M2: (16, 8, 100.0),
            ChipGeneration.M2_PRO: (16, 16, 200.0),
            ChipGeneration.M2_MAX: (16, 30, 400.0),
            ChipGeneration.M2_ULTRA: (32, 60, 800.0),
            ChipGeneration.M3: (16, 10, 100.0),
            ChipGeneration.M3_PRO: (16, 18, 150.0),
            ChipGeneration.M3_MAX: (16, 30, 300.0),
            ChipGeneration.M3_ULTRA: (32, 60, 600.0),
        }

        return specs.get(chip, (None, None, None))

    def _configure_device(self) -> None:
        """Configure device-specific optimizations."""
        hardware = self.hardware_info

        # M3-optimized configuration
        if hardware.chip_generation and hardware.chip_generation.value.startswith("m3"):
            batch_size = self._calculate_optimal_batch_size_m3()
            max_memory_fraction = 0.75  # Conservative for M3's unified memory
            enable_mixed_precision = True  # M3 supports efficient mixed precision
            enable_memory_pooling = True  # Beneficial for unified memory

        # M2 optimizations
        elif hardware.chip_generation and hardware.chip_generation.value.startswith(
            "m2"
        ):
            batch_size = max(16, int(hardware.memory_gb * 2))
            max_memory_fraction = 0.7
            enable_mixed_precision = True
            enable_memory_pooling = True

        # M1 optimizations
        elif hardware.chip_generation and hardware.chip_generation.value.startswith(
            "m1"
        ):
            batch_size = max(8, int(hardware.memory_gb * 1.5))
            max_memory_fraction = 0.6
            enable_mixed_precision = True
            enable_memory_pooling = True

        # CUDA optimizations
        elif hardware.device_type == DeviceType.CUDA:
            batch_size = 64  # CUDA generally handles larger batches well
            max_memory_fraction = 0.8
            enable_mixed_precision = True
            enable_memory_pooling = True

        # ROCm optimizations (AMD GPUs)
        elif hardware.device_type == DeviceType.ROCM:
            batch_size = 64  # ROCm has similar performance to CUDA
            max_memory_fraction = 0.8
            enable_mixed_precision = True
            enable_memory_pooling = True

        # Intel XPU optimizations
        elif hardware.device_type == DeviceType.XPU:
            batch_size = 32  # Conservative batch size for Intel XPU
            max_memory_fraction = 0.7
            enable_mixed_precision = True
            enable_memory_pooling = True

        # CPU fallback
        else:
            batch_size = max(1, min(8, hardware.cpu_count))
            max_memory_fraction = 0.5  # Conservative for CPU
            enable_mixed_precision = False  # Limited benefit on CPU
            enable_memory_pooling = False

        self._device_config = DeviceConfig(
            device_type=hardware.device_type,
            batch_size=batch_size,
            max_memory_fraction=max_memory_fraction,
            enable_mixed_precision=enable_mixed_precision,
            enable_memory_pooling=enable_memory_pooling,
            optimize_for_inference=True,
            fallback_enabled=True,
        )

        logger.info(
            f"Device configured: {hardware.device_type.value} with batch_size={batch_size}, "
            f"memory_fraction={max_memory_fraction}"
        )

    def _calculate_optimal_batch_size_m3(self) -> int:
        """Calculate optimal batch size for M3 chips based on available memory."""
        hardware = self.hardware_info

        # M3-specific batch size calculation
        base_batch_size = 32

        if hardware.memory_gb >= 36:  # M3 Max/Ultra
            return min(128, int(base_batch_size * 4))
        elif hardware.memory_gb >= 18:  # M3 Pro
            return min(64, int(base_batch_size * 2))
        else:  # M3 base
            return base_batch_size

    def get_device_recommendations(self) -> dict[str, Any]:
        """Get device-specific optimization recommendations."""
        hardware = self.hardware_info
        config = self.device_config

        recommendations = {
            "device_type": hardware.device_type.value,
            "batch_size": config.batch_size,
            "memory_optimization": {
                "max_memory_fraction": config.max_memory_fraction,
                "enable_memory_pooling": config.enable_memory_pooling,
                "unified_memory": hardware.unified_memory,
            },
            "performance_features": {
                "mixed_precision": config.enable_mixed_precision,
                "neural_engine_available": hardware.neural_engine_cores is not None,
                "gpu_cores": hardware.gpu_cores,
            },
        }

        # M3-specific recommendations
        if hardware.chip_generation and hardware.chip_generation.value.startswith("m3"):
            m3_optimizations: dict[str, Any] = {
                "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
                "recommended_chunk_size": 2048,  # Fixed size for code analysis (was too conservative at 240)
                "parallel_processing": True,
                "cache_optimization": "aggressive",
            }
            recommendations["m3_optimizations"] = m3_optimizations

        return recommendations

    def benchmark_device(self, duration_seconds: float = 5.0) -> dict[str, Any]:
        """Run a quick benchmark to measure device performance."""
        logger.info(f"Running {duration_seconds}s device benchmark")

        device = self.torch_device

        # Create test tensors
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warm up
        for _ in range(3):
            torch.mm(a, b)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

        # Benchmark
        start_time = time.time()
        operations = 0

        while time.time() - start_time < duration_seconds:
            torch.mm(a, b)
            operations += 1

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":  # Includes ROCm (uses CUDA API)
                torch.cuda.synchronize()
            elif device.type == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.synchronize()

        elapsed_time = time.time() - start_time
        ops_per_second = operations / elapsed_time

        benchmark_results = {
            "device": device.type,
            "operations_per_second": ops_per_second,
            "duration_seconds": elapsed_time,
            "total_operations": operations,
            "matrix_size": size,
            "performance_score": ops_per_second / 100,  # Normalized score
        }

        logger.info(
            f"Benchmark completed: {ops_per_second:.1f} ops/sec on {device.type}"
        )
        return benchmark_results

    def get_memory_info(self) -> dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {
            "system_memory_gb": self.hardware_info.memory_gb,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "memory_usage_percent": psutil.virtual_memory().percent,
        }

        # Add device-specific memory info
        if (
            self.hardware_info.device_type == DeviceType.CUDA
            and torch.cuda.is_available()
        ):
            memory_info.update(
                {
                    "cuda_memory_allocated_gb": torch.cuda.memory_allocated()
                    / (1024**3),
                    "cuda_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "cuda_memory_total_gb": torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / (1024**3),
                }
            )
        elif (
            self.hardware_info.device_type == DeviceType.ROCM
            and torch.cuda.is_available()
        ):
            # ROCm uses CUDA API compatibility (HIP)
            memory_info.update(
                {
                    "rocm_memory_allocated_gb": torch.cuda.memory_allocated()
                    / (1024**3),
                    "rocm_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "rocm_memory_total_gb": torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / (1024**3),
                }
            )
        elif self.hardware_info.device_type == DeviceType.MPS:
            # MPS shares system memory
            memory_info.update(
                {
                    "mps_unified_memory": True,
                    "mps_recommended_max_gb": self.hardware_info.memory_gb
                    * self.device_config.max_memory_fraction,
                }
            )
        elif self.hardware_info.device_type == DeviceType.XPU and hasattr(torch, "xpu"):
            # Intel XPU memory tracking
            try:
                if hasattr(torch.xpu, "memory_allocated"):
                    memory_info.update(
                        {
                            "xpu_memory_allocated_gb": torch.xpu.memory_allocated()
                            / (1024**3),
                        }
                    )
                if hasattr(torch.xpu, "memory_reserved"):
                    memory_info["xpu_memory_reserved_gb"] = (
                        torch.xpu.memory_reserved() / (1024**3)
                    )
            except Exception as e:
                logger.debug(f"Could not get XPU memory info: {e}")

        return memory_info

    def optimize_for_model_loading(self) -> dict[str, Any]:
        """Get optimized settings for model loading."""
        hardware = self.hardware_info

        optimization_settings = {
            "device_map": "auto" if hardware.device_type != DeviceType.CPU else None,
            "dtype": (
                torch.float16
                if self.device_config.enable_mixed_precision
                else torch.float32
            ),
            "low_cpu_mem_usage": hardware.unified_memory or hardware.memory_gb < 16,
            "trust_remote_code": False,  # Security best practice
        }

        # M3-specific optimizations
        if hardware.chip_generation and hardware.chip_generation.value.startswith("m3"):
            optimization_settings.update(
                {
                    "load_in_8bit": False,  # M3 handles full precision efficiently
                    "load_in_4bit": False,
                    "use_flash_attention": False,  # Not needed for embedding models
                    "optimize_for_inference": True,
                }
            )

        return optimization_settings

    def get_m3_performance_metrics(self) -> dict[str, Any]:
        """Get M3-specific performance metrics and optimization status.

        Returns:
            Dictionary with M3-specific performance metrics
        """
        hardware = self.hardware_info

        if not (
            hardware.is_apple_silicon
            and hardware.chip_generation
            and hardware.chip_generation.value.startswith("m3")
        ):
            return {"error": "M3 metrics only available on Apple M3 chips"}

        # Check MPS availability and current usage
        mps_available = hardware.supports_mps
        mps_active = mps_available and torch.backends.mps.is_available()

        # Get memory bandwidth utilization estimate
        memory_bandwidth_utilization = self._estimate_memory_bandwidth_utilization()

        # Get Neural Engine availability (M3 has 16-core Neural Engine)
        neural_engine_available = hardware.neural_engine_cores is not None

        metrics = {
            "chip_generation": hardware.chip_generation.value,
            "gpu_cores": hardware.gpu_cores,
            "neural_engine_cores": hardware.neural_engine_cores,
            "memory_bandwidth_gbps": hardware.memory_bandwidth_gbps,
            "unified_memory": hardware.unified_memory,
            "mps_acceleration": {
                "available": mps_available,
                "active": mps_active,
                "optimized_batch_size": self.device_config.batch_size,
                "mixed_precision_enabled": self.device_config.enable_mixed_precision,
            },
            "memory_optimization": {
                "max_memory_fraction": self.device_config.max_memory_fraction,
                "memory_pooling_enabled": self.device_config.enable_memory_pooling,
                "estimated_bandwidth_utilization": memory_bandwidth_utilization,
            },
            "neural_engine": {
                "available": neural_engine_available,
                "cores": hardware.neural_engine_cores,
                "ml_compute_available": self._check_ml_compute_availability(),
            },
            "optimization_features": {
                "tf32_enabled": self._check_tf32_status(),
                "inference_optimized": self.device_config.optimize_for_inference,
                "fallback_enabled": self.device_config.fallback_enabled,
            },
        }

        return metrics

    def _estimate_memory_bandwidth_utilization(self) -> float:
        """Estimate current memory bandwidth utilization.

        Returns:
            Estimated utilization as percentage (0.0 to 100.0)
        """
        try:
            # Get current memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Estimate bandwidth utilization based on memory usage and workload
            # This is a simplified estimation - actual bandwidth depends on access patterns
            base_utilization = min(
                memory_usage_percent * 0.8, 80.0
            )  # Conservative estimate

            return base_utilization
        except Exception:
            return 0.0

    def _check_ml_compute_availability(self) -> bool:
        """Check if ML Compute (CoreML with Neural Engine) is available.

        Returns:
            True if ML Compute is available, False otherwise
        """
        try:
            # Check if we're on macOS with Metal
            if platform.system() != "Darwin":
                return False

            # Simple heuristic: if MPS is available and we have Neural Engine cores
            return (
                self.hardware_info.supports_mps
                and self.hardware_info.neural_engine_cores is not None
            )
        except Exception:
            return False

    def _check_tf32_status(self) -> bool:
        """Check if TensorFloat-32 optimizations are enabled.

        Returns:
            True if TF32 is enabled, False otherwise
        """
        try:
            if self.hardware_info.device_type == DeviceType.MPS:
                # Check MPS TF32 setting
                return getattr(torch.backends.mps, "allow_tf32", False)
            elif self.hardware_info.device_type == DeviceType.CUDA:
                # Check CUDA TF32 settings
                return (
                    torch.backends.cuda.matmul.allow_tf32
                    and torch.backends.cudnn.allow_tf32
                )
            else:
                return False
        except Exception:
            return False

    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report for the current hardware.

        Returns:
            Formatted string report with optimization recommendations
        """
        hardware = self.hardware_info
        config = self.device_config

        report_lines = []
        report_lines.append("🚀 Hardware Acceleration Report")
        report_lines.append("=" * 40)
        report_lines.append(f"Device: {hardware.device_name}")
        report_lines.append(f"Platform: {hardware.platform} ({hardware.architecture})")
        report_lines.append(f"Memory: {hardware.memory_gb:.1f}GB")
        report_lines.append("")

        # Device-specific optimizations
        if hardware.is_apple_silicon:
            report_lines.append("🍎 Apple Silicon Optimizations:")
            if hardware.chip_generation:
                report_lines.append(
                    f"  • Chip Generation: {hardware.chip_generation.value.upper()}"
                )
            if hardware.gpu_cores:
                report_lines.append(f"  • GPU Cores: {hardware.gpu_cores}")
            if hardware.neural_engine_cores:
                report_lines.append(
                    f"  • Neural Engine: {hardware.neural_engine_cores} cores"
                )
            if hardware.memory_bandwidth_gbps:
                report_lines.append(
                    f"  • Memory Bandwidth: {hardware.memory_bandwidth_gbps:.1f} GB/s"
                )

            report_lines.append(
                f"  • MPS Acceleration: {'✅ Enabled' if hardware.supports_mps else '❌ Not Available'}"
            )
            report_lines.append(
                f"  • Unified Memory: {'✅ Yes' if hardware.unified_memory else '❌ No'}"
            )

        report_lines.append("")
        report_lines.append("⚙️ Current Configuration:")
        report_lines.append(f"  • Batch Size: {config.batch_size}")
        report_lines.append(f"  • Memory Fraction: {config.max_memory_fraction:.1%}")
        report_lines.append(
            f"  • Mixed Precision: {'✅ Enabled' if config.enable_mixed_precision else '❌ Disabled'}"
        )
        report_lines.append(
            f"  • Memory Pooling: {'✅ Enabled' if config.enable_memory_pooling else '❌ Disabled'}"
        )
        report_lines.append(
            f"  • Fallback: {'✅ Enabled' if config.fallback_enabled else '❌ Disabled'}"
        )

        # Performance recommendations
        report_lines.append("")
        report_lines.append("💡 Recommendations:")

        if hardware.is_apple_silicon and hardware.chip_generation:
            if hardware.chip_generation.value.startswith("m3"):
                report_lines.append("  • M3 detected: Excellent performance expected")
                report_lines.append(
                    "  • Consider enabling mixed precision for faster inference"
                )
                if hardware.memory_gb >= 24:
                    report_lines.append(
                        "  • High memory capacity: Can handle large batch sizes"
                    )
            elif hardware.chip_generation.value.startswith("m2"):
                report_lines.append("  • M2 detected: Good performance expected")
            elif hardware.chip_generation.value.startswith("m1"):
                report_lines.append(
                    "  • M1 detected: Adequate performance with optimizations"
                )

        if not hardware.supports_mps and hardware.is_apple_silicon:
            report_lines.append(
                "  ⚠️  MPS not available: Update PyTorch for better performance"
            )

        if hardware.device_type == DeviceType.CPU:
            report_lines.append(
                "  ⚠️  Using CPU: Consider hardware acceleration for better performance"
            )

        return "\n".join(report_lines)

    def get_device_status_report(self) -> dict[str, Any]:
        """Get detailed device status including validation results and warnings.

        Returns:
            Dictionary with device status, warnings, and performance implications.
        """
        hardware = self.hardware_info

        status_report: dict[str, Any] = {
            "intended_device": hardware.device_type.value,
            "device_name": hardware.device_name,
            "memory_gb": hardware.memory_gb,
            "is_apple_silicon": hardware.is_apple_silicon,
            "chip_generation": (
                hardware.chip_generation.value
                if hardware.chip_generation
                else "unknown"
            ),
            "mps_available": hardware.supports_mps,
            "cuda_available": hardware.supports_cuda,
            "rocm_available": hardware.supports_rocm,
            "xpu_available": hardware.supports_xpu,
            "warnings": [],
            "performance_notes": [],
            "troubleshooting": [],
        }

        # Add MPS-specific warnings and status
        if hardware.is_apple_silicon:
            if not hardware.supports_mps:
                status_report["warnings"].append(
                    {
                        "level": "warning",
                        "message": "MPS acceleration unavailable - falling back to CPU",
                        "impact": "Analysis will be 3-5x slower than expected",
                        "suggestion": "Update PyTorch version or check macOS compatibility",
                    }
                )
                status_report["performance_notes"].append(
                    "CPU fallback: Expect significantly longer analysis times"
                )
                status_report["troubleshooting"].append(
                    "Try: pip install --upgrade torch torchvision"
                )
            else:
                status_report["performance_notes"].append(
                    f"MPS acceleration active on {hardware.chip_generation.value if hardware.chip_generation else 'Apple Silicon'}"
                )

        # Memory warnings
        if hardware.memory_gb < 8:
            status_report["warnings"].append(
                {
                    "level": "warning",
                    "message": "Low system memory detected",
                    "impact": "May experience out-of-memory errors with large codebases",
                    "suggestion": "Close other applications or use smaller batch sizes",
                }
            )

        # Performance expectations
        if hardware.device_type == DeviceType.MPS:
            if hardware.chip_generation and hardware.chip_generation.value.startswith(
                "m3"
            ):
                status_report["performance_notes"].append(
                    "Excellent performance expected (M3 optimized)"
                )
            elif hardware.chip_generation and hardware.chip_generation.value.startswith(
                "m2"
            ):
                status_report["performance_notes"].append(
                    "Good performance expected (M2 optimized)"
                )
            else:
                status_report["performance_notes"].append(
                    "Adequate performance expected"
                )
        elif hardware.device_type == DeviceType.CPU:
            status_report["performance_notes"].append(
                "Slower performance - CPU processing only"
            )

        return status_report
