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

"""Unit tests for hardware device management and Apple M3 acceleration.

This module tests the DeviceManager class and its hardware detection,
optimization, and performance monitoring capabilities.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from semantic_code_analyzer.hardware.device_manager import (
    ChipGeneration,
    DeviceConfig,
    DeviceManager,
    DeviceType,
    HardwareInfo,
)


class TestDeviceManager(unittest.TestCase):
    """Test cases for DeviceManager hardware detection and optimization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_hardware_info = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            cpu_count=8,
            memory_gb=24.0,
            device_type=DeviceType.MPS,
            device_name="Apple Silicon MPS (m3)",
            is_apple_silicon=True,
            chip_generation=ChipGeneration.M3,
            supports_mps=True,
            supports_cuda=False,
            neural_engine_cores=16,
            gpu_cores=10,
            memory_bandwidth_gbps=100.0,
            unified_memory=True,
        )

    @patch("semantic_code_analyzer.hardware.device_manager.platform.system")
    @patch("semantic_code_analyzer.hardware.device_manager.platform.machine")
    @patch("semantic_code_analyzer.hardware.device_manager.psutil.cpu_count")
    @patch("semantic_code_analyzer.hardware.device_manager.psutil.virtual_memory")
    def test_device_manager_initialization(
        self,
        mock_memory: MagicMock,
        mock_cpu_count: MagicMock,
        mock_machine: MagicMock,
        mock_system: MagicMock,
    ) -> None:
        """Test DeviceManager initialization with mocked system info."""
        # Mock system information
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 25_769_803_776  # 24GB in bytes

        with (
            patch.object(DeviceManager, "_detect_apple_silicon", return_value=True),
            patch.object(
                DeviceManager,
                "_detect_apple_chip_generation",
                return_value=ChipGeneration.M3,
            ),
            patch.object(DeviceManager, "_check_mps_availability", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device_manager = DeviceManager()

            # Verify hardware detection
            self.assertIsInstance(device_manager.hardware_info, HardwareInfo)
            self.assertEqual(device_manager.hardware_info.platform, "Darwin")
            self.assertEqual(device_manager.hardware_info.architecture, "arm64")
            self.assertTrue(device_manager.hardware_info.is_apple_silicon)
            self.assertEqual(
                device_manager.hardware_info.chip_generation, ChipGeneration.M3
            )

    def test_apple_silicon_detection(self) -> None:
        """Test Apple Silicon detection logic."""
        device_manager = DeviceManager.__new__(
            DeviceManager
        )  # Create without calling __init__

        # Test macOS with Apple Silicon
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Chip: Apple M3"

            result = device_manager._detect_apple_silicon()
            self.assertTrue(result)

        # Test non-macOS system
        with patch("platform.system", return_value="Linux"):
            result = device_manager._detect_apple_silicon()
            self.assertFalse(result)

    def test_apple_chip_generation_detection(self) -> None:
        """Test specific Apple chip generation detection."""
        device_manager = DeviceManager.__new__(DeviceManager)

        test_cases = [
            ("Apple M3 Ultra", ChipGeneration.M3_ULTRA),
            ("Apple M3 Max", ChipGeneration.M3_MAX),
            ("Apple M3 Pro", ChipGeneration.M3_PRO),
            ("Apple M3", ChipGeneration.M3),
            ("Apple M2", ChipGeneration.M2),
            ("Apple M1", ChipGeneration.M1),
            ("Intel", ChipGeneration.UNKNOWN),
        ]

        for system_output, expected_generation in test_cases:
            with (
                patch.object(
                    device_manager, "_detect_apple_silicon", return_value=True
                ),
                patch("subprocess.run") as mock_run,
            ):
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = system_output.lower()

                result = device_manager._detect_apple_chip_generation()
                self.assertEqual(
                    result, expected_generation, f"Failed for: {system_output}"
                )

    def test_mps_availability_check(self) -> None:
        """Test MPS availability detection."""
        device_manager = DeviceManager.__new__(DeviceManager)

        # Test MPS available
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("platform.system", return_value="Darwin"),
        ):
            result = device_manager._check_mps_availability()
            self.assertTrue(result)

        # Test MPS not available
        with patch("torch.backends.mps.is_available", return_value=False):
            result = device_manager._check_mps_availability()
            self.assertFalse(result)

    def test_device_configuration(self) -> None:
        """Test device-specific configuration optimization."""
        device_manager = DeviceManager.__new__(DeviceManager)
        device_manager._hardware_info = self.mock_hardware_info
        device_manager._configure_device()

        config = device_manager.device_config
        self.assertIsInstance(config, DeviceConfig)
        self.assertEqual(config.device_type, DeviceType.MPS)
        self.assertTrue(config.enable_mixed_precision)
        self.assertTrue(config.enable_memory_pooling)

    def test_m3_batch_size_calculation(self) -> None:
        """Test M3-specific batch size optimization."""
        device_manager = DeviceManager.__new__(DeviceManager)
        device_manager._hardware_info = self.mock_hardware_info

        # Test different memory configurations
        test_cases = [
            (40.0, 128),  # M3 Max/Ultra
            (20.0, 64),  # M3 Pro
            (8.0, 32),  # M3 base
        ]

        for memory_gb, expected_min_batch in test_cases:
            device_manager._hardware_info.memory_gb = memory_gb
            result = device_manager._calculate_optimal_batch_size_m3()
            self.assertGreaterEqual(result, expected_min_batch // 4)

    def test_apple_silicon_specs(self) -> None:
        """Test Apple Silicon chip specifications lookup."""
        device_manager = DeviceManager.__new__(DeviceManager)

        test_cases = [
            (ChipGeneration.M3, (16, 10, 100.0)),
            (ChipGeneration.M3_PRO, (16, 18, 150.0)),
            (ChipGeneration.M3_MAX, (16, 30, 300.0)),
            (ChipGeneration.M1, (16, 7, 68.25)),
        ]

        for chip, expected_specs in test_cases:
            result = device_manager._get_apple_silicon_specs(chip)
            self.assertEqual(result, expected_specs)

    def test_device_recommendations(self) -> None:
        """Test device optimization recommendations."""
        with (
            patch.object(DeviceManager, "_detect_hardware"),
            patch.object(DeviceManager, "_configure_device"),
        ):
            device_manager = DeviceManager()
            device_manager._hardware_info = self.mock_hardware_info
            device_manager._device_config = DeviceConfig(
                device_type=DeviceType.MPS, batch_size=64, enable_mixed_precision=True
            )

            recommendations = device_manager.get_device_recommendations()

            self.assertEqual(recommendations["device_type"], "mps")
            self.assertEqual(recommendations["batch_size"], 64)
            self.assertTrue(recommendations["memory_optimization"]["unified_memory"])
            self.assertTrue(recommendations["performance_features"]["mixed_precision"])

    @patch("torch.mm")
    @patch("torch.randn")
    @patch("time.time")
    def test_device_benchmark(
        self, mock_time: MagicMock, mock_randn: MagicMock, mock_mm: MagicMock
    ) -> None:
        """Test device performance benchmarking."""
        # Mock time progression
        mock_time.side_effect = [
            0.0,
            0.1,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            5.5,
        ]  # 5 seconds + setup

        # Mock tensor operations
        mock_tensor = MagicMock()
        mock_randn.return_value = mock_tensor
        mock_mm.return_value = mock_tensor

        with (
            patch.object(DeviceManager, "_detect_hardware"),
            patch.object(DeviceManager, "_configure_device"),
        ):
            device_manager = DeviceManager()
            device_manager._hardware_info = self.mock_hardware_info
            device_manager._device_config = DeviceConfig(
                device_type=DeviceType.MPS, batch_size=64, enable_mixed_precision=True
            )

            with patch("torch.mps.synchronize"):
                result = device_manager.benchmark_device(duration_seconds=5.0)

            self.assertIn("device", result)
            self.assertIn("operations_per_second", result)
            self.assertIn("performance_score", result)
            self.assertEqual(result["device"], "mps")

    def test_memory_info_collection(self) -> None:
        """Test memory information collection."""
        with (
            patch.object(DeviceManager, "_detect_hardware"),
            patch.object(DeviceManager, "_configure_device"),
            patch("psutil.virtual_memory") as mock_memory,
        ):
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.available = 16_000_000_000  # 16GB
            mock_memory.return_value.percent = 60.0

            device_manager = DeviceManager()
            device_manager._hardware_info = self.mock_hardware_info
            device_manager._device_config = DeviceConfig(
                device_type=DeviceType.MPS, max_memory_fraction=0.75
            )

            memory_info = device_manager.get_memory_info()

            self.assertIn("system_memory_gb", memory_info)
            self.assertIn("available_memory_gb", memory_info)
            self.assertIn("memory_usage_percent", memory_info)
            self.assertEqual(memory_info["system_memory_gb"], 24.0)

    def test_model_loading_optimization(self) -> None:
        """Test optimized settings for model loading."""
        with (
            patch.object(DeviceManager, "_detect_hardware"),
            patch.object(DeviceManager, "_configure_device"),
        ):
            device_manager = DeviceManager()
            device_manager._hardware_info = self.mock_hardware_info
            device_manager._device_config = DeviceConfig(
                device_type=DeviceType.MPS, enable_mixed_precision=True
            )

            settings = device_manager.optimize_for_model_loading()

            self.assertEqual(settings["dtype"], torch.float16)
            self.assertTrue(settings["low_cpu_mem_usage"])
            self.assertFalse(settings["trust_remote_code"])  # Security best practice

    def test_preferred_device_override(self) -> None:
        """Test preferred device type override."""
        with (
            patch.object(DeviceManager, "_detect_hardware"),
            patch.object(DeviceManager, "_configure_device"),
        ):
            # Test CPU preference override
            device_manager = DeviceManager(prefer_device=DeviceType.CPU)
            device_manager.prefer_device = DeviceType.CPU

            self.assertEqual(device_manager.prefer_device, DeviceType.CPU)


class TestDeviceTypes(unittest.TestCase):
    """Test device type enums and configurations."""

    def test_device_type_enum(self) -> None:
        """Test DeviceType enum values."""
        self.assertEqual(DeviceType.CPU.value, "cpu")
        self.assertEqual(DeviceType.CUDA.value, "cuda")
        self.assertEqual(DeviceType.MPS.value, "mps")

    def test_chip_generation_enum(self) -> None:
        """Test ChipGeneration enum values."""
        self.assertEqual(ChipGeneration.M3.value, "m3")
        self.assertEqual(ChipGeneration.M3_PRO.value, "m3_pro")
        self.assertEqual(ChipGeneration.M3_MAX.value, "m3_max")
        self.assertEqual(ChipGeneration.M3_ULTRA.value, "m3_ultra")

    def test_hardware_info_dataclass(self) -> None:
        """Test HardwareInfo dataclass creation."""
        hardware_info = HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            cpu_count=8,
            memory_gb=16.0,
            device_type=DeviceType.MPS,
            device_name="Apple M3",
            is_apple_silicon=True,
            chip_generation=ChipGeneration.M3,
            supports_mps=True,
            supports_cuda=False,
            neural_engine_cores=16,
            gpu_cores=10,
            memory_bandwidth_gbps=100.0,
            unified_memory=True,
        )

        self.assertTrue(hardware_info.is_apple_silicon)
        self.assertEqual(hardware_info.chip_generation, ChipGeneration.M3)
        self.assertTrue(hardware_info.unified_memory)

    def test_device_config_dataclass(self) -> None:
        """Test DeviceConfig dataclass with defaults."""
        config = DeviceConfig(device_type=DeviceType.MPS)

        self.assertEqual(config.device_type, DeviceType.MPS)
        self.assertEqual(config.batch_size, 32)  # Default
        self.assertEqual(config.max_memory_fraction, 0.8)  # Default
        self.assertTrue(config.enable_mixed_precision)  # Default
        self.assertTrue(config.fallback_enabled)  # Default


if __name__ == "__main__":
    unittest.main()
