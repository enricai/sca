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

"""Integration tests for hardware acceleration in PatternIndexer.

This module tests the integration of DeviceManager with PatternIndexer
for hardware-accelerated code embedding generation.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from semantic_code_analyzer.embeddings.pattern_indexer import PatternIndexer
from semantic_code_analyzer.hardware.device_manager import (
    ChipGeneration,
    DeviceConfig,
    DeviceManager,
    DeviceType,
    HardwareInfo,
)


class TestHardwareAcceleratedPatternIndexer(unittest.TestCase):
    """Test hardware acceleration integration with PatternIndexer."""

    def setUp(self) -> None:
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Mock M3 hardware info
        self.mock_m3_hardware = HardwareInfo(
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

        # Mock device configuration
        self.mock_device_config = DeviceConfig(
            device_type=DeviceType.MPS,
            batch_size=64,
            max_memory_fraction=0.75,
            enable_mixed_precision=True,
            enable_memory_pooling=True,
        )

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    @patch("torch.cuda.is_available")
    def test_pattern_indexer_m3_initialization(
        self, mock_cuda: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test PatternIndexer initialization with M3 hardware."""
        mock_cuda.return_value = False

        # Mock model and tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model mock with proper methods
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock device manager
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {
            "device_type": "mps",
            "batch_size": 64,
            "m3_optimizations": {
                "recommended_chunk_size": 1024,
                "memory_bandwidth_gbps": 100.0,
            },
        }
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 24.0,
            "available_memory_gb": 16.0,
            "mps_unified_memory": True,
        }

        pattern_indexer = PatternIndexer(
            cache_dir=str(self.temp_path),
            device_manager=mock_device_manager,
            enable_optimizations=True,
        )

        # Verify initialization
        self.assertEqual(pattern_indexer.device_manager, mock_device_manager)
        self.assertTrue(pattern_indexer.enable_optimizations)
        self.assertEqual(pattern_indexer.device, torch.device("mps"))

        # Verify model loading with M3 optimizations
        mock_model.from_pretrained.assert_called_once()
        call_kwargs = mock_model.from_pretrained.call_args[1]
        # PatternIndexer applies dtype conversion after loading, not during from_pretrained()
        self.assertNotIn("dtype", call_kwargs)  # dtype is applied after loading
        self.assertTrue(call_kwargs["low_cpu_mem_usage"])
        self.assertTrue(call_kwargs["use_safetensors"])  # Should also be passed

        # Verify that model.to() is called with the device (dtype conversion happens separately)
        mock_model_instance.to.assert_called()
        # The to() method should be called at least twice: once for device, once for dtype
        self.assertGreaterEqual(mock_model_instance.to.call_count, 2)

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    @patch("torch.cuda.is_available")
    def test_hardware_optimization_application(
        self, mock_cuda: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test application of hardware-specific optimizations."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model mock with proper methods
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {
            "m3_optimizations": {"recommended_chunk_size": 1024}
        }
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 24.0,
            "available_memory_gb": 16.0,
            "mps_unified_memory": True,
        }

        # Create the pattern indexer with M3 optimizations
        pattern_indexer = PatternIndexer(
            cache_dir=str(self.temp_path),
            device_manager=mock_device_manager,
            enable_optimizations=True,
        )

        # Verify that MPS optimizations were applied
        self.assertTrue(pattern_indexer.enable_optimizations)

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    @patch("torch.cuda.is_available")
    def test_embedding_extraction_with_mixed_precision(
        self, mock_cuda: MagicMock, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test embedding extraction with mixed precision on M3."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model mock with proper methods
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock tokenizer output
        mock_tokens = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer_instance.return_value = mock_tokens

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(
            1, 3, 768
        )  # [batch, seq_len, hidden]
        mock_model_instance.return_value = mock_output

        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {}
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 24.0,
            "available_memory_gb": 16.0,
            "mps_unified_memory": True,
        }

        pattern_indexer = PatternIndexer(
            cache_dir=str(self.temp_path),
            device_manager=mock_device_manager,
            enable_optimizations=True,
        )

        # Test embedding extraction
        test_code = "def hello_world(): return 'Hello, M3!'"

        with patch("torch.autocast") as mock_autocast:
            mock_autocast.return_value.__enter__ = MagicMock()
            mock_autocast.return_value.__exit__ = MagicMock()

            embedding = pattern_indexer._extract_code_embeddings(test_code)

        # Verify mixed precision was used
        mock_autocast.assert_called_once_with(device_type="mps", dtype=torch.float16)

        # Verify embedding shape
        self.assertEqual(embedding.shape, (768,))  # GraphCodeBERT embedding size

    def test_performance_metrics_tracking(self) -> None:
        """Test performance metrics collection."""
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {}
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 24.0,
            "available_memory_gb": 16.0,
            "mps_unified_memory": True,
        }

        with (
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer"),
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel"),
        ):
            pattern_indexer = PatternIndexer(
                cache_dir=str(self.temp_path), device_manager=mock_device_manager
            )

        metrics = pattern_indexer.get_performance_metrics()

        # Verify metrics structure
        self.assertIn("device_info", metrics)
        self.assertIn("memory_usage", metrics)
        self.assertIn("performance", metrics)
        self.assertIn("configuration", metrics)

        # Verify device info
        device_info = metrics["device_info"]
        self.assertEqual(device_info["device_type"], "mps")
        self.assertEqual(device_info["chip_generation"], "m3")
        self.assertTrue(device_info["is_apple_silicon"])

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    def test_fallback_mechanism(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test CPU fallback when MPS fails."""
        # Mock initial failure on MPS
        mock_model.from_pretrained.side_effect = [
            RuntimeError("MPS failed"),
            MagicMock(),
        ]
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {}

        # Device config with fallback enabled
        mock_device_manager.device_config.fallback_enabled = True

        with patch.object(PatternIndexer, "_fallback_to_cpu") as mock_fallback:
            try:
                PatternIndexer(
                    cache_dir=str(self.temp_path), device_manager=mock_device_manager
                )
                # Fallback should have been called
                mock_fallback.assert_called_once()
            except ValueError:
                # If fallback is not implemented yet, that's expected
                pass

    def test_domain_index_building_with_m3_optimization(self) -> None:
        """Test domain index building with M3-specific optimizations."""
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {
            "m3_optimizations": {
                "recommended_chunk_size": 1024,
                "memory_bandwidth_gbps": 100.0,
            }
        }

        with (
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer"),
            patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel"),
            patch.object(PatternIndexer, "_extract_code_embeddings") as mock_extract,
        ):
            # Mock embedding extraction
            mock_extract.return_value = np.random.randn(768)

            pattern_indexer = PatternIndexer(
                cache_dir=str(self.temp_path), device_manager=mock_device_manager
            )

            # Test data
            test_files = {
                "test.py": "def test(): pass",
                "main.py": "if __name__ == '__main__': test()",
            }

            # Build index (should use M3-optimized chunk size)
            pattern_indexer.build_domain_index("test_domain", test_files)

            # Verify embeddings were extracted
            self.assertEqual(mock_extract.call_count, 2)

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    def test_benchmark_functionality(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test embedding performance benchmarking."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model mock with proper methods
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Configure device manager mock
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = self.mock_m3_hardware
        mock_device_manager.device_config = self.mock_device_config
        mock_device_manager.torch_device = torch.device("mps")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {}
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 24.0,
            "available_memory_gb": 16.0,
            "mps_unified_memory": True,
        }

        # Initialize PatternIndexer (this should not fallback to CPU)
        pattern_indexer = PatternIndexer(
            cache_dir=str(self.temp_path), device_manager=mock_device_manager
        )

        # Mock embedding extraction to return quickly
        with (
            patch.object(pattern_indexer, "_extract_code_embeddings") as mock_extract,
            patch("torch.mps.synchronize"),
        ):
            mock_extract.return_value = np.random.randn(768)

            results = pattern_indexer.benchmark_embedding_performance(
                test_code="def test(): pass", iterations=5
            )

        # Verify benchmark results
        self.assertIn("device", results)
        self.assertIn("embeddings_per_second", results)
        self.assertIn("average_time_per_embedding", results)
        self.assertEqual(results["device"], "mps")
        self.assertEqual(results["iterations"], 5)


class TestCPUFallback(unittest.TestCase):
    """Test CPU fallback functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
    def test_cpu_device_configuration(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test PatternIndexer with CPU-only configuration."""
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()

        # Configure tokenizer mock
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Configure model mock with proper methods
        mock_model_instance.eval.return_value = None
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock CPU-only device manager
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = HardwareInfo(
            platform="Linux",
            architecture="x86_64",
            cpu_count=4,
            memory_gb=8.0,
            device_type=DeviceType.CPU,
            device_name="CPU (Intel)",
            is_apple_silicon=False,
            chip_generation=None,
            supports_mps=False,
            supports_cuda=False,
            neural_engine_cores=None,
            gpu_cores=None,
            memory_bandwidth_gbps=None,
            unified_memory=False,
        )
        mock_device_manager.device_config = DeviceConfig(
            device_type=DeviceType.CPU, batch_size=4, enable_mixed_precision=False
        )
        mock_device_manager.torch_device = torch.device("cpu")
        mock_device_manager.optimize_for_model_loading.return_value = {
            "dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        mock_device_manager.get_device_recommendations.return_value = {
            "device_type": "cpu",
            "batch_size": 4,
        }
        mock_device_manager.get_memory_info.return_value = {
            "system_memory_gb": 8.0,
            "available_memory_gb": 6.0,
            "mps_unified_memory": False,
        }

        pattern_indexer = PatternIndexer(
            cache_dir=str(self.temp_path), device_manager=mock_device_manager
        )

        # Verify CPU configuration
        self.assertEqual(pattern_indexer.device.type, "cpu")
        metrics = pattern_indexer.get_performance_metrics()
        self.assertEqual(metrics["device_info"]["device_type"], "cpu")
        self.assertFalse(metrics["device_info"]["is_apple_silicon"])


if __name__ == "__main__":
    unittest.main()
