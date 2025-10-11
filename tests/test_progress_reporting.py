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

"""
Tests for progress reporting functionality.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from semantic_code_analyzer import EnhancedScorerConfig, MultiDimensionalScorer
from semantic_code_analyzer.analyzers.domain_adherence_analyzer import (
    DomainAwareAdherenceAnalyzer,
)
from semantic_code_analyzer.embeddings.pattern_indexer import PatternIndexer
from semantic_code_analyzer.hardware import DeviceManager


class TestProgressReporting:
    """Tests for progress reporting functionality."""

    def test_multidimensional_scorer_progress_callback(self) -> None:
        """Test that MultiDimensionalScorer calls progress callback during initialization."""
        # Mock the git repository
        with patch("semantic_code_analyzer.scorers.multi_dimensional_scorer.git.Repo"):
            # Track progress messages
            progress_messages = []

            def progress_callback(message: str) -> None:
                progress_messages.append(message)

            config = EnhancedScorerConfig(
                enable_domain_adherence_analysis=False,  # Disable to avoid model loading
            )

            # Create scorer with progress callback
            MultiDimensionalScorer(config=config, progress_callback=progress_callback)

            # Verify progress messages were called
            assert len(progress_messages) > 0

            # Check expected progress messages
            expected_messages = [
                "Setting up repository connection...",
                "Initializing scoring components...",
                "Initializing architectural analyzer...",
                "Initializing code quality analyzer...",
                "Initializing TypeScript analyzer...",
                "Initializing framework analyzer...",
                "Analyzers initialized successfully!",
            ]

            for expected in expected_messages:
                assert any(
                    expected in msg for msg in progress_messages
                ), f"Expected message '{expected}' not found in progress messages: {progress_messages}"

    @patch("semantic_code_analyzer.analyzers.domain_adherence_analyzer.PatternIndexer")
    def test_domain_adherence_analyzer_progress_callback(
        self, mock_pattern_indexer: MagicMock
    ) -> None:
        """Test that DomainAwareAdherenceAnalyzer calls progress callback during initialization."""
        # Mock the PatternIndexer to avoid model loading
        mock_pattern_indexer_instance = MagicMock()
        mock_pattern_indexer.return_value = mock_pattern_indexer_instance

        # Track progress messages
        progress_messages = []

        def progress_callback(message: str) -> None:
            progress_messages.append(message)

        config = {
            "model_name": "Qodo/Qodo-Embed-1-1.5B",
            "similarity_threshold": 0.3,
        }

        # Create analyzer with progress callback
        mock_device_manager = MagicMock(spec=DeviceManager)
        DomainAwareAdherenceAnalyzer(
            config=config,
            device_manager=mock_device_manager,
            progress_callback=progress_callback,
        )

        # Verify progress messages were called
        assert len(progress_messages) > 0

        # Check expected progress messages
        expected_messages = [
            "Initializing base analyzer...",
            "Initializing domain classifier...",
            "Checking ML model dependencies...",
            "Loading code embedding model (this may take a while)...",
            "Finalizing analyzer configuration...",
            "Domain adherence analyzer ready!",
        ]

        for expected in expected_messages:
            assert any(
                expected in msg for msg in progress_messages
            ), f"Expected message '{expected}' not found in progress messages: {progress_messages}"

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.AutoTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.AutoModel")
    def test_pattern_indexer_progress_callback(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test that PatternIndexer calls progress callback during initialization."""
        # Mock the tokenizer and model to avoid actual loading
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Track progress messages
        progress_messages = []

        def progress_callback(message: str) -> None:
            progress_messages.append(message)

        # Mock device manager
        mock_device_manager = MagicMock(spec=DeviceManager)
        mock_device_manager.hardware_info = MagicMock()
        mock_device_manager.hardware_info.device_name = "CPU"
        mock_device_manager.hardware_info.device_type = MagicMock()
        mock_device_manager.hardware_info.memory_gb = 16.0
        mock_device_manager.torch_device = MagicMock()
        mock_device_manager.optimize_for_model_loading.return_value = {
            "low_cpu_mem_usage": True,
            "dtype": MagicMock(),
        }
        mock_device_manager.get_device_recommendations.return_value = {}

        # Create pattern indexer with progress callback
        with patch.object(PatternIndexer, "_validate_mps_compatibility"):
            with patch.object(PatternIndexer, "_apply_hardware_optimizations"):
                PatternIndexer(
                    model_name="Qodo/Qodo-Embed-1-1.5B",
                    device_manager=mock_device_manager,
                    progress_callback=progress_callback,
                )

        # Verify progress messages were called
        assert len(progress_messages) > 0

        # Check expected progress messages
        expected_messages = [
            "Setting up cache directory...",
            "Initializing hardware acceleration...",
            "Configuring hardware optimizations...",
            "Validating device compatibility...",
            "Loading model tokenizer...",
            "Loading embedding model (this is the slow step)...",
            "Configuring model settings...",
            "Setting up device configuration...",
            "Pattern indexer ready!",
        ]

        for expected in expected_messages:
            assert any(
                expected in msg for msg in progress_messages
            ), f"Expected message '{expected}' not found in progress messages: {progress_messages}"

    def test_progress_callback_is_optional(self) -> None:
        """Test that components work correctly when progress callback is None."""
        # Mock the git repository
        with patch("semantic_code_analyzer.scorers.multi_dimensional_scorer.git.Repo"):
            config = EnhancedScorerConfig(
                enable_domain_adherence_analysis=False,  # Disable to avoid model loading
            )

            # Create scorer without progress callback (should not raise errors)
            scorer = MultiDimensionalScorer(config=config, progress_callback=None)

            # Verify scorer was created successfully
            assert scorer is not None
            assert scorer.progress_callback is None

    def test_nested_progress_callbacks(self) -> None:
        """Test that nested progress callbacks work correctly."""
        # Mock the git repository
        with patch("semantic_code_analyzer.scorers.multi_dimensional_scorer.git.Repo"):
            # Mock DomainAwareAdherenceAnalyzer to capture nested callbacks
            with patch(
                "semantic_code_analyzer.scorers.multi_dimensional_scorer.DomainAwareAdherenceAnalyzer"
            ) as mock_analyzer:
                # Track progress messages
                progress_messages = []

                def progress_callback(message: str) -> None:
                    progress_messages.append(message)

                config = EnhancedScorerConfig()

                # Create scorer with progress callback
                MultiDimensionalScorer(
                    config=config, progress_callback=progress_callback
                )

                # Verify the domain analyzer was called with a nested progress callback
                assert mock_analyzer.called
                call_args = mock_analyzer.call_args
                assert "progress_callback" in call_args.kwargs

                # Test the nested callback
                nested_callback = call_args.kwargs["progress_callback"]
                nested_callback("test message")

                # Verify the nested message was properly formatted
                assert any(
                    "Domain analyzer: test message" in msg for msg in progress_messages
                )
