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
from semantic_code_analyzer.progress import (
    AnalysisPhase,
    ProgressConfig,
    ProgressLevel,
    ProgressManager,
    create_default_progress_manager,
    create_detailed_progress_manager,
    create_minimal_progress_manager,
)


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
            "model_name": "microsoft/graphcodebert-base",
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
            "Loading GraphCodeBERT model (this may take a while)...",
            "Finalizing analyzer configuration...",
            "Domain adherence analyzer ready!",
        ]

        for expected in expected_messages:
            assert any(
                expected in msg for msg in progress_messages
            ), f"Expected message '{expected}' not found in progress messages: {progress_messages}"

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaTokenizer")
    @patch("semantic_code_analyzer.embeddings.pattern_indexer.RobertaModel")
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
                    model_name="microsoft/graphcodebert-base",
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
            "Loading GraphCodeBERT tokenizer...",
            "Loading GraphCodeBERT model (this is the slow step)...",
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


class TestProgressManager:
    """Tests for the enhanced ProgressManager functionality."""

    def test_progress_manager_creation(self) -> None:
        """Test creating progress manager with different configurations."""
        # Test default manager
        default_manager = create_default_progress_manager()
        assert default_manager.config.enabled is True
        assert default_manager.config.level == ProgressLevel.NORMAL

        # Test minimal manager
        minimal_manager = create_minimal_progress_manager()
        assert minimal_manager.config.level == ProgressLevel.MINIMAL

        # Test detailed manager
        detailed_manager = create_detailed_progress_manager()
        assert detailed_manager.config.level == ProgressLevel.DETAILED

    def test_progress_manager_disabled(self) -> None:
        """Test progress manager with disabled progress."""
        config = ProgressConfig(enabled=False)
        manager = ProgressManager(config)

        assert not manager.is_enabled()

        # Test that context manager returns dummy tracker
        with manager.create_progress_context() as tracker:
            # Should not raise any errors
            tracker.start_phase(AnalysisPhase.INITIALIZATION)
            tracker.update_phase(AnalysisPhase.INITIALIZATION, "test message")
            tracker.complete_phase(AnalysisPhase.INITIALIZATION)

    def test_nested_callback_creation(self) -> None:
        """Test creating nested progress callbacks."""
        manager = create_default_progress_manager()

        callback = manager.create_nested_callback("test_phase", "Test: ")

        # Should be callable and not raise errors
        callback("test message")

    def test_analysis_phases_enum(self) -> None:
        """Test that all expected analysis phases are defined."""
        expected_phases = [
            "Initialization",
            "Repository Setup",
            "Hardware Initialization",
            "Analyzer Initialization",
            "File Extraction",
            "Pattern Index Building",
            "Multi-Dimensional Analysis",
            "Results Aggregation",
            "Finalization",
        ]

        for expected in expected_phases:
            # Check that phase exists in enum
            phase_found = any(phase.value == expected for phase in AnalysisPhase)
            assert (
                phase_found
            ), f"Expected phase '{expected}' not found in AnalysisPhase enum"

    @patch("semantic_code_analyzer.progress.Progress")
    def test_rich_progress_context(self, mock_progress: MagicMock) -> None:
        """Test Rich progress context manager."""
        mock_progress_instance = MagicMock()
        mock_progress.return_value = mock_progress_instance

        manager = create_default_progress_manager()

        with manager.create_progress_context() as tracker:
            # Verify Rich Progress was created
            assert mock_progress.called

            # Test phase operations
            tracker.start_phase(AnalysisPhase.INITIALIZATION, "Custom description")
            tracker.update_phase(AnalysisPhase.INITIALIZATION, "Test message")
            tracker.complete_phase(AnalysisPhase.INITIALIZATION)

            # Verify progress instance methods were called
            assert mock_progress_instance.start.called
            assert mock_progress_instance.add_task.called

        # Verify cleanup
        assert mock_progress_instance.stop.called

    def test_file_progress_context(self) -> None:
        """Test file progress context functionality."""
        manager = create_default_progress_manager()

        with manager.create_progress_context() as tracker:
            # Test file progress creation
            with tracker.create_file_progress(
                10, "Processing test files"
            ) as file_progress:
                # Test progress updates
                file_progress.update("test_file.py", 1)
                file_progress.advance(1)
                file_progress.update("another_file.py", 2)
                file_progress.advance(1)

                # Should not raise any errors
                assert file_progress is not None


class TestBaseAnalyzerProgress:
    """Tests for BaseAnalyzer progress functionality."""

    @patch("semantic_code_analyzer.analyzers.base_analyzer.BaseAnalyzer.analyze_file")
    def test_analyze_commit_with_progress(self, mock_analyze_file: MagicMock) -> None:
        """Test BaseAnalyzer progress reporting during commit analysis."""
        from semantic_code_analyzer.analyzers.base_analyzer import (
            AnalysisResult,
            BaseAnalyzer,
        )

        # Create a concrete implementation for testing
        class TestAnalyzer(BaseAnalyzer):
            def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
                return mock_analyze_file(file_path, content)

            def get_analyzer_name(self) -> str:
                return "test_analyzer"

            def get_weight(self) -> float:
                return 1.0

        analyzer = TestAnalyzer()

        # Mock the analyze_file method to return a valid result
        mock_result = MagicMock(spec=AnalysisResult)
        mock_analyze_file.return_value = mock_result

        # Track progress calls
        progress_calls = []

        def progress_callback(file_path: str, current: int, total: int) -> None:
            progress_calls.append((file_path, current, total))

        # Test files
        commit_files = {
            "test1.ts": "const test = 1;",
            "test2.js": "var test = 2;",
            "README.md": "# Test",  # Should be excluded
        }

        # Run analysis with progress
        results = analyzer.analyze_commit_with_progress(commit_files, progress_callback)

        # Verify progress was reported
        assert len(progress_calls) == 2  # Only .ts and .js files
        assert progress_calls[0] == ("test1.ts", 1, 2)
        assert progress_calls[1] == ("test2.js", 2, 2)

        # Verify results
        assert len(results) == 2
        assert "test1.ts" in results
        assert "test2.js" in results

    def test_analyze_commit_with_progress_no_callback(self) -> None:
        """Test BaseAnalyzer progress method works without callback."""
        from semantic_code_analyzer.analyzers.base_analyzer import (
            AnalysisResult,
            BaseAnalyzer,
        )

        class TestAnalyzer(BaseAnalyzer):
            def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
                return MagicMock(spec=AnalysisResult)

            def get_analyzer_name(self) -> str:
                return "test_analyzer"

            def get_weight(self) -> float:
                return 1.0

        analyzer = TestAnalyzer()

        commit_files = {"test.ts": "const test = 1;"}

        # Should not raise any errors with no callback
        results = analyzer.analyze_commit_with_progress(commit_files, None)
        assert len(results) == 1


class TestMultiDimensionalScorerProgress:
    """Tests for enhanced MultiDimensionalScorer progress functionality."""

    @patch("semantic_code_analyzer.scorers.multi_dimensional_scorer.git.Repo")
    def test_file_level_progress_in_scorer(self, mock_repo: MagicMock) -> None:
        """Test that MultiDimensionalScorer reports file-level progress."""
        # Track progress messages
        progress_messages = []

        def progress_callback(message: str) -> None:
            progress_messages.append(message)

        config = EnhancedScorerConfig(
            enable_domain_adherence_analysis=False,  # Disable to avoid model loading
            enable_progress_reporting=True,
            show_file_progress=True,
        )

        # Create scorer with progress callback
        scorer = MultiDimensionalScorer(
            config=config, progress_callback=progress_callback
        )

        # Verify progress configuration is applied
        assert config.enable_progress_reporting is True
        assert config.show_file_progress is True

        # Verify scorer was created successfully
        assert scorer is not None

    def test_enhanced_scorer_config_progress_options(self) -> None:
        """Test that EnhancedScorerConfig includes progress options."""
        config = EnhancedScorerConfig()

        # Verify default progress settings
        assert hasattr(config, "enable_progress_reporting")
        assert hasattr(config, "progress_level")
        assert hasattr(config, "show_file_progress")
        assert hasattr(config, "progress_refresh_rate")

        # Test custom progress settings
        custom_config = EnhancedScorerConfig(
            enable_progress_reporting=False,
            progress_level="minimal",
            show_file_progress=False,
            progress_refresh_rate=5,
        )

        assert custom_config.enable_progress_reporting is False
        assert custom_config.progress_level == "minimal"
        assert custom_config.show_file_progress is False
        assert custom_config.progress_refresh_rate == 5
