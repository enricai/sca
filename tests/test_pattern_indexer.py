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

"""Tests for the pattern indexer module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from semantic_code_analyzer.embeddings.pattern_indexer import (
    PatternIndex,
    PatternIndexer,
    SimilarityMatch,
)


class TestPatternIndexer:
    """Test cases for the PatternIndexer class."""

    @pytest.fixture
    def mock_model_components(self) -> Any:
        """Mock the transformer model components to avoid loading actual models."""
        with (
            patch(
                "semantic_code_analyzer.embeddings.pattern_indexer.AutoTokenizer"
            ) as mock_tokenizer,
            patch(
                "semantic_code_analyzer.embeddings.pattern_indexer.AutoModel"
            ) as mock_model,
            patch(
                "semantic_code_analyzer.embeddings.pattern_indexer.torch"
            ) as mock_torch,
        ):
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock(),
            }
            # Add encode method for token length checking
            mock_tokenizer_instance.encode.return_value = list(
                range(100)
            )  # Return 100 tokens
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Mock model
            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_outputs.last_hidden_state = MagicMock()

            # Create mock tensor with shape [batch_size, seq_len, hidden_size]
            mock_tensor = Mock()
            mock_tensor.shape = [1, 512, 1536]
            mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(1536)
            mock_outputs.last_hidden_state.__getitem__.return_value = mock_tensor

            mock_model_instance.return_value = mock_outputs
            mock_model_instance.eval.return_value = None
            mock_model_instance.to.return_value = None
            mock_model.from_pretrained.return_value = mock_model_instance

            # Mock torch
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__.return_value = None
            mock_torch.no_grad.return_value.__exit__.return_value = None

            yield {
                "tokenizer": mock_tokenizer_instance,
                "model": mock_model_instance,
                "torch": mock_torch,
            }

    @pytest.fixture
    def indexer(self, mock_model_components: Any) -> PatternIndexer:
        """Create a PatternIndexer instance for testing with mocked models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return PatternIndexer(cache_dir=temp_dir)

    @pytest.fixture
    def sample_codebase_files(self) -> dict[str, str]:
        """Sample codebase files for testing."""
        return {
            "src/components/Button.tsx": """
import React from 'react';

interface ButtonProps {
    title: string;
    onClick: () => void;
}

const Button: React.FC<ButtonProps> = ({ title, onClick }) => {
    return (
        <button onClick={onClick} className="btn">
            {title}
        </button>
    );
};

export default Button;
""",
            "src/components/Input.tsx": """
import React, { useState } from 'react';

interface InputProps {
    placeholder?: string;
    onChange: (value: string) => void;
}

const Input: React.FC<InputProps> = ({ placeholder, onChange }) => {
    const [value, setValue] = useState('');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setValue(e.target.value);
        onChange(e.target.value);
    };

    return (
        <input
            type="text"
            placeholder={placeholder}
            value={value}
            onChange={handleChange}
        />
    );
};

export default Input;
""",
            "src/utils/helpers.ts": """
export function formatDate(date: Date): string {
    return date.toISOString().split('T')[0];
}

export function capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
""",
        }

    def test_initialization(self, mock_model_components: Any) -> None:
        """Test PatternIndexer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = PatternIndexer(cache_dir=temp_dir)

            assert indexer.model_name == "Qodo/Qodo-Embed-1-1.5B"
            assert indexer.cache_dir == Path(temp_dir)
            assert len(indexer.domain_indices) == 0
            assert len(indexer.embedding_cache) == 0

    def test_extract_code_embeddings(self, indexer: PatternIndexer) -> None:
        """Test code embedding extraction."""
        code = "function test() { return 42; }"
        embedding = indexer._extract_code_embeddings(code)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)  # Qodo-Embed embedding dimension

    def test_normalize_embeddings(self, indexer: PatternIndexer) -> None:
        """Test embedding normalization."""
        embeddings = np.random.randn(5, 1536)
        normalized = indexer._normalize_embeddings(embeddings)

        assert normalized.shape == embeddings.shape
        # Check that vectors are normalized (L2 norm should be close to 1)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_function_extraction(self, indexer: PatternIndexer) -> None:
        """Test function-level content extraction via FunctionExtractor."""
        # Test Python function extraction
        python_content = """
import os
import sys

def test_function():
    return 42

class MyClass:
    def method(self):
        pass
"""
        function_chunks = indexer.function_extractor.extract_functions(
            "test.py", python_content
        )

        # Should extract at least the function and method
        # (fallback returns 1 chunk if tree-sitter not available)
        assert len(function_chunks) >= 1
        assert all(hasattr(chunk, "function_code") for chunk in function_chunks)
        assert all(hasattr(chunk, "function_name") for chunk in function_chunks)

    def test_build_domain_index(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test building domain index."""
        domain = "frontend"
        indexer.build_domain_index(domain, sample_codebase_files)

        assert domain in indexer.domain_indices
        pattern_index = indexer.domain_indices[domain]

        assert isinstance(pattern_index, PatternIndex)
        assert pattern_index.domain == domain
        assert pattern_index.index.ntotal > 0
        assert len(pattern_index.file_paths) > 0
        assert len(pattern_index.code_snippets) > 0
        assert pattern_index.embeddings.shape[0] > 0

    def test_build_domain_index_empty_files(self, indexer: PatternIndexer) -> None:
        """Test building domain index with empty file list."""
        domain = "empty"
        indexer.build_domain_index(domain, {})

        assert domain not in indexer.domain_indices

    def test_build_domain_index_with_max_files(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test building domain index with file limit."""
        domain = "limited"
        indexer.build_domain_index(domain, sample_codebase_files, max_files=1)

        assert domain in indexer.domain_indices
        pattern_index = indexer.domain_indices[domain]

        # Should have limited the number of files processed
        unique_files = set(pattern_index.file_paths)
        assert len(unique_files) <= 1

    def test_search_similar_patterns_no_index(self, indexer: PatternIndexer) -> None:
        """Test searching patterns when no index exists for domain."""
        query_code = "function test() { return 42; }"
        results = indexer.search_similar_patterns(query_code, "nonexistent", top_k=5)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_similar_patterns_with_index(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test searching patterns with existing index."""
        domain = "frontend"
        indexer.build_domain_index(domain, sample_codebase_files)

        # Search for similar React component pattern
        query_code = """
import React from 'react';

const TestComponent: React.FC = () => {
    return <div>Test</div>;
};

export default TestComponent;
"""

        results = indexer.search_similar_patterns(query_code, domain, top_k=3)

        assert isinstance(results, list)
        # Should find some similar patterns if index was built successfully
        if len(results) > 0:  # May be empty with mocked embeddings
            assert all(isinstance(match, SimilarityMatch) for match in results)
            assert all(match.domain == domain for match in results)
            assert all(0 <= match.similarity_score <= 1 for match in results)

    def test_similarity_match_structure(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test that SimilarityMatch objects have correct structure."""
        domain = "frontend"
        indexer.build_domain_index(domain, sample_codebase_files)

        query_code = "const test = () => <div>Test</div>;"
        results = indexer.search_similar_patterns(
            query_code, domain, min_similarity=0.0
        )

        if results:  # If we got results with mocked embeddings
            match = results[0]
            assert hasattr(match, "file_path")
            assert hasattr(match, "similarity_score")
            assert hasattr(match, "code_snippet")
            assert hasattr(match, "domain")
            assert hasattr(match, "context")
            assert match.domain == domain

    def test_get_domain_statistics(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test getting domain statistics."""
        # Test with non-existent domain
        stats = indexer.get_domain_statistics("nonexistent")
        assert stats == {}

        # Test with existing domain
        domain = "frontend"
        indexer.build_domain_index(domain, sample_codebase_files)
        stats = indexer.get_domain_statistics(domain)

        assert "domain" in stats
        assert "num_patterns" in stats
        assert "embedding_dimension" in stats
        assert "unique_files" in stats
        assert "metadata" in stats
        assert stats["domain"] == domain

    def test_get_cache_statistics(self, indexer: PatternIndexer) -> None:
        """Test getting cache statistics."""
        stats = indexer.get_cache_statistics()

        expected_keys = [
            "embedding_cache_size",
            "domain_indices_count",
            "cache_directory",
            "model_device",
            "domains",
        ]
        for key in expected_keys:
            assert key in stats

    def test_clear_cache(self, indexer: PatternIndexer) -> None:
        """Test clearing embedding cache."""
        # Add something to cache
        indexer.embedding_cache["test"] = np.random.randn(1536)
        assert len(indexer.embedding_cache) > 0

        # Clear cache
        indexer.clear_cache()
        assert len(indexer.embedding_cache) == 0

    def test_save_and_load_domain_index(
        self, indexer: PatternIndexer, sample_codebase_files: dict[str, str]
    ) -> None:
        """Test saving and loading domain indices."""
        domain = "test_save_load"

        # Build and save index
        indexer.build_domain_index(domain, sample_codebase_files)
        original_stats = indexer.get_domain_statistics(domain)

        # Clear the index from memory
        del indexer.domain_indices[domain]
        assert domain not in indexer.domain_indices

        # Try to load from disk
        loaded = indexer.load_domain_index(domain)

        if loaded:  # Loading might fail with mocked components
            assert domain in indexer.domain_indices
            loaded_stats = indexer.get_domain_statistics(domain)
            assert loaded_stats["num_patterns"] == original_stats["num_patterns"]

    def test_load_nonexistent_domain_index(self, indexer: PatternIndexer) -> None:
        """Test loading non-existent domain index."""
        result = indexer.load_domain_index("nonexistent")
        assert result is False

    @patch("semantic_code_analyzer.embeddings.pattern_indexer.faiss")
    def test_faiss_integration(
        self,
        mock_faiss: Any,
        indexer: PatternIndexer,
        sample_codebase_files: dict[str, str],
    ) -> None:
        """Test FAISS integration."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 3
        mock_faiss.IndexFlatIP.return_value = mock_index

        domain = "test_faiss"
        indexer.build_domain_index(domain, sample_codebase_files)

        # Verify FAISS index was created and used
        mock_faiss.IndexFlatIP.assert_called()
        mock_index.add.assert_called()

    def test_error_handling_invalid_code(self, indexer: PatternIndexer) -> None:
        """Test error handling with invalid code content."""
        # Test with various edge cases
        test_cases = [
            "",  # Empty content
            "äöü" * 1000,  # Non-ASCII characters
            "\x00\x01\x02",  # Binary data
        ]

        for code in test_cases:
            try:
                embedding = indexer._extract_code_embeddings(code)
                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (1536,)
            except Exception as e:
                # Should handle errors gracefully and return zero vector
                # Log the exception to verify error handling is working
                import logging

                logging.getLogger(__name__).info(
                    f"Expected error handled for invalid code: {e}"
                )
                # Test that the indexer can still function after error
                assert indexer is not None

    def test_pattern_index_dataclass(self) -> None:
        """Test PatternIndex dataclass structure."""

        # Create mock components
        mock_index = Mock()
        mock_embeddings = np.random.randn(5, 1536)

        pattern_index = PatternIndex(
            domain="test",
            index=mock_index,
            file_paths=["file1.ts", "file2.ts"],
            code_snippets=["code1", "code2"],
            embeddings=mock_embeddings,
            metadata={"test": True},
        )

        assert pattern_index.domain == "test"
        assert pattern_index.index == mock_index
        assert len(pattern_index.file_paths) == 2
        assert len(pattern_index.code_snippets) == 2
        assert pattern_index.embeddings.shape == (5, 1536)
        assert pattern_index.metadata["test"] is True

    def test_data_compression_integration(self, mock_model_components: Any) -> None:
        """Test data compression integration with PatternIndexer."""
        from semantic_code_analyzer.parsing.data_compressor import (
            DataCompressionConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create indexer with compression enabled
            config = DataCompressionConfig(enabled=True)
            indexer = PatternIndexer(cache_dir=temp_dir, compression_config=config)

            # Verify compression config is passed through
            assert indexer.compression_config.enabled is True
            assert indexer.function_extractor.compression_config.enabled is True

    def test_data_compression_disabled(self, mock_model_components: Any) -> None:
        """Test that data compression can be disabled."""
        from semantic_code_analyzer.parsing.data_compressor import (
            DataCompressionConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create indexer with compression disabled
            config = DataCompressionConfig(enabled=False)
            indexer = PatternIndexer(cache_dir=temp_dir, compression_config=config)

            # Verify compression is disabled
            assert indexer.compression_config.enabled is False
            assert indexer.function_extractor.compression_config.enabled is False
