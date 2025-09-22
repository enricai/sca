"""
Tests for the CodeEmbedder module.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from semantic_code_analyzer.code_embedder import (
    CodeEmbedder,
    EmbeddingConfig,
    FunctionInfo
)


class TestEmbeddingConfig:
    """Test cases for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == "microsoft/graphcodebert-base"
        assert config.max_length == 512
        assert config.batch_size == 8
        assert config.use_mps is True
        assert config.cache_embeddings is True
        assert config.normalize_embeddings is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_name="custom/model",
            max_length=256,
            batch_size=4,
            use_mps=False,
            cache_embeddings=False,
            normalize_embeddings=False
        )

        assert config.model_name == "custom/model"
        assert config.max_length == 256
        assert config.batch_size == 4
        assert config.use_mps is False
        assert config.cache_embeddings is False
        assert config.normalize_embeddings is False


class TestFunctionInfo:
    """Test cases for FunctionInfo dataclass."""

    def test_function_info_creation(self):
        """Test creating FunctionInfo instance."""
        embedding = np.random.rand(768).astype(np.float32)

        func_info = FunctionInfo(
            name="test_function",
            code="def test_function():\n    pass",
            line_start=1,
            line_end=2,
            embedding=embedding
        )

        assert func_info.name == "test_function"
        assert func_info.code == "def test_function():\n    pass"
        assert func_info.line_start == 1
        assert func_info.line_end == 2
        assert np.array_equal(func_info.embedding, embedding)

    def test_function_info_without_embedding(self):
        """Test FunctionInfo without embedding."""
        func_info = FunctionInfo(
            name="test_function",
            code="def test_function():\n    pass",
            line_start=1,
            line_end=2
        )

        assert func_info.embedding is None


class TestCodeEmbedder:
    """Test cases for CodeEmbedder class."""

    @pytest.fixture
    def mock_model_components(self):
        """Create mock model components."""
        mock_tokenizer = Mock()
        mock_model = Mock()

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }

        # Mock model output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 5, 768)
        mock_model.return_value = mock_output
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model

        return mock_tokenizer, mock_model

    @pytest.fixture
    def embedder_with_mocks(self, mock_model_components, disable_mps):
        """Create CodeEmbedder with mocked components."""
        mock_tokenizer, mock_model = mock_model_components

        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            config = EmbeddingConfig(use_mps=False, cache_embeddings=False)
            embedder = CodeEmbedder(config)

            return embedder

    def test_init_default_config(self, disable_mps):
        """Test initialization with default configuration."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model_class.from_pretrained.return_value = mock_model

            embedder = CodeEmbedder()

            assert embedder.config.model_name == "microsoft/graphcodebert-base"
            assert embedder.device.type == "cpu"  # MPS disabled in test

    def test_init_custom_config(self, disable_mps):
        """Test initialization with custom configuration."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = Mock()
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model_class.from_pretrained.return_value = mock_model

            config = EmbeddingConfig(
                model_name="custom/model",
                use_mps=False,
                cache_embeddings=False
            )
            embedder = CodeEmbedder(config)

            assert embedder.config.model_name == "custom/model"
            assert embedder.config.cache_embeddings is False

    def test_setup_device_cpu(self, disable_mps):
        """Test device setup with CPU fallback."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer'), \
             patch('semantic_code_analyzer.code_embedder.AutoModel'):

            config = EmbeddingConfig(use_mps=False)
            embedder = CodeEmbedder(config)

            assert embedder.device.type == "cpu"

    @patch('torch.backends.mps.is_available', return_value=True)
    def test_setup_device_mps(self, mock_mps_available):
        """Test device setup with MPS."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer'), \
             patch('semantic_code_analyzer.code_embedder.AutoModel'):

            config = EmbeddingConfig(use_mps=True)
            embedder = CodeEmbedder(config)

            assert embedder.device.type == "mps"

    @patch('torch.cuda.is_available', return_value=True)
    def test_setup_device_cuda(self, mock_cuda_available, disable_mps):
        """Test device setup with CUDA."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer'), \
             patch('semantic_code_analyzer.code_embedder.AutoModel'):

            config = EmbeddingConfig(use_mps=False)
            embedder = CodeEmbedder(config)

            assert embedder.device.type == "cuda"

    def test_preprocess_code_python(self, embedder_with_mocks):
        """Test Python code preprocessing."""
        code = """
def hello():
    print("Hello")


def world():
    print("World")
"""

        processed = embedder_with_mocks.preprocess_code(code, "python")

        # Should be normalized and cleaned
        assert processed.strip()
        assert "def hello():" in processed
        assert "def world():" in processed

    def test_preprocess_code_invalid_python(self, embedder_with_mocks):
        """Test preprocessing invalid Python code."""
        code = "def invalid syntax here:"

        processed = embedder_with_mocks.preprocess_code(code, "python")

        # Should fallback to original code
        assert processed.strip() == code.strip()

    def test_preprocess_code_other_language(self, embedder_with_mocks):
        """Test preprocessing non-Python code."""
        code = """
function hello() {
    console.log("Hello");
}
"""

        processed = embedder_with_mocks.preprocess_code(code, "javascript")

        assert processed.strip() == code.strip()

    def test_preprocess_code_empty(self, embedder_with_mocks):
        """Test preprocessing empty code."""
        processed = embedder_with_mocks.preprocess_code("", "python")
        assert processed == ""

        processed = embedder_with_mocks.preprocess_code("   \n  \t  ", "python")
        assert processed == ""

    def test_get_code_embedding(self, embedder_with_mocks):
        """Test generating code embedding."""
        code = "def hello():\n    print('Hello, World!')"

        embedding = embedder_with_mocks.get_code_embedding(code, "python")

        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (768,)  # GraphCodeBERT embedding size

    def test_get_code_embedding_empty(self, embedder_with_mocks):
        """Test embedding generation for empty code."""
        embedding = embedder_with_mocks.get_code_embedding("", "python")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert np.allclose(embedding, 0.0)  # Should be zeros

    def test_get_code_embedding_with_cache(self, mock_model_components, disable_mps):
        """Test embedding generation with caching."""
        mock_tokenizer, mock_model = mock_model_components

        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            config = EmbeddingConfig(cache_embeddings=True)
            embedder = CodeEmbedder(config)

            code = "def test(): pass"

            # First call should generate embedding
            embedding1 = embedder.get_code_embedding(code, "python")

            # Second call should use cache
            embedding2 = embedder.get_code_embedding(code, "python")

            assert np.array_equal(embedding1, embedding2)

    def test_get_code_embedding_no_cache(self, embedder_with_mocks):
        """Test embedding generation without caching."""
        code = "def test(): pass"

        embedding = embedder_with_mocks.get_code_embedding(code, "python", use_cache=False)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_format_code_for_model(self, embedder_with_mocks):
        """Test code formatting for different languages."""
        code = "def hello(): pass"

        # Python
        formatted = embedder_with_mocks._format_code_for_model(code, "python")
        assert formatted.startswith("# Python code:")

        # JavaScript
        formatted = embedder_with_mocks._format_code_for_model(code, "javascript")
        assert formatted.startswith("// JavaScript code:")

        # Java
        formatted = embedder_with_mocks._format_code_for_model(code, "java")
        assert formatted.startswith("// Java code:")

        # Unknown language
        formatted = embedder_with_mocks._format_code_for_model(code, "unknown")
        assert formatted == code

    def test_extract_python_functions(self, embedder_with_mocks):
        """Test extracting Python functions from code."""
        code = '''
def function1():
    """First function."""
    return 1

def function2(x, y):
    """Second function."""
    return x + y

class TestClass:
    def method1(self):
        pass
'''

        functions = embedder_with_mocks._extract_python_functions(code)

        assert len(functions) >= 2  # Should find at least function1 and function2
        function_names = [f.name for f in functions]
        assert "function1" in function_names
        assert "function2" in function_names

        # Check function details
        func1 = next(f for f in functions if f.name == "function1")
        assert "def function1():" in func1.code
        assert func1.line_start > 0
        assert func1.line_end > func1.line_start

    def test_extract_python_functions_invalid_syntax(self, embedder_with_mocks):
        """Test function extraction with invalid Python syntax."""
        code = "def invalid syntax here:"

        functions = embedder_with_mocks._extract_python_functions(code)

        # Should fallback to whole file
        assert len(functions) == 1
        assert functions[0].name == "whole_file"
        assert functions[0].code == code

    def test_get_function_embeddings(self, embedder_with_mocks):
        """Test getting embeddings for individual functions."""
        code = '''
def function1():
    return 1

def function2():
    return 2
'''

        functions = embedder_with_mocks.get_function_embeddings(code, "python")

        assert len(functions) >= 2
        assert all(isinstance(f.embedding, np.ndarray) for f in functions)
        assert all(f.embedding.shape == (768,) for f in functions)

    def test_get_function_embeddings_non_python(self, embedder_with_mocks):
        """Test function embeddings for non-Python languages."""
        code = '''
function test() {
    return 1;
}
'''

        functions = embedder_with_mocks.get_function_embeddings(code, "javascript")

        # Should fallback to whole file
        assert len(functions) == 1
        assert functions[0].name == "whole_file"
        assert isinstance(functions[0].embedding, np.ndarray)

    def test_get_batch_embeddings(self, embedder_with_mocks):
        """Test batch embedding generation."""
        code_snippets = [
            "def func1(): pass",
            "def func2(): pass",
            "def func3(): pass"
        ]

        embeddings = embedder_with_mocks.get_batch_embeddings(code_snippets, "python")

        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (768,) for emb in embeddings)

    def test_get_batch_embeddings_empty(self, embedder_with_mocks):
        """Test batch embedding generation with empty list."""
        embeddings = embedder_with_mocks.get_batch_embeddings([], "python")
        assert embeddings == []

    def test_get_cache_key(self, embedder_with_mocks):
        """Test cache key generation."""
        code = "def test(): pass"
        language = "python"

        key1 = embedder_with_mocks._get_cache_key(code, language)
        key2 = embedder_with_mocks._get_cache_key(code, language)
        key3 = embedder_with_mocks._get_cache_key("different code", language)

        assert key1 == key2  # Same input should produce same key
        assert key1 != key3  # Different input should produce different key
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_clear_cache(self, mock_model_components, disable_mps):
        """Test cache clearing."""
        mock_tokenizer, mock_model = mock_model_components

        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            config = EmbeddingConfig(cache_embeddings=True)
            embedder = CodeEmbedder(config)

            # Add something to cache
            embedder.embedding_cache["test"] = np.array([1, 2, 3])

            embedder.clear_cache()

            assert len(embedder.embedding_cache) == 0

    def test_get_model_info(self, embedder_with_mocks):
        """Test retrieving model information."""
        info = embedder_with_mocks.get_model_info()

        assert "model_name" in info
        assert "device" in info
        assert "max_length" in info
        assert "embedding_dim" in info
        assert "cache_size" in info
        assert "mps_available" in info
        assert "cuda_available" in info

        assert info["embedding_dim"] == 768
        assert isinstance(info["cache_size"], int)

    def test_save_and_load_cache(self, mock_model_components, disable_mps):
        """Test cache persistence."""
        mock_tokenizer, mock_model = mock_model_components

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.pkl"

            with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
                 patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                mock_model_class.from_pretrained.return_value = mock_model

                config = EmbeddingConfig(cache_embeddings=True)

                # Create embedder and add to cache
                with patch.object(CodeEmbedder, 'cache_file', cache_file):
                    embedder1 = CodeEmbedder(config)
                    embedder1.embedding_cache["test"] = np.array([1, 2, 3])
                    embedder1.save_cache()

                # Create new embedder and load cache
                with patch.object(CodeEmbedder, 'cache_file', cache_file):
                    embedder2 = CodeEmbedder(config)

                assert "test" in embedder2.embedding_cache
                assert np.array_equal(embedder2.embedding_cache["test"], np.array([1, 2, 3]))

    def test_normalization(self, embedder_with_mocks):
        """Test embedding normalization."""
        # Test with normalization enabled
        embedder_with_mocks.config.normalize_embeddings = True

        code = "def test(): pass"
        embedding = embedder_with_mocks.get_code_embedding(code, "python")

        # Check if normalized (L2 norm should be close to 1)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.1  # Allow some tolerance

    def test_error_handling_model_failure(self, disable_mps):
        """Test error handling when model fails to load."""
        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.side_effect = Exception("Model load failed")

            config = EmbeddingConfig()

            with pytest.raises(Exception):
                CodeEmbedder(config)

    def test_error_handling_embedding_generation(self, embedder_with_mocks):
        """Test error handling during embedding generation."""
        # Mock tokenizer to raise exception
        embedder_with_mocks.tokenizer.side_effect = Exception("Tokenization failed")

        code = "def test(): pass"
        embedding = embedder_with_mocks.get_code_embedding(code, "python")

        # Should return zero embedding on error
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert np.allclose(embedding, 0.0)


class TestCodeEmbedderEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def embedder_with_mocks(self, mock_model_components, disable_mps):
        """Create CodeEmbedder with mocked components."""
        mock_tokenizer, mock_model = mock_model_components

        with patch('semantic_code_analyzer.code_embedder.AutoTokenizer') as mock_tokenizer_class, \
             patch('semantic_code_analyzer.code_embedder.AutoModel') as mock_model_class:

            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            config = EmbeddingConfig(use_mps=False, cache_embeddings=False)
            embedder = CodeEmbedder(config)

            return embedder

    def test_very_long_code(self, embedder_with_mocks):
        """Test handling of very long code snippets."""
        # Create code longer than max_length
        long_code = "def func():\n" + "    x = 1\n" * 1000

        embedding = embedder_with_mocks.get_code_embedding(long_code, "python")

        # Should handle gracefully with truncation
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_unicode_code(self, embedder_with_mocks):
        """Test handling of code with Unicode characters."""
        unicode_code = '''
def función():
    """Función con caracteres especiales: áéíóú"""
    return "¡Hola mundo! 🌍"
'''

        embedding = embedder_with_mocks.get_code_embedding(unicode_code, "python")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_mixed_content_code(self, embedder_with_mocks):
        """Test code with mixed content (code + comments + strings)."""
        mixed_code = '''
# This is a comment with special chars: @#$%^&*()
def process_data():
    """
    Multi-line docstring
    with various content
    """
    data = "String with \n newlines and \t tabs"
    regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return data
'''

        embedding = embedder_with_mocks.get_code_embedding(mixed_code, "python")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_batch_processing_fallback(self, embedder_with_mocks):
        """Test batch processing fallback to individual processing."""
        code_snippets = ["def func1(): pass", "def func2(): pass"]

        # Mock batch processing to fail
        with patch.object(embedder_with_mocks, '_process_batch', side_effect=Exception("Batch failed")):
            embeddings = embedder_with_mocks.get_batch_embeddings(code_snippets, "python")

        # Should fallback to individual processing
        assert len(embeddings) == 2
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)

    def test_memory_efficiency(self, embedder_with_mocks):
        """Test memory efficiency with large batches."""
        # Create many small code snippets
        code_snippets = [f"def func{i}(): return {i}" for i in range(100)]

        embeddings = embedder_with_mocks.get_batch_embeddings(code_snippets, "python")

        assert len(embeddings) == 100
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)

    @patch('semantic_code_analyzer.code_embedder.logger')
    def test_logging_calls(self, mock_logger, embedder_with_mocks):
        """Test that appropriate logging calls are made."""
        # Test various operations that should log
        embedder_with_mocks.get_code_embedding("def test(): pass", "python")
        embedder_with_mocks.get_function_embeddings("def test(): pass", "python")

        # Should have made some debug/info log calls
        assert mock_logger.debug.called or mock_logger.info.called