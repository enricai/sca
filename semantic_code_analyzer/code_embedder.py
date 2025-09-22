"""
Semantic code embedding module optimized for Apple M3 hardware acceleration.

This module provides functionality to generate semantic embeddings for code
using state-of-the-art models with MPS (Metal Performance Shaders) acceleration.
"""

import torch
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import ast
import re
import logging
from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
import time

# Suppress transformers warnings for cleaner output
transformers_logging.set_verbosity_warning()

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for code embedding generation."""
    model_name: str = "microsoft/graphcodebert-base"
    max_length: int = 512
    batch_size: int = 8
    use_mps: bool = True
    cache_embeddings: bool = True
    normalize_embeddings: bool = True


@dataclass
class FunctionInfo:
    """Information about a function extracted from code."""
    name: str
    code: str
    line_start: int
    line_end: int
    embedding: Optional[np.ndarray] = None


class CodeEmbedder:
    """
    Generates semantic embeddings for code using GraphCodeBERT with Apple M3 optimization.

    Features:
    - MPS (Metal Performance Shaders) acceleration for Apple Silicon
    - Mixed precision inference for faster processing
    - Intelligent caching system
    - Function-level and file-level embedding extraction
    - Multi-language support
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the CodeEmbedder.

        Args:
            config: Configuration object for embedding generation
        """
        self.config = config or EmbeddingConfig()

        # Setup device with Apple M3 optimization
        self.device = self._setup_device()

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()

        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_file = Path(".embedding_cache.pkl")

        if self.config.cache_embeddings:
            self._load_cache()

        logger.info(f"CodeEmbedder initialized with device: {self.device}")

    def _setup_device(self) -> torch.device:
        """
        Setup the optimal device for Apple M3 hardware.

        Returns:
            torch.device: The device to use for computation
        """
        if self.config.use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple M3 MPS acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA acceleration")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU computation")

        return device

    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            # Prevent tokenizer parallelism warning
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Suppress model loading warnings
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
                warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )

                # Load model with optimizations
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    dtype=torch.float16 if self.device.type == "mps" else torch.float32
                )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Enable mixed precision for MPS
            if self.device.type == "mps":
                self.model = torch.compile(self.model)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_code(self, code: str, language: str = "python") -> str:
        """
        Clean and normalize code for better embedding generation.

        Args:
            code: Raw code string
            language: Programming language of the code

        Returns:
            Preprocessed code string
        """
        if not code.strip():
            return ""

        # Remove excessive whitespace
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)

        # Language-specific preprocessing
        if language.lower() == "python":
            try:
                # Parse and unparse to normalize Python code
                tree = ast.parse(code)
                code = ast.unparse(tree)
            except SyntaxError:
                # If parsing fails, use original code with basic cleanup
                code = code.strip()
        else:
            # Basic cleanup for other languages
            code = code.strip()

        return code

    def get_code_embedding(self,
                          code: str,
                          language: str = "python",
                          use_cache: bool = True) -> np.ndarray:
        """
        Generate semantic embedding for a code snippet.

        Args:
            code: Code string to embed
            language: Programming language of the code
            use_cache: Whether to use cached embeddings

        Returns:
            Numpy array representing the code embedding
        """
        if not code.strip():
            return np.zeros(768)  # GraphCodeBERT embedding size

        # Create cache key
        cache_key = self._get_cache_key(code, language)

        # Check cache first
        if use_cache and self.config.cache_embeddings and cache_key in self.embedding_cache:
            logger.debug("Using cached embedding")
            return self.embedding_cache[cache_key]

        # Preprocess code
        processed_code = self.preprocess_code(code, language)

        # Tokenize with language-specific formatting
        formatted_code = self._format_code_for_model(processed_code, language)

        try:
            # Tokenize
            inputs = self.tokenizer(
                formatted_code,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding with mixed precision
            with torch.no_grad():
                if self.device.type == "mps":
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)

                # Convert to numpy
                embedding = embeddings.cpu().numpy().flatten().astype(np.float32)

                # Normalize if requested
                if self.config.normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(768)

        # Cache the result
        if self.config.cache_embeddings:
            self.embedding_cache[cache_key] = embedding

        logger.debug(f"Generated embedding of shape: {embedding.shape}")
        return embedding

    def get_function_embeddings(self,
                               code: str,
                               language: str = "python") -> List[FunctionInfo]:
        """
        Extract individual function embeddings from code.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            List of FunctionInfo objects with embeddings
        """
        functions = []

        if language.lower() == "python":
            functions = self._extract_python_functions(code)
        else:
            # Fallback to whole file for non-Python languages
            embedding = self.get_code_embedding(code, language)
            functions = [FunctionInfo(
                name="whole_file",
                code=code,
                line_start=1,
                line_end=len(code.split('\n')),
                embedding=embedding
            )]

        # Generate embeddings for extracted functions
        for func_info in functions:
            if func_info.embedding is None:
                func_info.embedding = self.get_code_embedding(func_info.code, language)

        logger.debug(f"Extracted {len(functions)} functions with embeddings")
        return functions

    def get_batch_embeddings(self,
                            code_snippets: List[str],
                            language: str = "python") -> List[np.ndarray]:
        """
        Generate embeddings for multiple code snippets efficiently.

        Args:
            code_snippets: List of code strings
            language: Programming language

        Returns:
            List of embedding arrays
        """
        if not code_snippets:
            return []

        embeddings = []

        # Process in batches for efficiency
        for i in range(0, len(code_snippets), self.config.batch_size):
            batch = code_snippets[i:i + self.config.batch_size]
            batch_embeddings = self._process_batch(batch, language)
            embeddings.extend(batch_embeddings)

        logger.info(f"Generated {len(embeddings)} embeddings in batch")
        return embeddings

    def _process_batch(self,
                      code_batch: List[str],
                      language: str) -> List[np.ndarray]:
        """
        Process a batch of code snippets.

        Args:
            code_batch: Batch of code strings
            language: Programming language

        Returns:
            List of embedding arrays
        """
        # Preprocess all codes in batch
        processed_codes = [
            self._format_code_for_model(self.preprocess_code(code, language), language)
            for code in code_batch
        ]

        try:
            # Tokenize batch
            inputs = self.tokenizer(
                processed_codes,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                if self.device.type == "mps":
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy().astype(np.float32)

            # Normalize if requested
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-8)

            return [emb for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.get_code_embedding(code, language, use_cache=False)
                   for code in code_batch]

    def _extract_python_functions(self, code: str) -> List[FunctionInfo]:
        """
        Extract Python functions from source code.

        Args:
            code: Python source code

        Returns:
            List of FunctionInfo objects
        """
        functions = []

        try:
            tree = ast.parse(code)
            lines = code.split('\n')

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function code
                    start_line = node.lineno - 1  # Convert to 0-based
                    end_line = node.end_lineno or start_line + 1

                    func_lines = lines[start_line:end_line]
                    func_code = '\n'.join(func_lines)

                    functions.append(FunctionInfo(
                        name=node.name,
                        code=func_code,
                        line_start=start_line + 1,  # Convert back to 1-based
                        line_end=end_line,
                        embedding=None
                    ))

        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")
            # Return whole file as single function
            functions = [FunctionInfo(
                name="whole_file",
                code=code,
                line_start=1,
                line_end=len(code.split('\n')),
                embedding=None
            )]

        return functions

    def _format_code_for_model(self, code: str, language: str) -> str:
        """
        Format code for the model with language-specific prefixes.

        Args:
            code: Preprocessed code
            language: Programming language

        Returns:
            Formatted code string
        """
        # Add language hint for better understanding
        if language.lower() in ["python", "py"]:
            return f"# Python code:\n{code}"
        elif language.lower() in ["javascript", "js"]:
            return f"// JavaScript code:\n{code}"
        elif language.lower() in ["typescript", "ts"]:
            return f"// TypeScript code:\n{code}"
        elif language.lower() in ["java"]:
            return f"// Java code:\n{code}"
        elif language.lower() in ["cpp", "c++"]:
            return f"// C++ code:\n{code}"
        else:
            return code

    def _get_cache_key(self, code: str, language: str) -> str:
        """
        Generate a cache key for the code.

        Args:
            code: Code string
            language: Programming language

        Returns:
            Cache key string
        """
        content = f"{language}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self):
        """Load embeddings from cache file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.embedding_cache = {}

    def save_cache(self):
        """Save embeddings to cache file."""
        if self.config.cache_embeddings and self.embedding_cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Embedding cache cleared")

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "max_length": self.config.max_length,
            "embedding_dim": 768,  # GraphCodeBERT embedding dimension
            "cache_size": len(self.embedding_cache),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available()
        }

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'config') and self.config.cache_embeddings:
            self.save_cache()