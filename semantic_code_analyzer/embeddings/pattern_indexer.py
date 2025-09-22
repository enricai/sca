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

"""Pattern indexing system using GraphCodeBERT embeddings and FAISS similarity search.

This module provides functionality to build domain-specific code pattern indices
using state-of-the-art code embeddings for fast similarity search and pattern
matching in domain-aware adherence analysis.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)

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


class PatternIndexer:
    """Build and manage domain-specific FAISS similarity indices using GraphCodeBERT embeddings.

    This class provides functionality to:
    - Extract code embeddings using GraphCodeBERT
    - Build FAISS indices for fast similarity search
    - Search for similar patterns within domain contexts
    - Cache embeddings for performance optimization
    """

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        cache_dir: str | None = None,
    ):
        """Initialize the PatternIndexer with GraphCodeBERT model.

        Args:
            model_name: Name of the GraphCodeBERT model to use
            cache_dir: Directory for caching models and indices
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".sca_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize model and tokenizer
        logger.info(f"Loading GraphCodeBERT model: {model_name}")
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                model_name, cache_dir=str(self.cache_dir)
            )
            self.model = RobertaModel.from_pretrained(
                model_name, cache_dir=str(self.cache_dir)
            )
            self.model.eval()

            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load GraphCodeBERT model: {e}")
            raise ValueError(f"Could not initialize GraphCodeBERT model: {e}") from e

        # Storage for domain indices
        self.domain_indices: dict[str, PatternIndex] = {}
        self.embedding_cache: dict[
            str, np.ndarray[Any, np.dtype[np.floating[Any]]]
        ] = {}

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
        chunk_size: int = 512,
    ) -> None:
        """Build a FAISS similarity index for a specific domain.

        Args:
            domain: Domain name (e.g., 'frontend', 'backend')
            codebase_files: Dictionary mapping file paths to their content
            max_files: Maximum number of files to process (for performance)
            chunk_size: Maximum token chunk size for processing
        """
        logger.info(f"Building pattern index for domain: {domain}")

        if not codebase_files:
            logger.warning(f"No files provided for domain {domain}")
            return

        # Limit files if specified
        files_to_process = (
            dict(list(codebase_files.items())[:max_files])
            if max_files
            else codebase_files
        )

        # Extract embeddings for all files
        embeddings_list = []
        file_paths = []
        code_snippets = []

        for file_path, content in files_to_process.items():
            try:
                # Process content in chunks if too large
                content_chunks = self._chunk_content(content, chunk_size)

                for i, chunk in enumerate(content_chunks):
                    cache_key = f"{domain}:{file_path}:{i}:{hash(chunk)}"

                    if cache_key in self.embedding_cache:
                        embedding = self.embedding_cache[cache_key]
                    else:
                        embedding = self._extract_code_embeddings(chunk)
                        self.embedding_cache[cache_key] = embedding

                    embeddings_list.append(embedding)
                    file_paths.append(file_path)
                    code_snippets.append(chunk)

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

        # Store the pattern index
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
            },
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
            # Extract query embedding
            query_embedding = self._extract_code_embeddings(query_code)
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))

            # Search in FAISS index
            scores, indices = pattern_index.index.search(query_embedding, top_k)

            # Create similarity matches
            similarity_matches = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx == -1 or score < min_similarity:
                    continue

                similarity_match = SimilarityMatch(
                    file_path=pattern_index.file_paths[idx],
                    similarity_score=float(score),
                    code_snippet=pattern_index.code_snippets[idx],
                    domain=domain,
                    context={
                        "index": int(idx),
                        "embedding_dimension": query_embedding.shape[1],
                        "search_parameters": {
                            "top_k": top_k,
                            "min_similarity": min_similarity,
                        },
                    },
                )
                similarity_matches.append(similarity_match)

            logger.debug(
                f"Found {len(similarity_matches)} similar patterns for domain {domain}"
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

    def _extract_code_embeddings(
        self, code_content: str
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Extract GraphCodeBERT embeddings for code content.

        Args:
            code_content: Source code content

        Returns:
            Normalized embedding vector
        """
        try:
            # Tokenize the code
            tokens = self.tokenizer(
                code_content,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            # Move tokens to device
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Use the [CLS] token embedding as the code representation
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embeddings.squeeze()

        except Exception as e:
            logger.warning(f"Failed to extract embeddings for code snippet: {e}")
            # Return zero vector as fallback
            return np.zeros(768)  # GraphCodeBERT has 768-dim embeddings

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

    def _chunk_content(self, content: str, max_tokens: int = 512) -> list[str]:
        """Split content into chunks that fit within token limits.

        Args:
            content: Source code content
            max_tokens: Maximum tokens per chunk

        Returns:
            List of content chunks
        """
        # Simple line-based chunking for now
        lines = content.split("\n")
        chunks = []
        current_chunk: list[str] = []
        current_length = 0

        for line in lines:
            # Rough token estimation (1 token ≈ 4 characters)
            line_tokens = len(line) // 4 + 1

            if current_length + line_tokens > max_tokens and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = line_tokens
            else:
                current_chunk.append(line)
                current_length += line_tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [content[:2000]]  # Fallback for very long lines

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
                "format_version": "json_v1",  # For future compatibility
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

            pattern_index = PatternIndex(
                domain=index_data["domain"],
                index=index,
                file_paths=index_data["file_paths"],
                code_snippets=index_data["code_snippets"],
                embeddings=embeddings,
                metadata=index_data["metadata"],
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
