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

"""Data preparation for masked language modeling (MLM) fine-tuning.

This module handles extracting code from git commits and preparing it for
GraphCodeBERT fine-tuning using masked language modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import git
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from ..analyzers.domain_classifier import DomainClassifier

logger = logging.getLogger(__name__)


@dataclass
class CodeSample:
    """Represents a code sample for training."""

    file_path: str
    content: str
    domain: str
    tokens: int


class CodeDatasetPreparator:
    """Prepare code datasets from git commits for fine-tuning.

    This class extracts code from git commits, classifies by domain,
    and prepares training samples for masked language modeling.
    """

    def __init__(
        self,
        repo_path: str,
        tokenizer: RobertaTokenizer,
        max_files: int = 1000,
        include_test_files: bool = False,
        include_generated_files: bool = False,
    ):
        """Initialize the dataset preparator.

        Args:
            repo_path: Path to git repository
            tokenizer: RobertaTokenizer for tokenization
            max_files: Maximum number of files to extract
            include_test_files: Whether to include test files
            include_generated_files: Whether to include generated files
        """
        self.repo_path = Path(repo_path)
        self.tokenizer = tokenizer
        self.max_files = max_files
        self.include_test_files = include_test_files
        self.include_generated_files = include_generated_files

        # Initialize git repo
        self.repo = git.Repo(repo_path)

        # Initialize domain classifier for domain-aware sampling
        self.domain_classifier = DomainClassifier()

        # Exclude patterns
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "dist",
            "build",
            ".next",
            "coverage",
            ".pytest_cache",
        ]

    def extract_code_from_commit(self, commit_hash: str) -> list[CodeSample]:
        """Extract all code files from a specific commit.

        Args:
            commit_hash: Git commit hash to extract from

        Returns:
            List of CodeSample objects
        """
        logger.info(f"Extracting code from commit {commit_hash}")

        try:
            commit = self.repo.commit(commit_hash)
        except Exception as e:
            raise ValueError(f"Invalid commit hash {commit_hash}: {e}") from e

        code_samples = []

        # Traverse the commit tree
        for item in commit.tree.traverse():
            if (
                hasattr(item, "type")
                and hasattr(item, "path")
                and hasattr(item, "data_stream")
                and item.type == "blob"
            ):
                file_path = str(item.path)

                # Apply filters
                if self._should_exclude_file(file_path):
                    continue

                try:
                    # Read file content
                    file_content = item.data_stream.read().decode("utf-8")

                    # Skip empty files
                    if not file_content.strip():
                        continue

                    # Classify domain
                    classification = self.domain_classifier.classify_domain(
                        file_path, file_content
                    )

                    # Count tokens
                    tokens = len(
                        self.tokenizer.encode(
                            file_content, truncation=True, max_length=512
                        )
                    )

                    code_sample = CodeSample(
                        file_path=file_path,
                        content=file_content,
                        domain=classification.domain.value,
                        tokens=tokens,
                    )
                    code_samples.append(code_sample)

                    # Limit total files
                    if len(code_samples) >= self.max_files:
                        logger.info(f"Reached max_files limit of {self.max_files}")
                        break

                except (UnicodeDecodeError, Exception) as e:
                    logger.debug(f"Skipping file {file_path}: {e}")
                    continue

        logger.info(
            f"Extracted {len(code_samples)} code samples from commit {commit_hash}"
        )

        # Log domain distribution
        domain_counts: dict[str, int] = {}
        for sample in code_samples:
            domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1

        logger.info(f"Domain distribution: {domain_counts}")

        return code_samples

    def _should_exclude_file(self, file_path: str) -> bool:
        """Check if a file should be excluded from training.

        Args:
            file_path: Path to file

        Returns:
            True if file should be excluded
        """
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in file_path:
                return True

        # Exclude test files if configured
        if not self.include_test_files:
            test_patterns = ["test_", "_test.", ".test.", "spec_", "_spec.", ".spec."]
            if any(pattern in file_path.lower() for pattern in test_patterns):
                return True

        # Exclude generated files if configured
        if not self.include_generated_files:
            generated_patterns = [".generated.", "_generated.", ".d.ts"]
            if any(pattern in file_path for pattern in generated_patterns):
                return True

        # Only include code files (basic extension filter)
        code_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".py",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".cs",
            ".sql",
        }
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in code_extensions:
            return True

        return False

    def create_mlm_dataset(
        self,
        code_samples: list[CodeSample],
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ) -> MLMDataset:
        """Create a masked language modeling dataset from code samples.

        Args:
            code_samples: List of code samples to use
            max_length: Maximum sequence length
            mlm_probability: Probability of masking each token

        Returns:
            MLMDataset ready for training
        """
        logger.info(
            f"Creating MLM dataset from {len(code_samples)} samples "
            f"(max_length={max_length}, mlm_probability={mlm_probability})"
        )

        dataset = MLMDataset(
            code_samples=code_samples,
            tokenizer=self.tokenizer,
            max_length=max_length,
            mlm_probability=mlm_probability,
        )

        logger.info(f"Created MLM dataset with {len(dataset)} samples")
        return dataset


class MLMDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for masked language modeling on code.

    This dataset handles tokenization and masking of code samples for
    training GraphCodeBERT using MLM objective.
    """

    def __init__(
        self,
        code_samples: list[CodeSample],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        """Initialize MLM dataset.

        Args:
            code_samples: List of code samples
            tokenizer: RobertaTokenizer for tokenization
            max_length: Maximum sequence length
            mlm_probability: Probability of masking tokens
        """
        self.code_samples = code_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        # Cache tokenized samples for faster training
        self.tokenized_samples: list[dict[str, Any]] = []
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """Pre-tokenize all samples for faster training."""
        logger.info("Pre-tokenizing code samples...")

        for sample in self.code_samples:
            # Tokenize - keep as lists, don't create tensors yet
            # Creating tensors here can conflict with MPS device initialization
            encoding = self.tokenizer(
                sample.content,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                # Don't use return_tensors - create tensors later in __getitem__
            )

            # Store as dict with raw token IDs (lists, not tensors)
            tokenized = {
                "input_ids": encoding["input_ids"],  # List of ints
                "attention_mask": encoding["attention_mask"],  # List of ints
                "file_path": sample.file_path,
                "domain": sample.domain,
            }

            self.tokenized_samples.append(tokenized)

        logger.info(f"Pre-tokenized {len(self.tokenized_samples)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.tokenized_samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample with masked language modeling applied.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, and labels for MLM
        """
        sample = self.tokenized_samples[idx]

        # Convert lists to tensors (delayed from _prepare_samples)
        # This ensures tensors are created in the correct device context
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)

        # Create labels (same as input_ids initially)
        labels = input_ids.clone()

        # Apply masking
        input_ids, labels = self._mask_tokens(input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "file_path": sample["file_path"],
            "domain": sample["domain"],
        }

    def _mask_tokens(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking to input tokens.

        Args:
            input_ids: Input token IDs
            labels: Label token IDs (same as input initially)

        Returns:
            Tuple of (masked input_ids, labels)
        """
        # Special tokens that should never be masked
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        # Create probability matrix
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Don't mask special tokens
        for special_token_id in special_tokens:
            if special_token_id is not None:
                probability_matrix[input_ids == special_token_id] = 0.0

        # Get mask indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK] token
        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace with random token
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_tokens = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_tokens[indices_random]

        # 10% of the time, keep original token (do nothing)

        return input_ids, labels
