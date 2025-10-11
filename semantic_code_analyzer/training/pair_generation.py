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

"""Automatic pair generation for contrastive learning.

This module generates positive and negative code pairs for training
code embedding models with contrastive learning objectives.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

from .data_preparation import CodeSample

logger = logging.getLogger(__name__)


@dataclass
class CodePair:
    """Represents a pair of code samples for contrastive learning."""

    anchor: CodeSample
    comparison: CodeSample
    is_positive: bool  # True if similar, False if dissimilar
    pair_type: str  # Description of why they're paired


class ContrastivePairGenerator:
    """Generate positive and negative pairs for contrastive learning.

    This class automatically creates training pairs from a codebase without
    requiring manual labeling.
    """

    def __init__(self, min_content_length: int = 50):
        """Initialize the pair generator.

        Args:
            min_content_length: Minimum code length to consider for pairing
        """
        self.min_content_length = min_content_length

    def generate_pairs(
        self, code_samples: list[CodeSample], pairs_per_sample: int = 3
    ) -> list[CodePair]:
        """Generate both positive and negative pairs from code samples.

        Args:
            code_samples: List of code samples from codebase
            pairs_per_sample: Number of pairs to generate per sample

        Returns:
            List of CodePair objects
        """
        logger.info(
            f"Generating contrastive pairs from {len(code_samples)} samples "
            f"({pairs_per_sample} pairs per sample)"
        )

        # Group by domain for efficient pairing
        domain_groups = self._group_by_domain(code_samples)

        all_pairs = []

        # Generate positive pairs
        positive_pairs = self._generate_positive_pairs(
            code_samples, domain_groups, pairs_per_sample
        )
        all_pairs.extend(positive_pairs)

        # Generate negative pairs (same number as positives)
        negative_pairs = self._generate_negative_pairs(
            code_samples, domain_groups, len(positive_pairs)
        )
        all_pairs.extend(negative_pairs)

        logger.info(
            f"Generated {len(all_pairs)} total pairs "
            f"({len(positive_pairs)} positive, {len(negative_pairs)} negative)"
        )

        return all_pairs

    def _group_by_domain(
        self, code_samples: list[CodeSample]
    ) -> dict[str, list[CodeSample]]:
        """Group code samples by domain.

        Args:
            code_samples: List of code samples

        Returns:
            Dictionary mapping domain to list of samples
        """
        groups: dict[str, list[CodeSample]] = {}

        for sample in code_samples:
            if len(sample.content) < self.min_content_length:
                continue

            if sample.domain not in groups:
                groups[sample.domain] = []

            groups[sample.domain].append(sample)

        return groups

    def _generate_positive_pairs(
        self,
        code_samples: list[CodeSample],
        domain_groups: dict[str, list[CodeSample]],
        pairs_per_sample: int,
    ) -> list[CodePair]:
        """Generate positive pairs (similar code).

        Args:
            code_samples: All code samples
            domain_groups: Samples grouped by domain
            pairs_per_sample: Pairs to generate per sample

        Returns:
            List of positive CodePair objects
        """
        positive_pairs = []

        for domain, samples in domain_groups.items():
            if len(samples) < 2:
                continue  # Need at least 2 samples to pair

            # Create pairs within same domain
            for i, anchor in enumerate(samples):
                # Find similar samples from same domain
                candidates = [s for j, s in enumerate(samples) if j != i]

                if not candidates:
                    continue

                # Sample random positives from same domain
                num_pairs = min(pairs_per_sample, len(candidates))
                positives = random.sample(candidates, num_pairs)  # nosec B311

                for positive in positives:
                    pair = CodePair(
                        anchor=anchor,
                        comparison=positive,
                        is_positive=True,
                        pair_type=f"same_domain_{domain}",
                    )
                    positive_pairs.append(pair)

        logger.info(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs

    def _generate_negative_pairs(
        self,
        code_samples: list[CodeSample],
        domain_groups: dict[str, list[CodeSample]],
        num_pairs: int,
    ) -> list[CodePair]:
        """Generate negative pairs (dissimilar code).

        Args:
            code_samples: All code samples
            domain_groups: Samples grouped by domain
            num_pairs: Number of negative pairs to generate

        Returns:
            List of negative CodePair objects
        """
        negative_pairs = []

        domain_list = list(domain_groups.keys())

        # Strategy 1: Cross-domain pairs (70% of negatives)
        cross_domain_count = int(num_pairs * 0.7)
        for _ in range(cross_domain_count):
            if len(domain_list) < 2:
                break

            # Pick two different domains
            domain1, domain2 = random.sample(domain_list, 2)  # nosec B311

            if not domain_groups[domain1] or not domain_groups[domain2]:
                continue

            anchor = random.choice(domain_groups[domain1])  # nosec B311
            negative = random.choice(domain_groups[domain2])  # nosec B311

            pair = CodePair(
                anchor=anchor,
                comparison=negative,
                is_positive=False,
                pair_type=f"cross_domain_{domain1}_vs_{domain2}",
            )
            negative_pairs.append(pair)

        # Strategy 2: Synthetic corruption (30% of negatives)
        corruption_count = num_pairs - len(negative_pairs)
        valid_samples = [
            s for s in code_samples if len(s.content) >= self.min_content_length
        ]

        for _ in range(corruption_count):
            if not valid_samples:
                break

            anchor = random.choice(valid_samples)  # nosec B311

            # Create corrupted version
            corrupted_content = self._corrupt_code_style(anchor.content)

            corrupted_sample = CodeSample(
                file_path=f"{anchor.file_path}_corrupted",
                content=corrupted_content,
                domain=anchor.domain,
                tokens=anchor.tokens,
            )

            pair = CodePair(
                anchor=anchor,
                comparison=corrupted_sample,
                is_positive=False,
                pair_type="synthetic_corruption",
            )
            negative_pairs.append(pair)

        logger.info(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs

    def _corrupt_code_style(self, code: str) -> str:
        """Apply synthetic style corruptions to code.

        Args:
            code: Original code content

        Returns:
            Style-corrupted code
        """
        corrupted = code

        # Random selection of corruption strategies
        corruptions = []

        # 1. Change camelCase to snake_case
        if random.random() < 0.3:  # nosec B311
            corrupted = self._camelcase_to_snake(corrupted)
            corruptions.append("camelCase→snake_case")

        # 2. Remove/add semicolons
        if random.random() < 0.3:  # nosec B311
            if ";" in corrupted:
                corrupted = corrupted.replace(";", "")
                corruptions.append("removed_semicolons")
            else:
                corrupted = re.sub(r"(\n)", r";\1", corrupted)
                corruptions.append("added_semicolons")

        # 3. Change const/let/var
        if random.random() < 0.3:  # nosec B311
            corrupted = re.sub(r"\bconst\b", "var", corrupted)
            corruptions.append("const→var")

        # 4. Change quote style
        if random.random() < 0.3:  # nosec B311
            if '"' in corrupted:
                corrupted = corrupted.replace('"', "'")
                corruptions.append("double→single_quotes")

        logger.debug(f"Applied corruptions: {corruptions}")

        return corrupted

    def _camelcase_to_snake(self, code: str) -> str:
        """Convert camelCase identifiers to snake_case.

        Args:
            code: Source code

        Returns:
            Code with snake_case identifiers
        """

        def convert_match(match: re.Match[str]) -> str:
            word = match.group(0)
            return re.sub(r"([a-z])([A-Z])", r"\1_\2", word).lower()

        pattern = r"\b[a-z]+[A-Z][a-zA-Z]*\b"
        return re.sub(pattern, convert_match, code)
