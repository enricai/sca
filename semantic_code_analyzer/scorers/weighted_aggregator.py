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

"""Weighted aggregator for combining scores from multiple dimensions.

This module handles the mathematical aggregation of scores from different
analyzers using configurable weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Results from weighted aggregation of dimensional scores."""

    overall_score: float
    dimensional_scores: dict[str, float]
    weighted_contributions: dict[str, float]
    confidence: float
    metadata: dict[str, Any]


class WeightedAggregator:
    """Aggregates scores from multiple dimensions using weighted averaging.

    Supports different aggregation methods and handles missing dimensions
    gracefully by normalizing weights.
    """

    def __init__(self, aggregation_method: str = "weighted_average"):
        """Initialize the aggregator.

        Args:
            aggregation_method: Method to use for aggregation
                - "weighted_average": Simple weighted average
                - "harmonic_mean": Weighted harmonic mean (penalizes low scores)
                - "geometric_mean": Weighted geometric mean
        """
        self.aggregation_method = aggregation_method
        self.supported_methods = {
            "weighted_average": self._weighted_average,
            "harmonic_mean": self._weighted_harmonic_mean,
            "geometric_mean": self._weighted_geometric_mean,
        }

        if aggregation_method not in self.supported_methods:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    def aggregate(
        self, dimensional_scores: dict[str, float], weights: dict[str, float]
    ) -> AggregatedResult:
        """Aggregate dimensional scores using the configured method.

        Args:
            dimensional_scores: Dictionary mapping dimension names to scores (0-1)
            weights: Dictionary mapping dimension names to weights

        Returns:
            AggregatedResult with overall score and detailed breakdown
        """
        # Validate inputs
        self._validate_inputs(dimensional_scores, weights)

        # Filter to only dimensions that have both scores and weights
        available_dimensions = set(dimensional_scores.keys()) & set(weights.keys())

        if not available_dimensions:
            logger.warning("No dimensions available for aggregation")
            return AggregatedResult(
                overall_score=0.0,
                dimensional_scores={},
                weighted_contributions={},
                confidence=0.0,
                metadata={"error": "No dimensions available"},
            )

        # Normalize weights for available dimensions
        normalized_weights = self._normalize_weights(weights, available_dimensions)

        # Extract scores and weights for available dimensions
        scores = {dim: dimensional_scores[dim] for dim in available_dimensions}
        norm_weights = {dim: normalized_weights[dim] for dim in available_dimensions}

        # Perform aggregation
        aggregation_func = self.supported_methods[self.aggregation_method]
        overall_score = aggregation_func(scores, norm_weights)

        # Calculate weighted contributions
        weighted_contributions = {
            dim: scores[dim] * norm_weights[dim] for dim in available_dimensions
        }

        # Calculate confidence based on score variance and weight distribution
        confidence = self._calculate_confidence(scores, norm_weights)

        # Create metadata
        metadata = {
            "aggregation_method": self.aggregation_method,
            "dimensions_used": list(available_dimensions),
            "dimensions_missing": list(set(weights.keys()) - available_dimensions),
            "weight_normalization_applied": len(available_dimensions) != len(weights),
            "score_statistics": self._calculate_score_statistics(scores),
        }

        return AggregatedResult(
            overall_score=overall_score,
            dimensional_scores=scores,
            weighted_contributions=weighted_contributions,
            confidence=confidence,
            metadata=metadata,
        )

    def _validate_inputs(
        self, dimensional_scores: dict[str, float], weights: dict[str, float]
    ) -> None:
        """Validate input parameters.

        Args:
            dimensional_scores: Dimensional scores to validate
            weights: Weights to validate

        Raises:
            ValueError: If inputs are invalid
        """
        # Check scores are in valid range
        for dim, score in dimensional_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Score for {dim} must be between 0 and 1, got {score}"
                )

        # Check weights are non-negative
        for dim, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Weight for {dim} must be non-negative, got {weight}")

        # Check at least one weight is positive
        if all(weight == 0 for weight in weights.values()):
            raise ValueError("At least one weight must be positive")

    def _normalize_weights(
        self, weights: dict[str, float], available_dimensions: set[str]
    ) -> dict[str, float]:
        """Normalize weights to sum to 1.0 for available dimensions.

        Args:
            weights: Original weights
            available_dimensions: Dimensions that have scores

        Returns:
            Normalized weights for available dimensions
        """
        # Get weights for available dimensions only
        available_weights = {dim: weights[dim] for dim in available_dimensions}

        # Calculate sum of available weights
        weight_sum = sum(available_weights.values())

        if weight_sum == 0:
            # Equal weights if all weights are zero
            equal_weight = 1.0 / len(available_dimensions)
            return dict.fromkeys(available_dimensions, equal_weight)

        # Normalize to sum to 1.0
        return {dim: weight / weight_sum for dim, weight in available_weights.items()}

    def _weighted_average(
        self, scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Calculate weighted average of scores."""
        return sum(scores[dim] * weights[dim] for dim in scores.keys())

    def _weighted_harmonic_mean(
        self, scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Calculate weighted harmonic mean of scores.

        Harmonic mean penalizes low scores more than arithmetic mean,
        useful when all dimensions should perform well.
        """
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-8
        adjusted_scores = {dim: max(score, epsilon) for dim, score in scores.items()}

        weighted_reciprocal_sum = sum(
            weights[dim] / adjusted_scores[dim] for dim in scores.keys()
        )

        return 1.0 / weighted_reciprocal_sum

    def _weighted_geometric_mean(
        self, scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Calculate weighted geometric mean of scores.

        Geometric mean also penalizes low scores but less severely than harmonic mean.
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        adjusted_scores = {dim: max(score, epsilon) for dim, score in scores.items()}

        log_sum = sum(
            weights[dim] * np.log(adjusted_scores[dim]) for dim in scores.keys()
        )

        return np.exp(log_sum)

    def _calculate_confidence(
        self, scores: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Calculate confidence score based on score variance and weight distribution.

        Args:
            scores: Dimensional scores
            weights: Normalized weights

        Returns:
            Confidence score between 0 and 1
        """
        if len(scores) < 2:
            return 1.0 if scores else 0.0

        score_values = list(scores.values())
        weight_values = list(weights.values())

        # Calculate score consistency (low variance = high confidence)
        score_variance = float(np.var(score_values, dtype=np.float64))
        consistency_factor = 1.0 / (1.0 + score_variance)

        # Calculate weight distribution (more balanced = higher confidence)
        weight_entropy = -sum(w * np.log(w + 1e-8) for w in weight_values if w > 0)
        max_entropy = np.log(len(weight_values))
        distribution_factor = weight_entropy / max_entropy if max_entropy > 0 else 1.0

        # Calculate average score level (higher scores = higher confidence)
        avg_score = float(np.mean(score_values, dtype=np.float64))
        score_level_factor = avg_score

        # Combine factors with weights
        confidence = (
            0.4 * consistency_factor
            + 0.3 * distribution_factor
            + 0.3 * score_level_factor
        )

        return min(1.0, max(0.0, float(confidence)))

    def _calculate_score_statistics(self, scores: dict[str, float]) -> dict[str, float]:
        """Calculate basic statistics for the scores."""
        if not scores:
            return {}

        score_values = list(scores.values())
        return {
            "mean": float(np.mean(score_values, dtype=np.float64)),
            "median": float(np.median(score_values)),
            "std": float(np.std(score_values, dtype=np.float64)),
            "min": float(np.min(score_values)),
            "max": float(np.max(score_values)),
            "range": float(np.max(score_values) - np.min(score_values)),
        }

    def get_dimension_importance(
        self, aggregated_result: AggregatedResult
    ) -> dict[str, float]:
        """Calculate the relative importance of each dimension to the overall score.

        Args:
            aggregated_result: Result from aggregation

        Returns:
            Dictionary mapping dimensions to their importance scores
        """
        contributions = aggregated_result.weighted_contributions
        total_contribution = sum(abs(c) for c in contributions.values())

        if total_contribution == 0:
            return {}

        return {
            dim: abs(contribution) / total_contribution
            for dim, contribution in contributions.items()
        }

    def explain_score(self, aggregated_result: AggregatedResult) -> dict[str, Any]:
        """Provide detailed explanation of how the overall score was calculated.

        Args:
            aggregated_result: Result from aggregation

        Returns:
            Detailed explanation of score calculation
        """
        importance = self.get_dimension_importance(aggregated_result)

        explanation: dict[str, Any] = {
            "overall_score": aggregated_result.overall_score,
            "confidence": aggregated_result.confidence,
            "method": aggregated_result.metadata.get("aggregation_method"),
            "dimension_breakdown": [],
        }
        breakdown_list = explanation["dimension_breakdown"]

        for dim, score in aggregated_result.dimensional_scores.items():
            contribution = aggregated_result.weighted_contributions.get(dim, 0)
            dim_importance = importance.get(dim, 0)

            breakdown_list.append(
                {
                    "dimension": dim,
                    "score": score,
                    "weighted_contribution": contribution,
                    "importance_percentage": dim_importance * 100,
                    "interpretation": self._interpret_score(score),
                }
            )

        # Sort by importance
        breakdown_list.sort(key=lambda x: x["importance_percentage"], reverse=True)

        return explanation

    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of a score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Needs Improvement"
        else:
            return "Poor"
