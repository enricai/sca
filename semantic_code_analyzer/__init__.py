"""
Semantic Code Analyzer - A tool for analyzing semantic similarity of code commits.

This package provides tools to:
- Extract code changes from Git commits
- Generate semantic embeddings for code using state-of-the-art models
- Calculate similarity scores between commits and existing codebase
- Optimize performance for Apple M3 hardware acceleration

Main Components:
- CommitExtractor: Git integration and code extraction
- CodeEmbedder: Semantic embedding generation with MPS acceleration
- SimilarityCalculator: Distance metrics and similarity scoring
- SemanticScorer: Main orchestration class

Example Usage:
    from semantic_code_analyzer import SemanticScorer

    scorer = SemanticScorer("/path/to/repo")
    results = scorer.score_commit_similarity("commit_hash")
    print(f"Similarity score: {results['aggregate_max_similarity']:.3f}")
"""

__version__ = "0.1.0"
__author__ = "SCA Team"

from .commit_extractor import CommitExtractor
from .code_embedder import CodeEmbedder
from .similarity_calculator import SimilarityCalculator
from .semantic_scorer import SemanticScorer

__all__ = [
    "CommitExtractor",
    "CodeEmbedder",
    "SimilarityCalculator",
    "SemanticScorer",
]