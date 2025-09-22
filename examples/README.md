# Multi-Dimensional Code Analyzer - Examples

This directory contains comprehensive examples demonstrating how to use the
Multi-Dimensional Code Analyzer (SCA) for various use cases with domain-aware
adherence analysis.

## 📁 Example Files

### `basic_usage.py`

Demonstrates fundamental SCA operations:

- Single commit analysis
- Batch commit analysis
- Commit comparison
- Multi-language support
- Distance metric comparison
- Scorer configuration

**Run it:**

```bash
cd examples
python basic_usage.py
```

### `advanced_usage.py`

Shows advanced features and integration patterns:

- Custom configurations for different scenarios
- Performance optimization techniques
- Function-level analysis
- CI/CD integration patterns
- Custom analysis workflows
- Cross-repository comparison

**Run it:**

```bash
cd examples
python advanced_usage.py
```

## 🎯 Use Case Examples

### Basic Analysis

```python
from semantic_code_analyzer import MultiDimensionalScorer, EnhancedScorerConfig

# Multi-dimensional analysis with default configuration
config = EnhancedScorerConfig()
scorer = MultiDimensionalScorer(config, repo_path="/path/to/repo")
results = scorer.analyze_commit("commit_hash")

print(f"Overall Adherence: {results['overall_adherence']:.3f}")
print(f"Domain Adherence: {results['dimensional_scores']['domain_adherence']:.3f}")

# Basic analysis without AI features
config = EnhancedScorerConfig(
    enable_domain_adherence_analysis=False,
    build_pattern_indices=False
)
scorer = MultiDimensionalScorer(config, repo_path="/path/to/repo")
results = scorer.analyze_commit("commit_hash")
```

### Custom Configuration

```python
# Domain-aware configuration with AI features
config = EnhancedScorerConfig(
    architectural_weight=0.20,
    quality_weight=0.30,
    typescript_weight=0.25,
    framework_weight=0.15,
    domain_adherence_weight=0.10,
    similarity_threshold=0.4,
    max_similar_patterns=15,
    build_pattern_indices=True
)

scorer = MultiDimensionalScorer(config, repo_path="/path/to/repo")
```

### Batch Analysis

```python
# Analyze multiple commits
commit_hashes = ["abc123", "def456", "789ghi"]
for commit_hash in commit_hashes:
    results = scorer.analyze_commit(commit_hash)
    print(f"{commit_hash}: {results['overall_adherence']:.3f}")
    print(f"  - Architectural: {results['dimensional_scores']['architectural']:.3f}")
    print(f"  - Quality: {results['dimensional_scores']['quality']:.3f}")
    print(f"  - Domain Adherence: {results['dimensional_scores']['domain_adherence']:.3f}")
```

### Performance Optimization

```python
# Fast analysis configuration
fast_config = ScorerConfig(
    distance_metric="euclidean",  # Fastest metric
    max_files=20,
    cache_embeddings=True,
    normalize_embeddings=False
)
```

## 📊 CLI Examples

### Single Commit Analysis

```bash
# Basic multi-dimensional analysis
sca-analyze analyze HEAD

# With custom weights
sca-analyze analyze abc123def456 \
  --architectural-weight 0.25 \
  --quality-weight 0.30 \
  --framework-weight 0.15 \
  --domain-adherence-weight 0.15 \
  --output results.json

# Enable AI-enhanced domain adherence
sca-analyze analyze abc123def456 \
  --domain-adherence-weight 0.25 \
  --similarity-threshold 0.4 \
  --max-similar-patterns 15

# Disable AI features for speed
sca-analyze analyze abc123def456 \
  --disable-domain-adherence \
  --disable-pattern-indices
```

### Multi-Commit Analysis

```bash
# Analyze last 10 commits
sca-analyze batch --count 10

# Analyze specific branch
sca-analyze batch --branch develop --count 5 --output batch_results.json
```

### Commit Comparison

```bash
# Compare two commits
sca-analyze compare abc123 def456 --output comparison.json
```

### Repository Information

```bash
# Get repo and configuration info
sca-analyze info
```

## 🔧 Configuration Examples

### For Different Scenarios

#### Large Codebase Analysis

```python
large_codebase_config = ScorerConfig(
    max_files=500,
    use_mps=True,
    cache_embeddings=True,
    detailed_output=False,  # Reduce output
    distance_metric="euclidean"
)
```

#### Detailed Code Review

```python
detailed_config = ScorerConfig(
    max_files=50,
    include_functions=True,
    detailed_output=True,
    distance_metric="cosine"
)
```

#### CI/CD Integration

```python
ci_config = ScorerConfig(
    max_files=100,
    cache_embeddings=True,
    detailed_output=False,
    save_results=True
)
```

#### Performance Benchmarking

```python
benchmark_config = ScorerConfig(
    cache_embeddings=False,  # Fresh calculations
    normalize_embeddings=True,
    use_mps=True,
    distance_metric="euclidean"
)
```

## 🎨 Output Interpretation

### Similarity Scores

| Score | Interpretation |
|-------|---------------|
| 0.8-1.0 | Very High - Follows existing patterns closely |
| 0.6-0.8 | Good - Reasonably consistent with codebase |
| 0.4-0.6 | Moderate - Some alignment but notable differences |
| 0.2-0.4 | Low - Different patterns from existing code |
| 0.0-0.2 | Very Low - Significantly different approach |

### Example Output

```json
{
  "commit_info": {
    "hash": "abc123",
    "message": "Add user authentication",
    "author": "John Doe",
    "files_changed": ["auth.py", "models.py"]
  },
  "aggregate_scores": {
    "max_similarity": 0.756,
    "mean_similarity": 0.623,
    "median_similarity": 0.698
  },
  "file_results": {
    "auth.py": {
      "overall_similarity": {
        "max_similarity": 0.756,
        "mean_similarity": 0.623
      },
      "most_similar_files": [
        {"file_path": "login.py", "similarity_score": 0.756},
        {"file_path": "security.py", "similarity_score": 0.698}
      ]
    }
  }
}
```

## 🔄 Integration Patterns

### Git Hooks

```bash
#!/bin/bash
# pre-commit hook
sca-analyze $(git rev-parse HEAD) --threshold 0.3
if [ $? -ne 0 ]; then
    echo "Commit similarity too low - consider code review"
    exit 1
fi
```

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Semantic Code Analysis
  run: |
    pip install semantic-code-analyzer
    sca-analyze ${{ github.sha }} --output sca_results.json

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: semantic-analysis
    path: sca_results.json
```

### Python Integration

```python
def check_commit_quality(commit_hash: str, threshold: float = 0.3):
    """Check if commit meets quality standards."""
    scorer = SemanticScorer(".")
    result = scorer.score_commit_similarity(commit_hash)

    similarity = result.aggregate_scores['max_similarity']
    return {
        'passed': similarity >= threshold,
        'score': similarity,
        'recommendation': get_recommendation(similarity)
    }
```

## 🚀 Performance Tips

### Optimization Strategies

1. **Enable Caching**: Set `cache_embeddings=True` for repeated analyses
2. **Limit File Count**: Use `max_files` for large repositories
3. **Use MPS**: Enable `use_mps=True` on Apple Silicon
4. **Choose Fast Metrics**: Use `euclidean` distance for speed
5. **Batch Processing**: Analyze multiple commits together

### Memory Management

```python
# For large repositories
config = ScorerConfig(
    max_files=200,  # Limit memory usage
    cache_embeddings=True,
    use_mps=True
)

# Clear caches periodically
scorer.clear_caches()
```

## 🐛 Troubleshooting

### Common Issues

#### "No commits found"

- Ensure you're in a Git repository
- Check that the repository has commits
- Verify the commit hash exists

#### "Model loading failed"

- Check internet connection for model download
- Verify sufficient disk space
- Ensure compatible PyTorch version

#### "Out of memory"

- Reduce `max_files` parameter
- Enable `cache_embeddings=False` temporarily
- Use smaller batch sizes

#### "MPS not available"

- Verify you're on Apple Silicon
- Check PyTorch MPS support: `torch.backends.mps.is_available()`
- Fall back to CPU: `use_mps=False`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
config = ScorerConfig(detailed_output=True)
```

## 📚 Additional Resources

- [Main README](../README.md) - Project overview and installation
- [API Documentation](../semantic_code_analyzer/) - Detailed API reference
- [Test Suite](../tests/) - Comprehensive tests and examples
- [Configuration Guide](../semantic_code_analyzer/README.md) - Advanced
  configuration options

## 🤝 Contributing Examples

Found a useful pattern or use case? Contribute additional examples:

1. Create a new example file
2. Follow the existing structure and documentation style
3. Include error handling and clear explanations
4. Add tests if applicable
5. Submit a pull request

---

## Happy analyzing! 🚀
