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

# Embeddings-only analysis (default mode, framework-agnostic)
config = EnhancedScorerConfig(
    enable_architectural_analysis=False,
    enable_quality_analysis=False,
    enable_typescript_analysis=False,
    enable_framework_analysis=False,
    enable_domain_adherence_analysis=True,  # Pure semantic embeddings
    build_pattern_indices=True
)
scorer = MultiDimensionalScorer(config, repo_path="/path/to/repo")
results = scorer.analyze_commit("commit_hash")

print(f"Overall Adherence: {results['overall_adherence']:.3f}")
print(f"Domain Adherence: {results['dimensional_scores']['domain_adherence']:.3f}")

# Multi-dimensional analysis with all analyzers
config = EnhancedScorerConfig(
    enable_architectural_analysis=True,
    enable_quality_analysis=True,
    enable_typescript_analysis=True,
    enable_framework_analysis=True,
    enable_domain_adherence_analysis=True
)
scorer = MultiDimensionalScorer(config, repo_path="/path/to/repo")
results = scorer.analyze_commit("commit_hash")

# Use a fine-tuned model
config = EnhancedScorerConfig(
    enable_domain_adherence_analysis=True,
    fine_tuned_model_commit="abc123d"  # Commit hash or model name
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

### Fine-Tuning Models

```python
from semantic_code_analyzer.training import (
    CodeStyleTrainer,
    FineTuningConfig,
    CodeDatasetPreparator
)

# Configure fine-tuning
config = FineTuningConfig(
    model_name="microsoft/graphcodebert-base",
    epochs=3,
    batch_size=8,
    learning_rate=5e-5,
    device="mps",  # or "cuda", "cpu", "auto"
    output_dir="./fine-tuned-models",
    max_length=512
)

# Prepare training data from commit
preparator = CodeDatasetPreparator(repo_path=".")
dataset = preparator.prepare_dataset_from_commit(
    commit_hash="HEAD",
    max_files=1000
)

# Train the model
trainer = CodeStyleTrainer(config)
model_path = trainer.train(dataset, output_name="my-custom-model")

print(f"Fine-tuned model saved to: {model_path}")

# Use the fine-tuned model for analysis
scorer_config = EnhancedScorerConfig(
    enable_domain_adherence_analysis=True,
    fine_tuned_model_commit="my-custom-model"
)
scorer = MultiDimensionalScorer(scorer_config, repo_path=".")
results = scorer.analyze_commit("new_commit_hash")
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
# Basic embeddings-only analysis (default mode)
sca-analyze analyze HEAD

# Enable verbose logging for detailed output
sca-analyze analyze HEAD --verbose

# Multi-dimensional analysis with regex analyzers
sca-analyze analyze HEAD --enable-regex-analyzers

# With custom weights (multi-dimensional mode)
sca-analyze analyze abc123def456 \
  --enable-regex-analyzers \
  --architectural-weight 0.25 \
  --quality-weight 0.30 \
  --framework-weight 0.15 \
  --domain-adherence-weight 0.15 \
  --output results.json

# Embeddings-only with custom similarity settings
sca-analyze analyze abc123def456 \
  --similarity-threshold 0.4 \
  --max-similar-patterns 15

# Compare against specific commit (default is parent)
sca-analyze analyze abc123def456 --pattern-index-commit main

# Specify hardware acceleration device
sca-analyze analyze HEAD --device mps  # Options: auto, cpu, mps, cuda

# Use a fine-tuned model
sca-analyze analyze HEAD --fine-tuned-model abc123d

# Disable AI features for speed
sca-analyze analyze abc123def456 \
  --disable-domain-adherence \
  --disable-pattern-indices
```

### Fine-Tuning (CLI)

```bash
# Fine-tune GraphCodeBERT on your codebase
sca-analyze fine-tune HEAD --repo-path . --epochs 3 --batch-size 8

# Fine-tune with custom settings
sca-analyze fine-tune HEAD \
  --repo-path ~/src/myproject \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --max-files 1000 \
  --device mps \
  --output-name my-custom-model

# Fine-tune on CPU (if GPU not available)
sca-analyze fine-tune HEAD --device cpu

# Use the fine-tuned model
sca-analyze analyze HEAD --fine-tuned-model abc123d

# Fine-tune with verbose output
sca-analyze fine-tune HEAD --verbose
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

# Compare with hardware acceleration
sca-analyze compare abc123 def456 --device mps --output comparison.json
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

### Domain-Stratified Scores

The analyzer provides two important scores in embeddings-only mode:

- **Overall Adherence**: Domain-weighted score across all file types
- **Code-Focused Score**: Weighted toward implementation code
  (backend/frontend), excluding config/docs

Domain weights:

- Backend: 1.0 (highest priority)
- Frontend: 1.0 (highest priority)
- Testing: 0.8
- Database: 0.7
- Infrastructure: 0.5
- Configuration: 0.3
- Documentation: 0.2

### Example Output

```json
{
  "commit_info": {
    "hash": "abc123",
    "message": "Add user authentication",
    "author": "John Doe",
    "files_changed": ["auth.py", "models.py", "README.md"]
  },
  "overall_adherence": 0.782,
  "code_focused_score": 0.854,
  "confidence": 0.923,
  "dimensional_scores": {
    "domain_adherence": 0.782
  },
  "domain_breakdown": {
    "backend": {
      "file_count": 2,
      "weighted_score": 0.854,
      "files": [
        {"path": "auth.py", "score": 0.876, "domain": "backend"},
        {"path": "models.py", "score": 0.832, "domain": "backend"}
      ]
    },
    "documentation": {
      "file_count": 1,
      "weighted_score": 0.412,
      "files": [
        {"path": "README.md", "score": 0.412, "domain": "documentation"}
      ]
    }
  },
  "pattern_analysis": {
    "total_patterns_found": 24,
    "pattern_confidence_avg": 0.847
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
- Fall back to CPU: `use_mps=False` or `--device cpu`

#### "Fine-tuning out of memory"

- Reduce batch size: `--batch-size 4`
- Reduce max files: `--max-files 500`
- Use CPU: `--device cpu`
- Close other memory-intensive applications

#### "Fine-tuning taking too long"

- Reduce epochs: `--epochs 2`
- Reduce max files: `--max-files 500`
- Use GPU acceleration: `--device mps` or `--device cuda`
- Training typically takes 30-45 minutes on M3

#### "Fine-tuned model not found"

- Verify model was saved successfully
- Check model directory: `~/.semantic-code-analyzer/fine-tuned-models/`
- Use correct commit hash or model name
- Re-run fine-tuning if model is missing

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
config = ScorerConfig(detailed_output=True)
```

```bash
# CLI verbose mode
sca-analyze analyze HEAD --verbose
sca-analyze fine-tune HEAD --verbose
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
