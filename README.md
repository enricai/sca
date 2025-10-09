# Multi-Dimensional Code Analyzer (SCA)

A comprehensive tool for analyzing code quality through multi-dimensional
pattern recognition. Evaluates architectural adherence, code quality,
TypeScript usage, and framework-specific patterns to provide meaningful
insights into code implementation quality.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Multi--Dimensional-green.svg)](https://github.com/sca/semantic-code-analyzer)

## 🎯 Purpose

Distinguish between high-quality, thoughtful code and basic implementations
that merely compile. While traditional similarity analysis often shows only 5%
difference between quality levels, our multi-dimensional approach reveals
20-40% differences through pattern analysis.

## 🚀 Features

- **🏗️ Architectural Analysis**: File structure, import patterns, Next.js conventions
- **✨ Code Quality Assessment**: React best practices, security patterns, accessibility
- **🔷 TypeScript Analysis**: Type safety, advanced patterns, strict typing adherence
- **⚡ Framework Recognition**: Next.js features, API routes, integration patterns
- **🧠 Domain-Aware Adherence**: AI-powered pattern matching using
  GraphCodeBERT embeddings
- **📊 Weighted Scoring**: Configurable importance weights for different dimensions
- **💡 Actionable Feedback**: Specific recommendations for code improvement
- **🔍 Pattern Recognition**: Framework-aware analysis for modern web development
- **🚀 FAISS Similarity Search**: Fast semantic pattern matching within
  architectural domains
- **⚡ Smart Analysis**: Lightweight pattern analysis with optional AI-enhanced features

## 📊 Results Comparison

**Traditional Semantic Similarity**:

```text
Your Code:     0.734 similarity
AI Code:       0.697 similarity
Difference:    0.037 (5% better)
```

**Enhanced Multi-Dimensional Analysis**:

```text
Your Code:     0.850 adherence
AI Code:       0.700 adherence
Difference:    0.150 (21% better)
```

## 🛠 Installation

### Prerequisites

- Python 3.9 or higher
- Git repository
- TypeScript/JavaScript codebase (React/Next.js optimized)

### Install

```bash
pip install semantic-code-analyzer
```

### Development Install

```bash
git clone https://github.com/sca/semantic-code-analyzer
cd semantic-code-analyzer
pip install -e .
```

## 🚀 Quick Start

### Analyze a Single Commit

```bash
# Analyze latest commit (default: semantic embeddings only)
sca-analyze analyze HEAD

# Analyze specific commit (compares against its parent)
sca-analyze analyze abc123def

# Compare against a specific commit (e.g., main branch)
sca-analyze analyze abc123def --pattern-index-commit main

# Enable regex-based analyzers (multi-dimensional mode)
sca-analyze analyze abc123def --enable-regex-analyzers

# Custom configuration (embeddings-only)
sca-analyze analyze HEAD \
  --similarity-threshold 0.4 \
  --max-similar-patterns 15
```

### Compare Multiple Commits

```bash
# Compare your implementation vs AI implementations
sca-analyze compare \
  --base-commit your_commit_hash \
  --compare-commits ai_commit1,ai_commit2,ai_commit3
```

### Programmatic Usage

```python
from semantic_code_analyzer import MultiDimensionalScorer, EnhancedScorerConfig

# Configure analysis
config = EnhancedScorerConfig(
    architectural_weight=0.25,
    quality_weight=0.25,
    typescript_weight=0.25,
    framework_weight=0.15,
    domain_adherence_weight=0.10
)

# Analyze commit
scorer = MultiDimensionalScorer(config, repo_path=".")
results = scorer.analyze_commit("commit_hash")

print(f"Overall adherence: {results['overall_adherence']:.3f}")
print(f"Confidence: {results['confidence']:.3f}")

# Show dimensional breakdown
for dimension, score in results['dimensional_scores'].items():
    print(f"{dimension}: {score:.3f}")

# Get actionable recommendations
for feedback in results['actionable_feedback']:
    print(f"• {feedback['message']}")
    if feedback['suggested_fix']:
        print(f"  Fix: {feedback['suggested_fix']}")
```

## 📊 Analysis Dimensions

### 🏗️ Architectural Analysis (Default: 25%)

- Next.js app router structure and conventions
- Import organization (absolute vs relative)
- File naming and directory structure
- Code organization and modularity

### ✨ Quality Analysis (Default: 25%)

- React component best practices
- Error handling patterns
- Security considerations
- Accessibility implementation
- Performance optimization patterns

### 🔷 TypeScript Analysis (Default: 25%)

- Type safety and explicit typing
- Interface and type definitions
- Generic usage and advanced patterns
- Strict typing adherence

### ⚡ Framework Analysis (Default: 15%)

- Next.js specific features (metadata, API routes)
- React patterns (hooks, context, performance)
- Integration patterns (i18n, authentication)
- Framework conventions

### 🧠 Domain-Aware Adherence Analysis (Default: 10%)

- **GraphCodeBERT Embeddings**: Semantic code understanding using
  state-of-the-art transformer models
- **Architectural Domain Classification**: Automatically categorizes code into
  frontend, backend, testing, or database domains
- **FAISS Similarity Search**: Fast similarity search within domain-specific
  pattern indices
- **Pattern Matching**: Finds similar high-quality patterns in your existing
  codebase
- **Parent Commit Comparison**: By default, compares new code against patterns
  from the commit **before** it was made (ensures fair adherence measurement)
- **Context-Aware Recommendations**: Suggestions based on actual patterns from
  your domain-specific code
- **Confidence Scoring**: Measures certainty of pattern matches and domain classification

**Pattern Index Commit Options:**
- `--pattern-index-commit parent` (default): Compare against the parent commit's codebase
- `--pattern-index-commit main`: Compare against the main branch patterns
- `--pattern-index-commit HEAD`: Compare against current repository state

## 🎛️ Configuration

### Weight Configuration

Customize analysis focus based on project priorities:

```python
# Frontend-heavy project
config = EnhancedScorerConfig(
    architectural_weight=0.20,
    quality_weight=0.20,
    typescript_weight=0.25,
    framework_weight=0.25,     # Equal emphasis
    domain_adherence_weight=0.10
)

# API-heavy project
config = EnhancedScorerConfig(
    architectural_weight=0.30,  # Structure important
    quality_weight=0.35,       # Security critical
    typescript_weight=0.25,    # Type safety
    framework_weight=0.05,     # Less framework patterns
    domain_adherence_weight=0.05
)

# Type-safety focused
config = EnhancedScorerConfig(
    architectural_weight=0.15,
    quality_weight=0.20,
    typescript_weight=0.45,    # Heavy TypeScript emphasis
    framework_weight=0.10,
    domain_adherence_weight=0.10
)

# AI-enhanced analysis (emphasize domain patterns)
config = EnhancedScorerConfig(
    architectural_weight=0.20,
    quality_weight=0.20,
    typescript_weight=0.20,
    framework_weight=0.15,
    domain_adherence_weight=0.25  # Heavy pattern matching
)
```

### Analysis Options

```python
config = EnhancedScorerConfig(
    # Enable/disable specific analyzers
    enable_architectural_analysis=True,
    enable_quality_analysis=True,
    enable_typescript_analysis=True,
    enable_framework_analysis=True,
    enable_domain_adherence_analysis=True,

    # Domain adherence configuration
    similarity_threshold=0.3,          # Minimum similarity for pattern matches
    max_similar_patterns=10,           # Max patterns to consider per analysis
    build_pattern_indices=True,        # Enable automatic pattern indexing

    # Output configuration
    include_actionable_feedback=True,
    include_pattern_details=True,
    max_recommendations_per_file=10,

    # File filtering
    include_test_files=False,
    include_generated_files=False
)
```

## 📈 Output Format

```json
{
  "overall_adherence": 0.834,
  "confidence": 0.945,
  "dimensional_scores": {
    "architectural": 0.828,
    "quality": 0.759,
    "typescript": 0.889,
    "framework": 0.903,
    "domain_adherence": 0.741
  },
  "pattern_analysis": {
    "total_patterns_found": 51,
    "pattern_confidence_avg": 0.847,
    "patterns_by_type": {
      "architectural": 12,
      "component": 15,
      "type_safety": 18,
      "framework": 6,
      "domain_adherence": 8
    },
    "domain_classification": {
      "frontend_files": 23,
      "backend_files": 12,
      "testing_files": 8,
      "database_files": 3,
      "unknown_files": 2
    }
  },
  "actionable_feedback": [
    {
      "severity": "warning",
      "category": "react_patterns",
      "message": "Add key prop to list items for better performance",
      "file": "src/components/UserList.tsx",
      "line": 45,
      "suggested_fix": "Add key={item.id} to list items",
      "rule_id": "MISSING_KEY_PROP"
    }
  ]
}
```

## 🎯 Use Cases

### Code Review Automation

```bash
# Check if commit meets quality standards
sca-analyze analyze $COMMIT_HASH
if [ $? -eq 0 ]; then
    echo "✅ Code quality check passed"
else
    echo "❌ Code quality issues found"
fi
```

### Implementation Comparison

```bash
# Compare your implementation vs AI-generated code
sca-analyze compare \
  --base-commit your_implementation \
  --compare-commits claude_impl,gpt_impl,copilot_impl
```

### CI/CD Integration

```python
# Quality gate in CI/CD pipeline
def quality_gate(commit_hash: str) -> bool:
    config = EnhancedScorerConfig(quality_weight=0.4)  # Emphasize quality
    scorer = MultiDimensionalScorer(config)
    results = scorer.analyze_commit(commit_hash)

    return results['overall_adherence'] >= 0.7  # 70% threshold
```

## 🏆 Analysis Modes

### Embeddings-Only Mode (Default) ⭐
Pure style matching using only GraphCodeBERT semantic embeddings:
- Domain-aware comparison (frontend vs frontend, backend vs backend)
- Learns YOUR coding style from parent commit
- Framework-agnostic (works with any language)
- No subjective regex rules
- 100% based on semantic similarity to YOUR code
- Analyzes only changed lines (not entire files)

### Multi-Dimensional Mode (`--enable-regex-analyzers`)
Combines semantic embeddings with regex-based pattern analyzers:
- 5 specialized analyzers (architectural, quality, TypeScript, framework, domain adherence)
- Checks against hardcoded best practices
- Framework-specific (Next.js/React/TypeScript)
- Useful for best-practice compliance checking

## 🏆 Benefits Over Traditional Analysis

| Traditional Semantic Similarity | Embeddings-Only Mode | Multi-Dimensional Mode |
|--------------------------------|---------------------|------------------------|
| Single embedding-based score | Domain-aware embeddings | 5 specialized analyzers |
| 5% differentiation | 15-25% differentiation | 20-40% differentiation |
| No actionable feedback | Similarity-based feedback | Specific rule-based recommendations |
| Framework-agnostic | Framework-agnostic ✅ | Next.js/React/TS specific |
| ML overhead (15-30s init) | ML overhead (15-30s init) | Fast regex + optional AI |
| Dominated by boilerplate | Changed lines only ✅ | Focuses on quality patterns |
| Generic pattern matching | Domain-aware ✅ | Domain-aware + hardcoded rules |

## 📚 Examples

See `examples/` directory for comprehensive usage examples:

- `basic_usage.py` - Getting started guide
- `advanced_usage.py` - Advanced features and configurations

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

## 🔗 Links

- [Documentation](https://semantic-code-analyzer.readthedocs.io/)
- [Issues](https://github.com/sca/semantic-code-analyzer/issues)
- [Repository](https://github.com/sca/semantic-code-analyzer)

---

**Transform your code quality analysis from basic similarity checking to
comprehensive pattern recognition.** 🚀
