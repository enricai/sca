# Contributing to Semantic Code Analyzer

Thank you for your interest in contributing to the Semantic Code Analyzer! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** (recommended: Python 3.11 for best performance)
- **Git** for version control
- **Apple M3 system** (recommended) or any system with Python support
- **16GB+ RAM** recommended for development with ML models

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/semantic-code-analyzer.git
   cd semantic-code-analyzer
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install main dependencies
   pip install -r requirements.txt

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install package in development mode
   pip install -e .
   ```

4. **Set Up Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   # Run tests
   pytest

   # Run CLI
   sca-analyze --help
   ```

## 🧪 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Development Guidelines

#### Code Style
- **Follow PEP 8** with line length of 88 characters
- **Use Black** for code formatting: `black semantic_code_analyzer/`
- **Use Ruff** for linting: `ruff semantic_code_analyzer/`
- **Use isort** for import sorting: `isort semantic_code_analyzer/`

#### Type Hints
- **Use type hints** for all function signatures
- **Use mypy** for type checking: `mypy semantic_code_analyzer/`

#### Documentation
- **Write docstrings** for all public functions and classes
- **Use Google-style docstrings**
- **Update documentation** when adding new features

#### Testing
- **Write tests** for all new functionality
- **Maintain 90%+ test coverage**
- **Use pytest** for testing framework
- **Mock external dependencies** (especially ML models)

### 3. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_code_analyzer --cov-report=html

# Run specific test files
pytest tests/test_semantic_scorer.py

# Run tests in parallel
pytest -n auto

# Run performance benchmarks
pytest --benchmark-only
```

### 4. Code Quality Checks

```bash
# Format code
black semantic_code_analyzer/ tests/ examples/

# Sort imports
isort semantic_code_analyzer/ tests/ examples/

# Lint code
ruff semantic_code_analyzer/ tests/

# Type checking
mypy semantic_code_analyzer/

# Security scanning
bandit -r semantic_code_analyzer/

# Run all checks
pre-commit run --all-files
```

### 5. Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Feature
git commit -m "feat: add support for Rust language analysis"

# Bug fix
git commit -m "fix: resolve embedding cache corruption issue"

# Documentation
git commit -m "docs: update API documentation for SemanticScorer"

# Tests
git commit -m "test: add integration tests for CLI commands"

# Refactor
git commit -m "refactor: optimize similarity calculation algorithm"

# Performance
git commit -m "perf: improve batch embedding generation speed"
```

## 📝 Contribution Types

### 🐛 Bug Reports

Before submitting a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Use the latest version** of the package
3. **Provide minimal reproduction** steps

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- Package Version: [e.g., 0.1.0]
- Hardware: [e.g., Apple M3, Intel x86_64]

## Reproduction Steps
1. Step one
2. Step two
3. Expected vs actual behavior

## Additional Context
- Error messages
- Log output
- Screenshots (if applicable)
```

### ✨ Feature Requests

For feature requests, please:

1. **Check existing issues** and discussions
2. **Explain the use case** and motivation
3. **Provide examples** of how it would work

**Feature Request Template:**
```markdown
## Feature Description
Brief description of the proposed feature

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Implementation
How would this feature work? Any API design ideas?

## Additional Context
- Similar features in other tools
- Alternative solutions considered
- Impact on existing functionality
```

### 🔧 Code Contributions

#### Areas for Contribution

1. **Core Features**
   - New language support
   - Additional distance metrics
   - Performance optimizations
   - Memory efficiency improvements

2. **Testing**
   - Integration tests
   - Performance benchmarks
   - Edge case coverage
   - Mock improvements

3. **Documentation**
   - API documentation
   - Usage examples
   - Performance guides
   - Troubleshooting guides

4. **Developer Experience**
   - IDE integrations
   - CLI improvements
   - Error messages
   - Logging enhancements

5. **Infrastructure**
   - CI/CD improvements
   - Docker support
   - Package distribution
   - Security enhancements

## 🧠 Architecture Overview

### Core Components

```
semantic_code_analyzer/
├── commit_extractor.py     # Git integration and code extraction
├── code_embedder.py        # ML model integration with MPS acceleration
├── similarity_calculator.py # Distance metrics and similarity algorithms
├── semantic_scorer.py      # Main orchestration and API
└── cli.py                 # Command-line interface
```

### Key Design Principles

1. **Modular Architecture** - Each component has clear responsibilities
2. **Apple M3 Optimization** - MPS acceleration throughout
3. **Extensibility** - Easy to add new languages and metrics
4. **Performance** - Caching and batch processing optimizations
5. **Testing** - Comprehensive test coverage with mocking

### Adding New Features

#### Adding a New Programming Language

1. **Update `SUPPORTED_EXTENSIONS`** in `commit_extractor.py`
2. **Add language-specific preprocessing** in `code_embedder.py`
3. **Add AST parsing support** (if applicable)
4. **Update tests** and documentation
5. **Add examples** for the new language

#### Adding a New Distance Metric

1. **Add enum value** to `DistanceMetric` in `similarity_calculator.py`
2. **Implement distance function** following naming convention
3. **Add to `_distance_functions` mapping**
4. **Add comprehensive tests**
5. **Update CLI options** and documentation

## 🚨 Common Issues and Solutions

### Development Environment

**Issue**: Model download fails
```bash
# Solution: Check internet connection and disk space
pip install transformers --upgrade
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/graphcodebert-base')"
```

**Issue**: MPS not available on Apple Silicon
```bash
# Solution: Update PyTorch
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cpu
```

**Issue**: Out of memory during testing
```bash
# Solution: Use smaller test datasets or mock models
pytest tests/ -k "not integration"
```

### Code Quality

**Issue**: Pre-commit hooks failing
```bash
# Solution: Run fixes manually
black .
isort .
ruff --fix .
```

**Issue**: Type checking errors
```bash
# Solution: Add type ignore for complex ML types
# type: ignore[attr-defined]
```

## 📚 Resources

### Documentation
- [README.md](README.md) - Project overview and usage
- [API Documentation](semantic_code_analyzer/) - Detailed API reference
- [Examples](examples/) - Usage examples and tutorials

### External Resources
- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html)

### Community
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and community discussions
- **Code Reviews** - Learning from peer feedback

## 🏆 Recognition

Contributors will be recognized in:
- **CHANGELOG.md** - All contributions noted in release notes
- **README.md** - Major contributors listed
- **GitHub Contributors** - Automatic recognition

## 📧 Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Code Reviews** - For implementation guidance

## 🚀 Release Process

### For Maintainers

1. **Update version** in `pyproject.toml` and `__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Create release tag**: `git tag v0.2.0`
4. **Build package**: `python -m build`
5. **Upload to PyPI**: `twine upload dist/*`

### Testing Releases

```bash
# Build package locally
python -m build

# Install from local build
pip install dist/semantic_code_analyzer-*.whl

# Test installation
sca-analyze --help
```

## 🎯 Roadmap

### Short Term (v0.2.0)
- Enhanced language support (JavaScript, Java, C++)
- Performance optimizations
- Better error handling
- IDE integrations

### Medium Term (v0.3.0)
- Custom model fine-tuning
- API server mode
- Real-time analysis
- Advanced visualization

### Long Term (v1.0.0)
- Enterprise features
- Team collaboration tools
- Machine learning improvements
- Cloud deployment options

---

Thank you for contributing to Semantic Code Analyzer! Your contributions help make code analysis better for everyone. 🚀