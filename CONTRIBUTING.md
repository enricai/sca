# Contributing to Multi-Dimensional Code Analyzer

Thank you for your interest in contributing to the Multi-Dimensional Code
Analyzer! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all
contributors. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (required for ML library compatibility)
- **Git** for version control
- **TypeScript/JavaScript codebase** for optimal analysis results

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
- **Mock external dependencies** (especially git operations)

### 3. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_code_analyzer --cov-report=html

# Run specific test files
pytest tests/test_multi_dimensional_analysis.py

# Run tests in parallel
pytest -n auto
```

### 4. Code Quality Checks

```bash
# Format code
black semantic_code_analyzer/ tests/ examples/

# Sort imports
isort semantic_code_analyzer/ tests/ examples/

# Lint code
ruff check semantic_code_analyzer/ tests/ examples/

# Type checking
mypy semantic_code_analyzer/ tests/ examples/

# Run all checks
pre-commit run --all-files
```

### 5. Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Feature
git commit -m "feat: add new pattern analyzer for Vue.js"

# Bug fix
git commit -m "fix: resolve TypeScript analyzer false positives"

# Documentation
git commit -m "docs: update API documentation for analyzers"

# Tests
git commit -m "test: add integration tests for CLI commands"

# Refactor
git commit -m "refactor: optimize pattern matching algorithms"

# Performance
git commit -m "perf: improve batch analysis speed"
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
- Package Version: [e.g., 0.3.0]

## Reproduction Steps
1. Step one
2. Step two
3. Expected vs actual behavior

## Additional Context
- Error messages
- Log output
- Code samples (if applicable)
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

1. **Pattern Analyzers**
   - New framework support (Vue.js, Svelte, Angular)
   - Additional language support (Python, Java, Go)
   - Enhanced pattern recognition
   - Performance optimizations

2. **Analysis Features**
   - Security pattern detection
   - Performance anti-pattern recognition
   - Accessibility analysis enhancement
   - Code complexity metrics

3. **Testing**
   - Integration tests
   - Performance benchmarks
   - Edge case coverage
   - Mock improvements

4. **Documentation**
   - API documentation
   - Usage examples
   - Best practices guides
   - Troubleshooting guides

5. **Developer Experience**
   - IDE integrations
   - CLI improvements
   - Error messages
   - Output formatting

6. **Infrastructure**
   - CI/CD improvements
   - Package distribution
   - Security enhancements
   - Configuration management

## 🏗️ Architecture Overview

### Core Components

```text
semantic_code_analyzer/
├── __init__.py              # Main package interface
├── cli.py                   # Command-line interface
├── analyzers/               # Specialized pattern analyzers
│   ├── __init__.py
│   ├── base_analyzer.py     # Abstract base class and data structures
│   ├── architectural_analyzer.py  # Next.js structure and imports
│   ├── quality_analyzer.py        # Security, performance, best practices
│   ├── typescript_analyzer.py     # Type safety and patterns
│   ├── framework_analyzer.py      # Framework-specific features
│   ├── domain_classifier.py       # Architectural domain classification
│   └── domain_adherence_analyzer.py  # Semantic pattern matching
├── embeddings/              # GraphCodeBERT embeddings and pattern indexing
│   ├── __init__.py
│   └── pattern_indexer.py   # FAISS pattern indexing and similarity search
├── hardware/                # Hardware acceleration management
│   ├── __init__.py
│   └── device_manager.py    # MPS/CUDA device management
├── scorers/                 # Scoring and aggregation
│   ├── __init__.py
│   ├── multi_dimensional_scorer.py  # Main orchestrator with domain weights
│   └── weighted_aggregator.py       # Mathematical aggregation
└── training/                # Fine-tuning support
    ├── __init__.py
    ├── data_preparation.py  # Training data preparation
    └── model_trainer.py     # GraphCodeBERT fine-tuning
```

### Key Design Principles

1. **Domain-Aware Semantic Analysis** - GraphCodeBERT embeddings with domain classification
2. **Pattern-Based Analysis** - Rule-based recognition of code quality patterns (optional)
3. **Modular Architecture** - Each analyzer has clear responsibilities
4. **Framework Awareness** - Specialized knowledge of React/Next.js/TypeScript
5. **Extensibility** - Easy to add new analyzers and patterns
6. **Performance** - Hardware acceleration (MPS/CUDA) with graceful CPU fallback
7. **Actionable Output** - Specific recommendations for improvement
8. **Fine-Tuning Support** - Custom model training for project-specific patterns

### Adding New Features

#### Adding a New Pattern Analyzer

1. **Create new analyzer** in `semantic_code_analyzer/analyzers/`
2. **Inherit from `BaseAnalyzer`** and implement required methods
3. **Define pattern recognition rules** using regex and AST analysis
4. **Add to analyzer imports** in `analyzers/__init__.py`
5. **Update `MultiDimensionalScorer`** to include new analyzer
6. **Add comprehensive tests** and documentation

Example:

```python
class VueAnalyzer(BaseAnalyzer):
    def analyze_file(self, file_path: str, content: str) -> AnalysisResult:
        patterns = []
        recommendations = []

        # Check for Vue 3 composition API
        if "setup()" in content:
            patterns.append(PatternMatch(...))

        # Check for proper reactive declarations
        if "ref(" in content and "reactive(" in content:
            patterns.append(PatternMatch(...))

        return AnalysisResult(...)
```

#### Adding Support for New Languages

1. **Update file extension checks** in `BaseAnalyzer._get_supported_extensions()`
2. **Add language-specific patterns** to relevant analyzers
3. **Create language-specific analyzers** if needed
4. **Update tests** with sample files
5. **Add examples** for the new language

#### Adding New Pattern Types

1. **Add enum value** to `PatternType` in `base_analyzer.py`
2. **Implement detection logic** in appropriate analyzer
3. **Add scoring logic** for the new pattern type
4. **Update tests** and documentation

## 🚨 Common Issues and Solutions

### Development Environment

**Issue**: Git repository not found

```bash
# Solution: Initialize git repo or run from git directory
git init
# or
cd /path/to/git/repo
```

**Issue**: Import errors during development

```bash
# Solution: Install in development mode
pip install -e .
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
# Solution: Add proper type annotations
mypy semantic_code_analyzer/ --show-error-codes
```

### Test Issues

**Issue**: Tests failing due to missing fixtures

```bash
# Solution: Use provided fixtures in conftest.py
def test_example(enhanced_scorer_config, sample_typescript_files):
    # Test implementation
```

## 📚 Resources

### Additional Documentation

- [README.md](README.md) - Project overview and usage
- [API Documentation](semantic_code_analyzer/) - Detailed API reference
- [Examples](examples/) - Usage examples and tutorials

### External Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Best Practices](https://react.dev/learn)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

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
3. **Create release tag**: `git tag v0.3.0`
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

### Short Term (v0.4.0)

- Enhanced pattern recognition for more frameworks (Vue.js, Svelte, Angular)
- Performance optimizations for large codebases
- Better error handling and user feedback
- IDE integrations

### Medium Term (v0.5.0)

- Additional language support (Python, Java, Go)
- Custom pattern definition system
- API server mode
- Real-time analysis
- Advanced visualization

### Long Term (v1.0.0)

- Enterprise features
- Team collaboration tools
- Custom rule definition UI
- Cloud deployment options
- Integration with code review tools

---

Thank you for contributing to Multi-Dimensional Code Analyzer! Your
contributions help make code quality analysis better for everyone. 🚀
