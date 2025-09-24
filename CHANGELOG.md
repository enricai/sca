# Changelog

All notable changes to the Semantic Code Analyzer project will be documented
in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-22

### New Features

- **NEW ANALYZER**: Domain-Aware Adherence Analysis using GraphCodeBERT
  embeddings
- **AI-POWERED PATTERN MATCHING**: FAISS similarity search for
  domain-specific patterns
- **ARCHITECTURAL DOMAIN CLASSIFICATION**: Automatic categorization into
  frontend, backend, testing, database domains
- **ENHANCED CLI**: New options for domain adherence configuration
  (`--domain-adherence-weight`, `--similarity-threshold`,
  `--max-similar-patterns`)
- **PATTERN INDEXING**: Automatic building of domain-specific pattern indices
- **CONFIDENCE SCORING**: Measures certainty of pattern matches and domain
  classification
- **CONTEXT-AWARE RECOMMENDATIONS**: Suggestions based on actual patterns
  from domain-specific code

### Breaking Changes

- **BREAKING**: Updated from 4-dimensional to 5-dimensional analysis
  (added domain adherence)
- **WEIGHTS**: Adjusted default weights - Architectural: 25%, Quality: 25%,
  TypeScript: 25%, Framework: 15%, Domain Adherence: 10%
- **API**: Enhanced `EnhancedScorerConfig` with domain adherence parameters
- **OUTPUT FORMAT**: Added domain adherence scores and domain classification
  breakdown
- **DEPENDENCIES**: Added transformers, torch, faiss-cpu, and numpy for
  AI features

### Bug Fixes

- **TYPE ANNOTATIONS**: Fixed missing type annotation for self parameter in
  mock functions
- **PRE-COMMIT CONFLICTS**: Resolved black/ruff-format conflicts by removing
  redundant ruff-format hook
- **IMPORT ERRORS**: Graceful handling when AI dependencies (transformers,
  faiss) are not available

## [0.1.2] - 2025-01-22

### Fixed

- **CRITICAL**: Fixed severely deflated similarity scores caused by improper
  per-file normalization
- **CRITICAL**: Added missing directory filtering to exclude `node_modules`
  in parent commit extraction
- **CRITICAL**: Fixed directory filtering in `_find_code_files()` method to
  exclude build artifacts
- Improved aggregation logic to use mean of max similarities instead of mean
  of mean similarities
- Fixed raw similarity preservation for better score interpretation

### Changed

- Disabled harmful per-file score normalization by default (`normalize_scores = False`)
- Improved similarity aggregation to provide more meaningful scores
- Enhanced CLI display to include 75th percentile similarity metric
- Better similarity score interpretation for commit analysis

### Improved

- Similarity scores now accurately reflect semantic relationship
  (e.g., 0.878 vs previous 0.476)
- Parent commit comparison now properly excludes dependency files and build
  artifacts
- More reliable and interpretable similarity analysis results
- Better performance when analyzing repositories with large dependency directories

## [0.1.1] - 2024-01-XX

### Added

- Full TypeScript support in CLI with `--language typescript` option
- TypeScript-specific code formatting for better embeddings
- Parent commit comparison as default behavior (compares against pre-commit state)
- `--compare-current` CLI option to use filesystem comparison (old behavior)
- Processing summary with performance metrics and hardware acceleration status
- Better progress indicators with specific model and commit information

### v0.1.1 Breaking Changes

- **BREAKING**: Default comparison behavior now uses parent commit instead of
  current filesystem
- Cleaner default output with WARNING level logging (use `-v` for verbose)
- Improved progress indicator descriptions with context-specific information

### v0.1.1 Bug Fixes

- SciPy import error: replaced `manhattan` with `cityblock` for Manhattan distance
- Eliminated `torch_dtype` deprecation warning (now uses `dtype`)
- Suppressed RobertaModel weight initialization warnings (cosmetic only)
- Eliminated Hugging Face tokenizers parallelism fork warnings
- CLI import error for `asdict` function (now imports from dataclasses)

### User Experience Improvements

- Much cleaner user experience with minimal warning output
- More meaningful semantic analysis through temporal comparison
- Professional output formatting with performance statistics
- Better TypeScript language support throughout the system

## [0.1.0] - 2024-01-XX

### Initial Features

- Initial release of Semantic Code Analyzer
- Core semantic similarity analysis for Git commits
- Apple M3 hardware acceleration with MPS support
- Multi-language support (Python, JavaScript, Java, C++, C, Go, etc.)
- Multiple distance metrics (Euclidean, Cosine, Manhattan, Chebyshev)
- Rich CLI interface with progress tracking
- Comprehensive test suite with 95%+ coverage
- Batch analysis for multiple commits
- Commit comparison functionality
- Intelligent embedding caching system
- Function-level analysis for Python code
- Cross-similarity matrix calculations
- Configurable analysis parameters
- JSON export capabilities
- Performance optimization features

### Features

#### Core Analysis Engine

- `CommitExtractor`: Git integration and code extraction
- `CodeEmbedder`: Semantic embedding generation using GraphCodeBERT
- `SimilarityCalculator`: Advanced distance metric calculations
- `SemanticScorer`: Main orchestration class

#### CLI Interface

- `sca-analyze`: Single commit analysis
- `sca-analyze batch`: Multi-commit analysis
- `sca-analyze compare`: Commit comparison
- `sca-analyze info`: Repository information

#### Apple M3 Optimizations

- MPS (Metal Performance Shaders) acceleration
- Mixed precision inference
- Optimized memory usage for 24GB RAM systems
- Batch processing optimizations

#### Multi-Language Support

- Python (`.py`)
- JavaScript/TypeScript (`.js`, `.ts`, `.jsx`, `.tsx`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)
- Go (`.go`)
- Rust (`.rs`)
- PHP (`.php`)
- Ruby (`.rb`)
- Swift (`.swift`)
- Kotlin (`.kt`)
- Scala (`.scala`)

#### Distance Metrics

- Euclidean distance (recommended, 24-66% better performance)
- Cosine similarity
- Manhattan distance
- Chebyshev distance
- Dot product similarity

#### Configuration Options

- Model selection (default: microsoft/graphcodebert-base)
- Distance metric selection
- File count limits for large repositories
- Caching controls
- Output format options
- Apple M3 MPS toggle
- Embedding normalization

### Dependencies

- Python 3.9+
- PyTorch 2.6.0+ (with MPS support and security fixes)
- Transformers 4.35.0+
- GitPython 3.1.40+
- NumPy 1.24.0+
- Scikit-learn 1.3.0+
- SciPy 1.11.0+
- Rich 13.0.0+ (CLI interface)
- Click 8.1.0+ (CLI framework)

### Performance

- Optimized for Apple M3 hardware
- Intelligent caching reduces repeated analysis time by 80%+
- Batch processing for efficient multi-commit analysis
- Memory optimization for large codebases
- Processing speeds:
  - Small repos (10 files): 2-5 seconds
  - Medium repos (100 files): 15-30 seconds
  - Large repos (1000 files): 2-5 minutes

### Documentation

- Comprehensive README with usage examples
- API documentation for all classes and methods
- CLI help system with detailed options
- Advanced usage examples
- Integration patterns for CI/CD
- Performance optimization guides

### Testing

- 95%+ test coverage
- Unit tests for all core components
- Integration tests for complete workflows
- Performance benchmarks
- Edge case handling
- Mock-based testing for ML models

### Known Limitations

- Requires internet connection for initial model download
- Large repositories may require significant memory (recommend 8GB+)
- Some programming languages have limited AST parsing support
- First-time model loading can take 1-2 minutes

### Compatibility

- macOS 12+ (optimized for Apple Silicon)
- Linux (x86_64, ARM64)
- Windows 10+ (limited testing)
- Python 3.9, 3.10, 3.11, 3.12

## [Unreleased]

### Planned Features

- Support for additional programming languages
- Custom model fine-tuning capabilities
- Integration with popular IDEs
- Real-time analysis during development
- Team collaboration features
- Advanced visualization dashboards
- API server mode for enterprise integration
- Database storage for analysis history
- Machine learning model improvements
- Performance optimizations for very large repositories

### Under Consideration

- Support for other code embedding models
- Integration with code review tools
- Automated code quality suggestions
- Repository health scoring
- Team coding pattern analysis
- Custom distance metric definitions
- Plugin system for extensibility

---

## Version History

- **0.1.0**: Initial release with core functionality
- **Planned 0.2.0**: Enhanced language support and IDE integration
- **Planned 0.3.0**: Enterprise features and API server
- **Planned 1.0.0**: Production-ready release with full feature set

## Migration Guide

### From Development to 0.1.0

This is the initial release, no migration needed.

### Future Migration Notes

- Breaking changes will be documented here
- Migration scripts will be provided for major version changes
- Backward compatibility will be maintained within major versions

## Support and Feedback

- Report bugs via GitHub Issues
- Request features via GitHub Discussions
- Contribute improvements via Pull Requests
- Ask questions in the community forums

---

**Note**: This changelog follows the principles of
[Keep a Changelog](https://keepachangelog.com/) and will be updated with each
release.
