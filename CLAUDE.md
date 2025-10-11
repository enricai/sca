# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build/Lint/Test Commands

- Setup environment: `conda create -n sca python=3.10 -y && conda activate sca`
  (Python 3.10+ required)
- Fix OpenMP conflicts and tokenizer parallelism warnings:

  ```bash
  conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
  conda deactivate
  conda activate sca
  ```

- Install dependencies: `pip install -e .`
- Install dev dependencies: `pip install -r requirements-dev.txt`
- Format code: `black .`
- Lint code: `ruff check .`
- Lint fix: `ruff check . --fix`
- Sort imports: `isort .`
- Type checking: `mypy semantic_code_analyzer`
- Run all tests: `pytest`
- Run tests with coverage: `pytest --cov=semantic_code_analyzer`
- Run tests in parallel: `pytest -n auto`
- Run single test: `pytest tests/test_file.py::test_function`
- Build package: `python -m build`
- CLI command (embeddings-only, default): `sca-analyze analyze HEAD`
- Verbose output: `sca-analyze analyze HEAD --verbose` or
  `sca-analyze analyze HEAD -v`
- Multi-dimensional mode (with regex analyzers): `sca-analyze analyze HEAD --enable-regex-analyzers`
- Custom similarity threshold: `sca-analyze analyze HEAD --similarity-threshold 0.4`
- Custom configuration: `sca-analyze analyze HEAD \
  --similarity-threshold 0.4 --max-similar-patterns 15`
- Compare against specific commit:
  `sca-analyze analyze HEAD --pattern-index-commit main`
- Compare against parent (default):
  `sca-analyze analyze HEAD --pattern-index-commit parent`
- Disable AI features:
  `sca-analyze analyze HEAD --disable-domain-adherence --disable-pattern-indices`
- Fine-tune model:
  `sca-analyze fine-tune HEAD --repo-path . --epochs 3 --batch-size 8`
- Fine-tune and push to HuggingFace Hub (public):
  `sca-analyze fine-tune HEAD --repo-path . --epochs 3 --batch-size 8 --push-to-hub --hub-model-id username/model-name`
- Fine-tune and push to HuggingFace Hub (private):
  `sca-analyze fine-tune HEAD --repo-path . --epochs 3 --batch-size 8 --push-to-hub --hub-model-id username/model-name --private`
- Use fine-tuned model (local): `sca-analyze analyze HEAD --fine-tuned-model abc123d`
- Use fine-tuned model (from Hub):
  `sca-analyze analyze HEAD --fine-tuned-model username/model-name`
- Specify hardware device: `sca-analyze analyze HEAD --device mps`
  (options: auto, cpu, mps, cuda)

## MANDATORY Requirements (MUST follow)

- ALWAYS use type hints for function parameters and return types
- ALWAYS use docstrings for modules, classes, and public functions (Google style)
- ALWAYS use logging instead of print statements
- ALWAYS use `logger = logging.getLogger(__name__)` for module-level logging
- ALWAYS use dataclasses for structured data instead of dictionaries
- ALWAYS use enums for constants and choices instead of string literals
- ALWAYS use Path objects from pathlib instead of string paths
- ALWAYS use f-strings for string formatting instead of .format() or %
- ALWAYS use explicit imports (no `from module import *`)
- ALWAYS follow PEP 8 naming conventions (snake_case for
  variables/functions, PascalCase for classes)
- ALWAYS use abstract base classes (ABC) for defining interfaces
- ALWAYS handle exceptions explicitly with try/except blocks
- ALWAYS use context managers (with statements) for file operations and resources
- ALWAYS run black, ruff, and mypy before completing tasks
- ALWAYS prefer immutable data structures when possible
- ALWAYS use list comprehensions and generator expressions for simple transformations
- ALWAYS use Optional[Type] for nullable parameters
- ALWAYS validate input parameters at function entry points
- ALWAYS use early returns to reduce nesting
- ALWAYS prefer composition over inheritance
- ALWAYS use constants for magic numbers and strings (UPPER_CASE naming)
- ALWAYS use rich library for CLI output formatting instead of plain print
- ALWAYS use click decorators for CLI command structure
- ALWAYS use pytest fixtures for test setup instead of setUp methods

## File Naming Convention

Follow Python module naming conventions with flat directory structure and
descriptive names.

### Examples

**Preferred:**

- `semantic_code_analyzer/analyzers/base_analyzer.py` instead of `semantic_code_analyzer/baseAnalyzer.py`
- `semantic_code_analyzer/scorers/weighted_aggregator.py` instead of `semantic_code_analyzer/weightedAggregator.py`
- `semantic_code_analyzer/cli.py` instead of `semantic_code_analyzer/command_line_interface.py`

**Pattern:**
Use snake_case for all Python module names. Group related functionality in
packages with descriptive directory names.

**Exception:** Test files should use `test_` prefix (e.g.,
`test_base_analyzer.py`, `test_cli.py`)

## Code Style Guidelines

- **Formatting:** Follow Black formatting (88 character line length). Use
  double quotes for strings.
- **Imports:** Organized by groups (standard library, third-party, local)
  with isort configuration
- **Types:** Python 3.10+ with strict type checking. Use Union sparingly,
  prefer Optional for nullable types.
- **Naming:** Use descriptive variable/function names with snake_case for
  variables/functions, PascalCase for classes
- **Error Handling:** Use specific exception types with proper logging and context
- **File Structure:** Follow established package patterns with proper
  `__init__.py` files
- **Tests:** Create pytest tests with descriptive names and proper fixtures
- **Documentation:** Use Google-style docstrings with proper Args, Returns,
  and Raises sections
- **Logic:** Avoid complex conditional chains. Use early returns and guard clauses.
- **Data Structures:** Prefer dataclasses for structured data, enums for
  choices, and typed collections

## Python-Specific Patterns

- **Dataclasses:** Use @dataclass decorator for data containers with proper
  type hints
- **Enums:** Use Enum classes for constants and choices instead of string literals
- **Abstract Base Classes:** Use ABC for defining interfaces and abstract methods
- **Context Managers:** Use `with` statements for resource management
  (files, connections, etc.)
- **Logging:** Use module-level loggers with proper formatting and levels
- **Path Handling:** Use pathlib.Path for all path operations instead of os.path
- **String Formatting:** Use f-strings for all string interpolation
- **Collections:** Use typing.List, typing.Dict, etc. for type hints
  (Python 3.10+ syntax with improved union types)
- **Error Handling:** Create custom exception classes inheriting from built-in exceptions
- **Configuration:** Use dataclasses or Pydantic models for configuration
  management

## CLI Framework Patterns

- Use Click for command-line interface with proper decorators and options
- Use Rich library for formatted terminal output (tables, panels, progress bars)
- Organize CLI commands in groups with @click.group() decorator
- Use @click.option() and @click.argument() for command parameters
- Implement proper error handling with sys.exit() and appropriate exit codes
- Use Rich Console for all output formatting instead of plain print statements
- Validate CLI inputs at the command level before passing to business logic
- Use Click's context passing for shared configuration between commands

## Testing Standards

- Test files MUST use `test_` prefix and be placed in `tests/` directory
- Use pytest fixtures for test setup and teardown
- Mock external dependencies using pytest-mock or unittest.mock
- Use descriptive test function names that explain what is being tested
- Organize tests with `class Test*` for grouping related tests
- Use parametrize decorator for testing multiple scenarios
- Assert with descriptive messages using pytest's assert statements
- Use pytest markers for categorizing tests (unit, integration, slow)
- Maintain high test coverage (aim for >90%) with meaningful tests

## Validation & Error Handling Patterns

- Input validation MUST happen at function entry points
- Use specific exception types (ValueError, TypeError, FileNotFoundError, etc.)
- Create custom exception classes for domain-specific errors
- Always include context in exception messages
- Use logging for error tracking with appropriate levels (ERROR, WARNING, INFO)
- Validate file paths exist before processing
- Handle CLI argument validation in Click command functions
- Use typing for static validation and runtime validation for dynamic inputs

## Code Examples

✅ Good: `logger.info("processing commit %s", commit_hash)`
❌ Bad: `print(f"Processing commit {commit_hash}")`

✅ Good: `raise ValueError("Invalid commit hash provided: commit hash cannot be empty")`
❌ Bad: `raise Exception("invalid commit hash")`

✅ Good: `from pathlib import Path` then `Path("src/analyzers")`
❌ Bad: `import os` then `os.path.join("src", "analyzers")`

✅ Good: `def analyze_commit(self, commit_hash: str) -> AnalysisResult:`
❌ Bad: `def analyze_commit(self, commit_hash):`

✅ Good: `logger = logging.getLogger(__name__)`
❌ Bad: `import logging` then `logging.info(...)`

✅ Good: `if not commit_hash: return None`
❌ Bad: `if not commit_hash: { return None }`

✅ Good: `results = [item.score for item in items if item.is_valid]`
❌ Bad: `results = []; for item in items: if item.is_valid: results.append(item.score)`

✅ Good: `from typing import Optional` then `def get_score(self) -> Optional[float]:`
❌ Bad: `def get_score(self):` returning None sometimes

✅ Good: `with open(file_path) as f: content = f.read()`
❌ Bad: `f = open(file_path); content = f.read(); f.close()`

✅ Good:

```python
@dataclass
class AnalysisResult:
    overall_adherence: float
    confidence: float
    dimensional_scores: Dict[str, float]
```

❌ Bad: `result = {"overall_adherence": 0.8, "confidence": 0.9,
"dimensional_scores": {}}`

✅ Good:

```python
class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

❌ Bad: `severity = "warning"  # magic string`

✅ Good:

```python
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, content: str) -> AnalysisResult:
        """Analyze code content and return results."""
        pass
```

❌ Bad: Base class without ABC that expects subclasses to implement methods

✅ Good: Early return with validation

```python
def process_commit(commit_hash: str) -> Optional[AnalysisResult]:
    if not commit_hash:
        logger.warning("Empty commit hash provided")
        return None

    if not self._is_valid_commit(commit_hash):
        logger.error("Invalid commit hash: %s", commit_hash)
        return None

    return self._analyze_commit(commit_hash)
```

❌ Bad: Nested conditionals

```python
def process_commit(commit_hash: str) -> Optional[AnalysisResult]:
    if commit_hash:
        if self._is_valid_commit(commit_hash):
            return self._analyze_commit(commit_hash)
        else:
            logger.error("Invalid commit hash: %s", commit_hash)
            return None
    else:
        logger.warning("Empty commit hash provided")
        return None
```

✅ Good: Rich CLI output

```python
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Analysis Results")
table.add_column("Dimension", style="cyan")
table.add_column("Score", style="magenta")
console.print(table)
```

❌ Bad: `print("Analysis Results\nDimension | Score")`

✅ Good: Click command structure

```python
@click.command()
@click.argument('commit_hash')
@click.option('--repo-path', '-r', default='.', help='Repository path')
@click.option('--output', '-o', type=click.File('w'), help='Output file')
def analyze(commit_hash: str, repo_path: str, output: Optional[TextIO]) -> None:
    """Analyze a specific commit for code quality patterns."""
    pass
```

❌ Bad: Manual argument parsing with sys.argv

✅ Good: Proper exception handling

```python
try:
    result = self._analyze_file(file_path)
except FileNotFoundError:
    logger.error("File not found: %s", file_path)
    return None
except PermissionError:
    logger.error("Permission denied accessing file: %s", file_path)
    return None
except Exception as e:
    logger.exception("Unexpected error analyzing file %s: %s", file_path, e)
    return None
```

❌ Bad: `except Exception: pass` or bare `except:`

## Package Structure Patterns

- Use proper `__init__.py` files with explicit exports using `__all__`
- Organize modules by functionality (analyzers, scorers, cli, etc.)
- Keep the main package interface simple and well-documented
- Use relative imports within packages: `from .base_analyzer import BaseAnalyzer`
- Export main interfaces at package level for easy importing
- Follow semantic versioning for package releases

## Configuration Management

- Use dataclasses for configuration objects with sensible defaults
- Support both programmatic and file-based configuration
- Validate configuration parameters at initialization
- Use environment variables for sensitive or deployment-specific settings
- Document all configuration options with type hints and docstrings
- Use enums for configuration choices to prevent invalid values

## Development Workflow

- Branch naming: Use descriptive names with appropriate prefixes
  (feature/, fix/, docs/)
- Commits MUST follow conventional commit format
- Pre-commit hooks: Use pre-commit framework with black, ruff, mypy, and tests
- Code review: Ensure all code passes linting, type checking, and tests
- Documentation: Update docstrings and README for public API changes
- Testing: Add tests for new functionality before merging
- Version management: Update version in pyproject.toml for releases

## Performance Considerations

- Use generators for large data processing instead of loading everything into memory
- Profile code with cProfile or line_profiler for performance bottlenecks
- Use appropriate data structures (sets for membership testing, deques for queues)
- Cache expensive computations using functools.lru_cache or similar
- Use numpy for numerical computations when dealing with large datasets
- Consider using async/await for I/O-bound operations

## Security Best Practices

- Validate all external inputs (file paths, user data, command arguments)
- Use pathlib.Path.resolve() to prevent path traversal attacks
- Sanitize data before logging to prevent log injection
- Use subprocess with shell=False when calling external commands
- Handle sensitive data appropriately (don't log secrets or personal information)
- Keep dependencies updated and scan for vulnerabilities

## Task Completion Checklist

Before marking any task complete, ALWAYS:

- Run `black .` to format code
- Run `ruff check . --fix` to fix linting issues
- Run `mypy semantic_code_analyzer` and ensure no type errors
- Run `pytest` and ensure all tests pass
- Run `pytest --cov=semantic_code_analyzer` to check coverage
- Update docstrings for any new public functions/classes
- Add tests for new functionality
