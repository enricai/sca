"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import git
import numpy as np
from unittest.mock import Mock, patch
import torch

# Set random seeds for reproducible tests
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture(scope="session")
def test_repo():
    """
    Create a test Git repository for the entire test session.
    This is a session-scoped fixture to avoid recreating the repo for each test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize git repo
        repo = git.Repo.init(repo_path)

        # Configure git user
        repo.config_writer().set_value("user", "name", "Test User").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create sample Python files
        files_to_create = {
            "main.py": '''
def main():
    """Main function of the application."""
    print("Hello, World!")
    calculate_fibonacci(10)

def calculate_fibonacci(n):
    """Calculate fibonacci numbers."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

if __name__ == "__main__":
    main()
''',
            "utils.py": '''
import os
import sys
from typing import List, Dict, Optional

def read_file(filepath: str) -> str:
    """Read content from a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def write_file(filepath: str, content: str) -> bool:
    """Write content to a file."""
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    except Exception:
        return False

class DataProcessor:
    """Process data in various formats."""

    def __init__(self, config: Dict):
        self.config = config

    def process_list(self, data: List) -> List:
        """Process a list of data."""
        return [item * 2 for item in data if isinstance(item, (int, float))]

    def process_dict(self, data: Dict) -> Dict:
        """Process a dictionary of data."""
        return {k: v for k, v in data.items() if v is not None}
''',
            "constants.py": '''
"""Application constants."""

VERSION = "1.0.0"
DEBUG = True
MAX_RETRIES = 3
TIMEOUT = 30

DEFAULT_CONFIG = {
    "host": "localhost",
    "port": 8080,
    "ssl": False,
    "max_connections": 100
}

ERROR_MESSAGES = {
    "connection_failed": "Failed to connect to server",
    "timeout": "Operation timed out",
    "invalid_input": "Invalid input provided"
}
''',
            "models/user.py": '''
"""User model definition."""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    """User model."""
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    last_login: Optional[datetime] = None

    def __post_init__(self):
        if not self.username:
            raise ValueError("Username cannot be empty")
        if "@" not in self.email:
            raise ValueError("Invalid email format")

    def activate(self):
        """Activate the user account."""
        self.is_active = True

    def deactivate(self):
        """Deactivate the user account."""
        self.is_active = False

    def update_last_login(self):
        """Update the last login timestamp."""
        self.last_login = datetime.now()
''',
            "tests/test_main.py": '''
"""Tests for main module."""

import pytest
from main import main, calculate_fibonacci

def test_fibonacci():
    """Test fibonacci calculation."""
    assert calculate_fibonacci(0) == 0
    assert calculate_fibonacci(1) == 1
    assert calculate_fibonacci(5) == 5
    assert calculate_fibonacci(10) == 55

def test_fibonacci_negative():
    """Test fibonacci with negative input."""
    assert calculate_fibonacci(-1) == -1
'''
        }

        # Create directory structure
        (repo_path / "models").mkdir()
        (repo_path / "tests").mkdir()

        # Create and commit files
        commits = []

        # Initial commit with main.py
        (repo_path / "main.py").write_text(files_to_create["main.py"])
        repo.index.add(["main.py"])
        initial_commit = repo.index.commit("Initial commit - Add main.py")
        commits.append(initial_commit.hexsha)

        # Second commit with utils
        (repo_path / "utils.py").write_text(files_to_create["utils.py"])
        repo.index.add(["utils.py"])
        utils_commit = repo.index.commit("Add utils module")
        commits.append(utils_commit.hexsha)

        # Third commit with constants
        (repo_path / "constants.py").write_text(files_to_create["constants.py"])
        repo.index.add(["constants.py"])
        constants_commit = repo.index.commit("Add constants module")
        commits.append(constants_commit.hexsha)

        # Fourth commit with models
        (repo_path / "models" / "user.py").write_text(files_to_create["models/user.py"])
        repo.index.add(["models/user.py"])
        models_commit = repo.index.commit("Add user model")
        commits.append(models_commit.hexsha)

        # Fifth commit with tests
        (repo_path / "tests" / "test_main.py").write_text(files_to_create["tests/test_main.py"])
        repo.index.add(["tests/test_main.py"])
        tests_commit = repo.index.commit("Add tests for main module")
        commits.append(tests_commit.hexsha)

        yield {
            'repo_path': str(repo_path),
            'repo': repo,
            'commits': commits,
            'files': files_to_create
        }


@pytest.fixture
def sample_embeddings():
    """Create reproducible sample embeddings for testing."""
    np.random.seed(42)

    return {
        'target': np.random.rand(768).astype(np.float32),
        'reference1': np.random.rand(768).astype(np.float32),
        'reference2': np.random.rand(768).astype(np.float32),
        'similar': np.random.rand(768).astype(np.float32),
        'different': np.random.rand(768).astype(np.float32),
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing without loading actual models."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Mock tokenizer behavior
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }

    # Mock model output
    mock_output = Mock()
    mock_output.last_hidden_state = torch.randn(1, 5, 768)
    mock_model.return_value = mock_output

    return {
        'model': mock_model,
        'tokenizer': mock_tokenizer
    }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for caching tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="function")
def disable_mps():
    """Disable MPS for tests that need CPU-only execution."""
    with patch('torch.backends.mps.is_available', return_value=False):
        yield


@pytest.fixture(scope="function")
def enable_mps():
    """Enable MPS for tests that need to test MPS functionality."""
    with patch('torch.backends.mps.is_available', return_value=True):
        yield


@pytest.fixture
def sample_code_snippets():
    """Provide sample code snippets for testing."""
    return {
        'python_function': '''
def calculate_area(radius):
    """Calculate the area of a circle."""
    import math
    return math.pi * radius ** 2
''',
        'python_class': '''
class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
''',
        'javascript_function': '''
function calculateArea(radius) {
    // Calculate the area of a circle
    return Math.PI * radius * radius;
}

function formatNumber(num, decimals = 2) {
    return num.toFixed(decimals);
}
''',
        'java_class': '''
public class Calculator {
    private List<String> history;

    public Calculator() {
        this.history = new ArrayList<>();
    }

    public int add(int a, int b) {
        int result = a + b;
        history.add(a + " + " + b + " = " + result);
        return result;
    }

    public int multiply(int a, int b) {
        int result = a * b;
        history.add(a + " * " + b + " = " + result);
        return result;
    }
}
'''
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that use actual models as slow
        if "embedder" in item.name.lower() and "mock" not in item.name.lower():
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name.lower() or "test_semantic_scorer" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# Utility functions for tests
def create_temp_git_repo(files_dict: dict) -> str:
    """
    Create a temporary git repository with specified files.

    Args:
        files_dict: Dictionary mapping file paths to content

    Returns:
        Path to the temporary repository
    """
    tmpdir = tempfile.mkdtemp()
    repo_path = Path(tmpdir)

    # Initialize git repo
    repo = git.Repo.init(repo_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create files
    for filepath, content in files_dict.items():
        full_path = repo_path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    # Add and commit
    repo.index.add(list(files_dict.keys()))
    commit = repo.index.commit("Test commit")

    return str(repo_path), commit.hexsha


def cleanup_temp_repo(repo_path: str):
    """Clean up a temporary repository."""
    if Path(repo_path).exists():
        shutil.rmtree(repo_path)