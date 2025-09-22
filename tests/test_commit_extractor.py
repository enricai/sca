"""
Tests for the CommitExtractor module.
"""

import pytest
import tempfile
import os
from pathlib import Path
import git
from unittest.mock import Mock, patch

from semantic_code_analyzer.commit_extractor import CommitExtractor, CommitInfo


class TestCommitExtractor:
    """Test cases for CommitExtractor class."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary Git repository for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Initialize git repo
            repo = git.Repo.init(repo_path)

            # Configure git user (required for commits)
            repo.config_writer().set_value("user", "name", "Test User").release()
            repo.config_writer().set_value("user", "email", "test@example.com").release()

            # Create initial file
            test_file = repo_path / "test.py"
            test_file.write_text("def hello():\n    print('Hello, World!')\n")

            # Add and commit
            repo.index.add(["test.py"])
            initial_commit = repo.index.commit("Initial commit")

            # Create another file and commit
            test_file2 = repo_path / "test2.py"
            test_file2.write_text("def goodbye():\n    print('Goodbye!')\n")

            repo.index.add(["test2.py"])
            second_commit = repo.index.commit("Add test2.py")

            yield {
                'repo_path': str(repo_path),
                'repo': repo,
                'initial_commit': initial_commit.hexsha,
                'second_commit': second_commit.hexsha
            }

    @pytest.fixture
    def extractor(self, temp_repo):
        """Create CommitExtractor instance with temp repo."""
        return CommitExtractor(temp_repo['repo_path'])

    def test_init_valid_repo(self, temp_repo):
        """Test initialization with valid repository."""
        extractor = CommitExtractor(temp_repo['repo_path'])
        assert extractor.repo_path == Path(temp_repo['repo_path']).resolve()
        assert extractor.repo is not None

    def test_init_invalid_repo(self):
        """Test initialization with invalid repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory, not a git repo
            with pytest.raises(git.InvalidGitRepositoryError):
                CommitExtractor(tmpdir)

    def test_get_commit_info(self, extractor, temp_repo):
        """Test getting commit information."""
        commit_hash = temp_repo['second_commit']
        commit_info = extractor.get_commit_info(commit_hash)

        assert isinstance(commit_info, CommitInfo)
        assert commit_info.hash == commit_hash[:8]
        assert commit_info.message == "Add test2.py"
        assert commit_info.author == "Test User"
        assert len(commit_info.files_changed) == 1
        assert "test2.py" in commit_info.files_changed

    def test_get_commit_info_invalid_hash(self, extractor):
        """Test getting commit info with invalid hash."""
        with pytest.raises(git.BadName):
            extractor.get_commit_info("invalid_hash")

    def test_extract_commit_changes(self, extractor, temp_repo):
        """Test extracting commit changes."""
        commit_hash = temp_repo['second_commit']
        changes = extractor.extract_commit_changes(commit_hash)

        assert isinstance(changes, dict)
        assert "test2.py" in changes
        assert "def goodbye():" in changes["test2.py"]
        assert "print('Goodbye!')" in changes["test2.py"]

    def test_extract_commit_changes_initial_commit(self, extractor, temp_repo):
        """Test extracting changes from initial commit."""
        commit_hash = temp_repo['initial_commit']
        changes = extractor.extract_commit_changes(commit_hash)

        assert isinstance(changes, dict)
        assert "test.py" in changes
        assert "def hello():" in changes["test.py"]

    def test_get_existing_codebase(self, extractor, temp_repo):
        """Test getting existing codebase files."""
        codebase = extractor.get_existing_codebase()

        assert isinstance(codebase, dict)
        assert len(codebase) >= 1  # Should have at least one Python file

        # Check if our test files are included
        file_paths = list(codebase.keys())
        python_files = [f for f in file_paths if f.endswith('.py')]
        assert len(python_files) >= 1

    def test_get_existing_codebase_with_excludes(self, extractor):
        """Test getting codebase with excluded files."""
        exclude_files = ["test.py"]
        codebase = extractor.get_existing_codebase(exclude_files=exclude_files)

        assert isinstance(codebase, dict)
        assert "test.py" not in codebase

    def test_get_existing_codebase_max_files(self, extractor):
        """Test getting codebase with file limit."""
        codebase = extractor.get_existing_codebase(max_files=1)

        assert isinstance(codebase, dict)
        assert len(codebase) <= 1

    def test_is_supported_file(self, extractor):
        """Test file type checking."""
        assert extractor._is_supported_file("test.py")
        assert extractor._is_supported_file("test.js")
        assert extractor._is_supported_file("test.java")
        assert extractor._is_supported_file("test.cpp")

        # Unsupported files
        assert not extractor._is_supported_file("test.txt")
        assert not extractor._is_supported_file("test.pdf")
        assert not extractor._is_supported_file("README.md")

    def test_get_commit_list(self, extractor, temp_repo):
        """Test getting list of commits."""
        commits = extractor.get_commit_list(max_count=5)

        assert isinstance(commits, list)
        assert len(commits) == 2  # We created 2 commits

        # Check commit order (most recent first)
        assert commits[0].hash == temp_repo['second_commit'][:8]
        assert commits[1].hash == temp_repo['initial_commit'][:8]

    def test_validate_repository(self, extractor):
        """Test repository validation."""
        assert extractor.validate_repository() is True

    def test_validate_empty_repository(self):
        """Test validation with empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty git repo
            repo = git.Repo.init(tmpdir)
            extractor = CommitExtractor(tmpdir)

            assert extractor.validate_repository() is False

    def test_find_code_files(self, extractor, temp_repo):
        """Test finding code files in repository."""
        code_files = extractor._find_code_files()

        assert isinstance(code_files, list)
        assert len(code_files) >= 2  # Should find our test files

        # All files should be Path objects
        assert all(isinstance(f, Path) for f in code_files)

        # All files should be supported
        assert all(extractor._is_supported_file(str(f)) for f in code_files)


class TestCommitInfo:
    """Test cases for CommitInfo dataclass."""

    def test_commit_info_creation(self):
        """Test creating CommitInfo instance."""
        commit_info = CommitInfo(
            hash="abc123",
            message="Test commit",
            author="Test Author",
            timestamp="2023-01-01T00:00:00",
            files_changed=["test.py"],
            insertions=10,
            deletions=5
        )

        assert commit_info.hash == "abc123"
        assert commit_info.message == "Test commit"
        assert commit_info.author == "Test Author"
        assert commit_info.files_changed == ["test.py"]
        assert commit_info.insertions == 10
        assert commit_info.deletions == 5


# Performance and edge case tests
class TestCommitExtractorEdgeCases:
    """Test edge cases and error conditions."""

    def test_extract_changes_nonexistent_commit(self, temp_repo):
        """Test extracting changes from nonexistent commit."""
        extractor = CommitExtractor(temp_repo['repo_path'])

        # Should return empty dict for invalid commit
        changes = extractor.extract_commit_changes("nonexistent")
        assert changes == {}

    def test_large_file_handling(self, temp_repo):
        """Test handling of large files."""
        repo_path = Path(temp_repo['repo_path'])

        # Create a large file (over 1MB)
        large_file = repo_path / "large.py"
        large_content = "# Large file\n" + "x = 1\n" * 50000
        large_file.write_text(large_content)

        extractor = CommitExtractor(temp_repo['repo_path'])
        codebase = extractor.get_existing_codebase()

        # Large files should be skipped
        assert "large.py" not in codebase

    def test_binary_file_handling(self, temp_repo):
        """Test handling of binary files."""
        repo_path = Path(temp_repo['repo_path'])

        # Create a binary file
        binary_file = repo_path / "test.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        extractor = CommitExtractor(temp_repo['repo_path'])

        # Binary files should not be considered supported
        assert not extractor._is_supported_file("test.bin")

    def test_empty_file_handling(self, temp_repo):
        """Test handling of empty files."""
        repo_path = Path(temp_repo['repo_path'])

        # Create empty Python file
        empty_file = repo_path / "empty.py"
        empty_file.write_text("")

        extractor = CommitExtractor(temp_repo['repo_path'])
        codebase = extractor.get_existing_codebase()

        # Empty files should be skipped
        assert "empty.py" not in codebase

    @patch('semantic_code_analyzer.commit_extractor.logger')
    def test_logging_calls(self, mock_logger, temp_repo):
        """Test that appropriate logging calls are made."""
        extractor = CommitExtractor(temp_repo['repo_path'])

        # Should log initialization
        mock_logger.info.assert_called()

        # Reset mock
        mock_logger.reset_mock()

        # Extract changes should log
        extractor.extract_commit_changes(temp_repo['second_commit'])
        mock_logger.info.assert_called()

    def test_corrupted_file_handling(self, temp_repo):
        """Test handling of files that can't be read."""
        repo_path = Path(temp_repo['repo_path'])

        # Create a file with invalid UTF-8
        bad_file = repo_path / "bad.py"
        bad_file.write_bytes(b'\xff\xfe# Invalid UTF-8')

        extractor = CommitExtractor(temp_repo['repo_path'])

        # Should handle gracefully and not crash
        codebase = extractor.get_existing_codebase()
        assert isinstance(codebase, dict)  # Should still return a dict