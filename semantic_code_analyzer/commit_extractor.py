"""
Git commit extraction and codebase analysis module.

This module provides functionality to extract code changes from Git commits
and analyze existing codebase files for semantic similarity comparison.
"""

import git
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Information about a commit and its changes."""
    hash: str
    message: str
    author: str
    timestamp: str
    files_changed: List[str]
    insertions: int
    deletions: int


class CommitExtractor:
    """Extracts code changes from Git commits and existing codebase files."""

    SUPPORTED_EXTENSIONS = {
        '.py',    # Python
        '.js',    # JavaScript
        '.ts',    # TypeScript
        '.jsx',   # React JSX
        '.tsx',   # React TSX
        '.java',  # Java
        '.cpp',   # C++
        '.c',     # C
        '.h',     # C/C++ headers
        '.hpp',   # C++ headers
        '.cs',    # C#
        '.go',    # Go
        '.rs',    # Rust
        '.php',   # PHP
        '.rb',    # Ruby
        '.swift', # Swift
        '.kt',    # Kotlin
        '.scala', # Scala
        '.r',     # R
        '.m',     # Objective-C/MATLAB
        '.sql',   # SQL
        '.sh',    # Shell scripts
        '.ps1',   # PowerShell
    }

    def __init__(self, repo_path: str):
        """
        Initialize the CommitExtractor.

        Args:
            repo_path: Path to the Git repository

        Raises:
            git.InvalidGitRepositoryError: If the path is not a valid Git repository
        """
        self.repo_path = Path(repo_path).resolve()

        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError as e:
            logger.error(f"Invalid Git repository at {repo_path}: {e}")
            raise

        logger.info(f"Initialized CommitExtractor for repository: {self.repo_path}")

    def get_commit_info(self, commit_hash: str) -> CommitInfo:
        """
        Get basic information about a commit.

        Args:
            commit_hash: Hash of the commit to analyze

        Returns:
            CommitInfo object with commit details

        Raises:
            git.BadName: If the commit hash is invalid
        """
        try:
            commit = self.repo.commit(commit_hash)
        except git.BadName as e:
            logger.error(f"Invalid commit hash {commit_hash}: {e}")
            raise

        # Get file statistics
        stats = commit.stats
        files_changed = list(stats.files.keys())

        return CommitInfo(
            hash=commit.hexsha[:8],
            message=commit.message.strip(),
            author=commit.author.name,
            timestamp=commit.committed_datetime.isoformat(),
            files_changed=files_changed,
            insertions=stats.total['insertions'],
            deletions=stats.total['deletions']
        )

    def extract_commit_changes(self, commit_hash: str) -> Dict[str, str]:
        """
        Extract code changes from a specific commit.

        Args:
            commit_hash: Hash of the commit to analyze

        Returns:
            Dictionary mapping file paths to their new content after the commit
        """
        logger.info(f"Extracting changes from commit {commit_hash}")

        try:
            commit = self.repo.commit(commit_hash)
        except git.BadName as e:
            logger.error(f"Invalid commit hash {commit_hash}: {e}")
            return {}

        changes = {}

        # Get the diff for this commit
        if len(commit.parents) > 0:
            # Regular commit with parent(s)
            diff = commit.parents[0].diff(commit)
        else:
            # First commit in repository
            diff = commit.diff(git.NULL_TREE)

        for diff_item in diff:
            file_path = diff_item.a_path or diff_item.b_path

            if not file_path:
                continue

            # Check if file is a supported code file
            if not self._is_supported_file(file_path):
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue

            # Skip deleted files
            if diff_item.deleted_file:
                logger.debug(f"Skipping deleted file: {file_path}")
                continue

            # Get the new content
            try:
                if diff_item.b_blob:
                    content = diff_item.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                    changes[file_path] = content
                    logger.debug(f"Extracted content from {file_path}: {len(content)} characters")
            except Exception as e:
                logger.warning(f"Failed to extract content from {file_path}: {e}")
                continue

        logger.info(f"Extracted changes from {len(changes)} files")
        return changes

    def get_existing_codebase(self,
                             exclude_files: Optional[List[str]] = None,
                             max_files: Optional[int] = None) -> Dict[str, str]:
        """
        Extract existing codebase files for comparison.

        Args:
            exclude_files: List of file paths to exclude from analysis
            max_files: Maximum number of files to process (for large codebases)

        Returns:
            Dictionary mapping file paths to their content
        """
        logger.info("Extracting existing codebase files")

        exclude_files = exclude_files or []
        exclude_set = set(exclude_files)
        codebase = {}
        files_processed = 0

        # Common directories to exclude
        exclude_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            '.pytest_cache', '.mypy_cache', 'dist', 'build', '.next',
            'coverage', '.coverage', 'target', 'bin', 'obj'
        }

        for file_path in self._find_code_files():
            # Skip if we've hit the max files limit
            if max_files and files_processed >= max_files:
                logger.info(f"Reached maximum file limit ({max_files})")
                break

            relative_path = file_path.relative_to(self.repo_path)
            str_path = str(relative_path)

            # Skip excluded files
            if str_path in exclude_set:
                logger.debug(f"Skipping excluded file: {str_path}")
                continue

            # Skip files in excluded directories
            if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                logger.debug(f"Skipping file in excluded directory: {str_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Skip empty files and very large files (>1MB)
                if not content.strip() or len(content) > 1_000_000:
                    logger.debug(f"Skipping empty or very large file: {str_path}")
                    continue

                codebase[str_path] = content
                files_processed += 1
                logger.debug(f"Loaded {str_path}: {len(content)} characters")

            except Exception as e:
                logger.warning(f"Failed to read file {str_path}: {e}")
                continue

        logger.info(f"Loaded {len(codebase)} files from existing codebase")
        return codebase

    def get_codebase_at_parent_commit(self,
                                    commit_hash: str,
                                    max_files: Optional[int] = None) -> Dict[str, str]:
        """
        Extract codebase files from the parent commit state (before target commit).

        Args:
            commit_hash: Hash of the target commit
            max_files: Maximum number of files to process (for large codebases)

        Returns:
            Dictionary mapping file paths to their content at parent commit state
        """
        logger.info(f"Extracting codebase at parent of commit {commit_hash}")

        try:
            commit = self.repo.commit(commit_hash)
        except git.BadName as e:
            logger.error(f"Invalid commit hash {commit_hash}: {e}")
            return {}

        # Handle edge cases
        if len(commit.parents) == 0:
            logger.info("First commit - no parent to compare against")
            return {}

        # Get parent commit
        parent_commit = commit.parents[0]  # Use first parent for merge commits
        logger.info(f"Using parent commit: {parent_commit.hexsha[:8]}")

        codebase = {}
        files_processed = 0

        try:
            # Traverse the parent commit tree
            for item in parent_commit.tree.traverse():
                # Skip if we've hit the max files limit
                if max_files and files_processed >= max_files:
                    logger.info(f"Reached maximum file limit ({max_files})")
                    break

                # Only process blob objects (files, not directories)
                if item.type != 'blob':
                    continue

                # Check if file is supported
                if not self._is_supported_file(item.path):
                    logger.debug(f"Skipping unsupported file: {item.path}")
                    continue

                try:
                    # Extract file content from parent commit
                    content = item.data_stream.read().decode('utf-8', errors='ignore')

                    # Skip empty files and very large files (>1MB)
                    if not content.strip() or len(content) > 1_000_000:
                        logger.debug(f"Skipping empty or very large file: {item.path}")
                        continue

                    codebase[item.path] = content
                    files_processed += 1
                    logger.debug(f"Loaded {item.path}: {len(content)} characters")

                except Exception as e:
                    logger.warning(f"Failed to read file {item.path} from parent commit: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to traverse parent commit tree: {e}")
            return {}

        logger.info(f"Loaded {len(codebase)} files from parent commit state")
        return codebase

    def _find_code_files(self) -> List[Path]:
        """
        Find all supported code files in the repository.

        Returns:
            List of Path objects for supported code files
        """
        code_files = []

        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and self._is_supported_file(str(file_path)):
                code_files.append(file_path)

        # Sort by modification time (most recent first) for better caching
        code_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return code_files

    def _is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is a supported code file.

        Args:
            file_path: Path to the file

        Returns:
            True if the file is supported, False otherwise
        """
        path_obj = Path(file_path)

        # Check extension
        if path_obj.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False

        # Additional checks for specific file types
        filename = path_obj.name.lower()

        # Skip certain common non-code files
        skip_patterns = {
            'package-lock.json', 'yarn.lock', 'poetry.lock',
            'requirements.txt', 'setup.py', 'setup.cfg',
            'makefile', 'dockerfile', 'docker-compose.yml'
        }

        if filename in skip_patterns:
            return False

        # Skip test files if they're not the main focus
        # (can be enabled if needed for analysis)
        if any(pattern in filename for pattern in ['test_', '_test.', '.test.']):
            return True  # Include tests for now

        return True

    def get_commit_list(self,
                       branch: str = 'HEAD',
                       max_count: int = 100) -> List[CommitInfo]:
        """
        Get a list of recent commits from the repository.

        Args:
            branch: Branch to get commits from
            max_count: Maximum number of commits to return

        Returns:
            List of CommitInfo objects
        """
        logger.info(f"Getting commit list from {branch} (max: {max_count})")

        commits = []

        try:
            for commit in self.repo.iter_commits(branch, max_count=max_count):
                commit_info = self.get_commit_info(commit.hexsha)
                commits.append(commit_info)
        except Exception as e:
            logger.error(f"Failed to get commit list: {e}")
            return []

        logger.info(f"Retrieved {len(commits)} commits")
        return commits

    def validate_repository(self) -> bool:
        """
        Validate that the repository is in a good state for analysis.

        Returns:
            True if repository is valid, False otherwise
        """
        try:
            # Check if repo has commits
            if not list(self.repo.iter_commits(max_count=1)):
                logger.error("Repository has no commits")
                return False

            # Check if working directory is clean (warn but don't fail)
            if self.repo.is_dirty():
                logger.warning("Repository has uncommitted changes")

            # Check if we can access basic repo info
            _ = self.repo.head.commit

            logger.info("Repository validation successful")
            return True

        except Exception as e:
            logger.error(f"Repository validation failed: {e}")
            return False