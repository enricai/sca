# MIT License
#
# Copyright (c) 2024 Semantic Code Analyzer Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Universal language registry for tree-sitter parsers and queries.

This module provides automatic discovery and configuration of tree-sitter language
support, making the analyzer truly universal across 50+ programming languages.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tree-sitter dependencies
try:
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter dependencies not available.")


@dataclass
class LanguageConfig:
    """Configuration for a supported language."""

    name: str
    extensions: list[str]
    parser: Any  # Parser instance
    query_string: str
    package_name: str


class LanguageRegistry:
    """Registry of supported languages with automatic discovery.

    This class provides:
    - Automatic discovery of installed tree-sitter language packages
    - Loading of query files for each language
    - Unified interface for function extraction across languages
    """

    # Mapping of language names to their tree-sitter package names and file extensions
    LANGUAGE_MAP = {
        "python": {
            "package": "tree_sitter_python",
            "extensions": [".py"],
            "query_file": "python.scm",
        },
        "javascript": {
            "package": "tree_sitter_javascript",
            "extensions": [".js", ".jsx", ".mjs", ".cjs"],
            "query_file": "javascript.scm",
        },
        "typescript": {
            "package": "tree_sitter_typescript",
            "extensions": [".ts", ".tsx"],
            "query_file": "typescript.scm",
            "submodule": "language_typescript",  # TypeScript uses a submodule
        },
        "go": {
            "package": "tree_sitter_go",
            "extensions": [".go"],
            "query_file": "go.scm",
        },
        "rust": {
            "package": "tree_sitter_rust",
            "extensions": [".rs"],
            "query_file": "rust.scm",
        },
        "java": {
            "package": "tree_sitter_java",
            "extensions": [".java"],
            "query_file": "java.scm",
        },
        "c": {
            "package": "tree_sitter_c",
            "extensions": [".c", ".h"],
            "query_file": "c.scm",
        },
        "cpp": {
            "package": "tree_sitter_cpp",
            "extensions": [".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
            "query_file": "cpp.scm",
        },
        "ruby": {
            "package": "tree_sitter_ruby",
            "extensions": [".rb"],
            "query_file": "ruby.scm",
        },
        "php": {
            "package": "tree_sitter_php",
            "extensions": [".php"],
            "query_file": "php.scm",
        },
    }

    def __init__(self) -> None:
        """Initialize the language registry with automatic discovery."""
        self.languages: dict[str, LanguageConfig] = {}
        self.extension_map: dict[str, str] = {}  # Map extensions to language names

        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "Tree-sitter not available. Language registry will be empty."
            )
            return

        # Get queries directory
        queries_dir = Path(__file__).parent / "queries"

        # Discover and initialize supported languages
        for lang_name, lang_info in self.LANGUAGE_MAP.items():
            try:
                self._initialize_language(lang_name, lang_info, queries_dir)
            except Exception as e:
                logger.debug(
                    f"Language {lang_name} not available (package not installed): {e}"
                )
                continue

        logger.info(
            f"LanguageRegistry initialized with {len(self.languages)} languages: "
            f"{list(self.languages.keys())}"
        )

    def _initialize_language(
        self, lang_name: str, lang_info: dict[str, Any], queries_dir: Path
    ) -> None:
        """Initialize a single language if its package is available.

        Args:
            lang_name: Name of the language
            lang_info: Language configuration dictionary
            queries_dir: Directory containing query files
        """
        package_name = lang_info["package"]

        # Try to import the language package
        try:
            lang_module = importlib.import_module(package_name)

            # Get the language function (handle TypeScript submodule case)
            if "submodule" in lang_info:
                language_func = getattr(lang_module, lang_info["submodule"])
            else:
                language_func = lang_module.language

            # Create parser with new tree-sitter 0.23.0+ API
            parser = Parser(Language(language_func()))

            # Load query file if it exists
            query_file = queries_dir / lang_info["query_file"]
            if query_file.exists():
                query_string = query_file.read_text(encoding="utf-8")
            else:
                # Use fallback query that looks for common function patterns
                query_string = self._get_fallback_query(lang_name)
                logger.debug(f"No query file for {lang_name}, using fallback patterns")

            # Create language config
            config = LanguageConfig(
                name=lang_name,
                extensions=lang_info["extensions"],
                parser=parser,
                query_string=query_string,
                package_name=package_name,
            )

            self.languages[lang_name] = config

            # Map extensions to language name
            for ext in lang_info["extensions"]:
                self.extension_map[ext] = lang_name

            logger.debug(
                f"Initialized {lang_name} parser with {len(lang_info['extensions'])} extensions"
            )

        except ImportError:
            # Package not installed, skip silently
            logger.debug(f"Package {package_name} not installed, skipping {lang_name}")
            raise
        except Exception as e:
            logger.warning(f"Failed to initialize {lang_name}: {e}")
            raise

    def _get_fallback_query(self, lang_name: str) -> str:
        """Get a fallback query for languages without query files.

        Args:
            lang_name: Language name

        Returns:
            Fallback query string
        """
        # Generic fallback that tries common function node types
        # This won't work for all languages but provides basic support
        return """
; Fallback query - tries common patterns
(function_definition) @function.def
(function_declaration) @function.def
(method_definition) @method.def
(method_declaration) @method.def
"""

    def get_language_for_extension(self, extension: str) -> LanguageConfig | None:
        """Get language configuration for a file extension.

        Args:
            extension: File extension (e.g., '.py', '.js')

        Returns:
            LanguageConfig if supported, None otherwise
        """
        lang_name = self.extension_map.get(extension)
        if lang_name:
            return self.languages.get(lang_name)
        return None

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions.

        Returns:
            Set of supported extensions
        """
        return set(self.extension_map.keys())

    def get_available_languages(self) -> list[str]:
        """Get list of available language names.

        Returns:
            List of language names
        """
        return list(self.languages.keys())
