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

"""Function-level code extraction using Tree-sitter for universal AST parsing.

This module provides functionality to extract functions from source code files
across multiple programming languages using tree-sitter query-based parsing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_compressor import DataCompressionConfig, DataCompressor
from .language_registry import LanguageRegistry

logger = logging.getLogger(__name__)

# Try to import tree-sitter dependencies
try:
    from tree_sitter import Query, QueryCursor  # type: ignore[attr-defined]

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "Tree-sitter dependencies not available. Function extraction will be limited."
    )


@dataclass
class FunctionChunk:
    """Represents a function chunk with its metadata.

    The chunk includes both the function code and its import context.
    Use the `code` property to get the complete code (imports + function),
    or access `function_code` directly for just the function body.
    """

    function_code: str  # Just the function, no imports
    function_name: str
    file_path: str
    is_method: bool
    imports: str
    start_line: int  # Line number where function starts in original file
    end_line: int  # Line number where function ends in original file

    @property
    def code(self) -> str:
        """Get complete code with imports prepended.

        Returns:
            String containing imports (if any) followed by function code
        """
        if self.imports:
            return f"{self.imports}\n\n{self.function_code}"
        return self.function_code


class FunctionExtractor:
    """Extract functions from source code using tree-sitter query-based parsing.

    This class provides universal function extraction across 10+ programming
    languages using declarative tree-sitter queries, making it extensible to
    any language with a tree-sitter grammar.
    """

    def __init__(self, compression_config: DataCompressionConfig | None = None) -> None:
        """Initialize the FunctionExtractor with language registry.

        Args:
            compression_config: Configuration for data compression (uses defaults if None)
        """
        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "FunctionExtractor initialized without tree-sitter support. "
                "Function extraction will fall back to simple chunking."
            )
            self.language_registry: LanguageRegistry | None = None
            self.compression_config = compression_config or DataCompressionConfig()
            return

        # Initialize language registry for automatic parser discovery
        try:
            self.language_registry = LanguageRegistry()
            supported_langs = self.language_registry.get_available_languages()
            supported_exts = self.language_registry.get_supported_extensions()

            logger.info(
                f"FunctionExtractor initialized with {len(supported_langs)} languages: {supported_langs}"
            )
            logger.debug(f"Supported extensions: {sorted(supported_exts)}")

        except Exception as e:
            logger.error(f"Failed to initialize LanguageRegistry: {e}")
            self.language_registry = None

        # Store compression configuration
        self.compression_config = compression_config or DataCompressionConfig()

    def extract_functions(self, file_path: str, content: str) -> list[FunctionChunk]:
        """Extract functions from source code file using universal query-based parsing.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            List of FunctionChunk objects containing functions with their metadata
        """
        if not TREE_SITTER_AVAILABLE or not self.language_registry:
            logger.debug(
                f"Tree-sitter not available for {file_path}, using fallback chunking"
            )
            return self._fallback_extraction(file_path, content)

        # Get file extension
        file_ext = Path(file_path).suffix.lower()

        # Get language configuration
        lang_config = self.language_registry.get_language_for_extension(file_ext)
        if not lang_config:
            logger.debug(f"No language support for {file_ext}, using fallback chunking")
            return self._fallback_extraction(file_path, content)

        try:
            # Parse the content
            tree = lang_config.parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node

            # Extract imports using queries
            imports = self._extract_imports_with_query(lang_config, root_node, content)

            # Extract functions using queries
            functions = self._extract_functions_with_query(
                lang_config, root_node, content
            )

            # Create FunctionChunk objects
            chunks = []
            for func_info in functions:
                chunk = FunctionChunk(
                    function_code=func_info["code"],
                    function_name=func_info["name"],
                    file_path=file_path,
                    is_method=func_info["is_method"],
                    imports=imports,
                    start_line=func_info["start_line"],
                    end_line=func_info["end_line"],
                )
                chunks.append(chunk)

            logger.debug(
                f"Extracted {len(chunks)} functions from {file_path} using {lang_config.name} queries"
            )
            return chunks

        except Exception as e:
            logger.warning(f"Failed to extract functions from {file_path}: {e}")
            return self._fallback_extraction(file_path, content)

    def _extract_imports_with_query(
        self, lang_config: Any, root_node: Any, content: str
    ) -> str:
        """Extract import statements using tree-sitter queries.

        Args:
            lang_config: Language configuration with query string
            root_node: Tree-sitter root node
            content: File content

        Returns:
            String containing all import statements
        """
        try:
            # Create query and cursor
            query = Query(lang_config.parser.language, lang_config.query_string)
            cursor = QueryCursor(query)

            # Execute query - returns dict of {capture_name: [nodes]}
            captures_dict = cursor.captures(root_node)

            # Collect import nodes
            import_lines = []
            import_nodes = captures_dict.get("import", [])

            # Convert content to bytes for correct byte offset slicing
            content_bytes = bytes(content, "utf8")

            for node in import_nodes:
                # Use byte offsets on bytes, then decode
                import_text_bytes = content_bytes[node.start_byte : node.end_byte]
                import_text = import_text_bytes.decode("utf-8")
                import_lines.append(import_text)

            return "\n".join(import_lines)

        except Exception as e:
            logger.debug(f"Failed to extract imports with query: {e}")
            return ""

    def _extract_functions_with_query(
        self, lang_config: Any, root_node: Any, content: str
    ) -> list[dict[str, Any]]:
        """Extract functions using tree-sitter queries (universal approach).

        Args:
            lang_config: Language configuration with query string
            root_node: Tree-sitter root node
            content: File content

        Returns:
            List of function information dictionaries
        """
        functions = []

        try:
            # Create query and cursor
            query = Query(lang_config.parser.language, lang_config.query_string)
            cursor = QueryCursor(query)

            # Execute query - returns dict of {capture_name: [nodes]}
            captures_dict = cursor.captures(root_node)

            # Track which functions we've already seen to avoid duplicates
            seen_functions: set[tuple[str, int]] = set()

            # Convert content to bytes for correct byte offset slicing
            content_bytes = bytes(content, "utf8")

            # Process function.def captures
            function_nodes = captures_dict.get("function.def", [])
            for node in function_nodes:
                func_name = self._extract_name_from_node(node, content, content_bytes)
                # Use byte offsets on bytes, then decode
                func_code_bytes = content_bytes[node.start_byte : node.end_byte]
                func_code = func_code_bytes.decode("utf-8")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                func_key = (func_name, start_line)
                if func_key not in seen_functions:
                    seen_functions.add(func_key)
                    if func_name:
                        functions.append(
                            {
                                "name": func_name,
                                "code": func_code,
                                "is_method": False,
                                "start_line": start_line,
                                "end_line": end_line,
                            }
                        )

            # Process method.def captures
            method_nodes = captures_dict.get("method.def", [])
            for node in method_nodes:
                func_name = self._extract_name_from_node(node, content, content_bytes)
                # Use byte offsets on bytes, then decode
                func_code_bytes = content_bytes[node.start_byte : node.end_byte]
                func_code = func_code_bytes.decode("utf-8")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                func_key = (func_name, start_line)
                if func_key not in seen_functions:
                    seen_functions.add(func_key)
                    if func_name:
                        functions.append(
                            {
                                "name": func_name,
                                "code": func_code,
                                "is_method": True,
                                "start_line": start_line,
                                "end_line": end_line,
                            }
                        )

            logger.debug(
                f"Query-based extraction found {len(functions)} functions using {lang_config.name}"
            )
            return functions

        except Exception as e:
            logger.warning(f"Failed to extract functions with query: {e}")
            return []

    def _extract_name_from_node(
        self, node: Any, content: str, content_bytes: bytes | None = None
    ) -> str:
        """Extract name from a function/method node.

        Args:
            node: Tree-sitter node
            content: File content (string, kept for backwards compatibility)
            content_bytes: File content as bytes (for correct UTF-8 byte offset slicing)

        Returns:
            Function/method name or line-based fallback
        """
        # Convert content to bytes if not provided
        if content_bytes is None:
            content_bytes = bytes(content, "utf8")

        # Try to find identifier child nodes
        for child in node.children:
            if child.type in ["identifier", "property_identifier", "field_identifier"]:
                # Use byte offsets on bytes, then decode
                name_bytes = content_bytes[child.start_byte : child.end_byte]
                return name_bytes.decode("utf-8")

        # For arrow functions or variable declarators, check parent
        if node.parent:
            parent = node.parent
            if parent.type == "variable_declarator":
                for sibling in parent.children:
                    if sibling.type == "identifier":
                        # Use byte offsets on bytes, then decode
                        name_bytes = content_bytes[
                            sibling.start_byte : sibling.end_byte
                        ]
                        return name_bytes.decode("utf-8")

        # Fallback: use line number as identifier
        return f"func_line_{node.start_point[0] + 1}"

    def apply_data_compression(
        self, function_chunk: FunctionChunk, language: str | None = None
    ) -> FunctionChunk:
        """Apply data compression to a function chunk if it's data-heavy.

        Args:
            function_chunk: Function chunk to potentially compress
            language: Language name for language-specific optimization

        Returns:
            Function chunk with compressed data (or original if not data-heavy)
        """
        if not TREE_SITTER_AVAILABLE or not self.compression_config.enabled:
            return function_chunk

        if not self.language_registry:
            return function_chunk

        try:
            # Get language config
            file_ext = Path(function_chunk.file_path).suffix.lower()
            lang_config = self.language_registry.get_language_for_extension(file_ext)

            if not lang_config:
                return function_chunk

            # Parse the function code
            function_code = function_chunk.function_code
            tree = lang_config.parser.parse(bytes(function_code, "utf8"))
            root_node = tree.root_node

            # Create data compressor for this language
            compressor = DataCompressor(
                config=self.compression_config,
                language=language or lang_config.name,
            )

            # Check if function is data-heavy
            if compressor.detect_data_heavy_function(root_node):
                logger.info(
                    f"Compressing data in function {function_chunk.function_name} "
                    f"in {function_chunk.file_path}"
                )

                # Compress the data nodes
                compressed_code = compressor.compress_data_nodes(
                    root_node, bytes(function_code, "utf8")
                )

                # Return new chunk with compressed code
                return FunctionChunk(
                    function_code=compressed_code,
                    function_name=function_chunk.function_name,
                    file_path=function_chunk.file_path,
                    is_method=function_chunk.is_method,
                    imports=function_chunk.imports,
                    start_line=function_chunk.start_line,
                    end_line=function_chunk.end_line,
                )

            return function_chunk

        except Exception as e:
            logger.warning(
                f"Failed to apply data compression to {function_chunk.function_name}: {e}"
            )
            return function_chunk

    def _fallback_extraction(self, file_path: str, content: str) -> list[FunctionChunk]:
        """Fallback extraction when tree-sitter is not available.

        This creates a single chunk with the entire file content.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            List with single FunctionChunk containing entire file
        """
        # Create a single chunk with the whole file
        chunk = FunctionChunk(
            function_code=content,
            function_name=f"file_{Path(file_path).stem}",
            file_path=file_path,
            is_method=False,
            imports="",
            start_line=1,
            end_line=len(content.splitlines()),
        )

        return [chunk]
