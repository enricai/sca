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

"""Universal data compression for code functions using tree-sitter.

This module provides language-agnostic detection and compression of data-heavy
functions (e.g., functions with large embedded strings, arrays, or objects).
This allows the analyzer to focus on code logic patterns rather than embedded data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tree-sitter dependencies
try:
    from tree_sitter import Node

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a placeholder type for Node when tree-sitter is not available
    class Node:  # type: ignore[no-redef]
        """Placeholder Node class when tree-sitter is not available."""

        pass

    logger.warning("Tree-sitter not available. Data compression will be disabled.")


# Universal node types that contain data (not logic) across all languages
DATA_NODE_TYPES = {
    # String types (universal)
    "string",
    "string_literal",
    "string_content",
    "string_fragment",
    "raw_string_literal",
    "template_string",
    "interpreted_string_literal",
    "text_block",
    # Array/List types
    "array",
    "array_literal",
    "array_expression",
    "array_initializer",
    "list",
    "list_literal",
    "slice_literal",
    # Object/Dict types
    "object",
    "object_literal",
    "dictionary",
    "composite_literal",
    "struct_expression",
    # JSX/React specific
    "jsx_text",
    "jsx_expression",
}

# Language-specific data node mappings for fine-tuned detection
LANGUAGE_DATA_NODES: dict[str, set[str]] = {
    "python": {
        "string",
        "string_content",
        "list",
        "dictionary",
        "list_comprehension",
        "set",
        "tuple",
    },
    "javascript": {
        "string",
        "template_string",
        "array",
        "object",
        "string_fragment",
        "jsx_text",
    },
    "typescript": {
        "string",
        "template_string",
        "array",
        "object",
        "string_fragment",
        "jsx_text",
        "type_annotation",
    },
    "java": {"string_literal", "array_initializer", "text_block"},
    "go": {
        "interpreted_string_literal",
        "raw_string_literal",
        "composite_literal",
        "slice_literal",
    },
    "rust": {
        "string_literal",
        "raw_string_literal",
        "array_expression",
        "struct_expression",
    },
    "c": {"string_literal", "initializer_list"},
    "cpp": {"string_literal", "initializer_list", "raw_string_literal"},
    "ruby": {"string", "string_literal", "array", "hash"},
    "php": {"string", "array", "encapsed_string"},
}


@dataclass
class DataCompressionConfig:
    """Configuration for data compression in functions.

    Attributes:
        enabled: Whether data compression is enabled
        threshold_ratio: Minimum ratio of data to total bytes (0.0 to 1.0)
        min_data_size: Minimum data size in bytes to trigger compression
        max_string_size: Maximum string size before compression (in characters)
        max_array_items: Maximum number of array items to keep uncompressed

    Raises:
        ValueError: If configuration values are invalid
    """

    enabled: bool = True
    threshold_ratio: float = 0.7
    min_data_size: int = 1000
    max_string_size: int = 100
    max_array_items: int = 5

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If any parameter is invalid
        """
        if not 0.0 <= self.threshold_ratio <= 1.0:
            raise ValueError(
                f"threshold_ratio must be between 0.0 and 1.0, got {self.threshold_ratio}"
            )

        if self.min_data_size < 0:
            raise ValueError(
                f"min_data_size must be non-negative, got {self.min_data_size}"
            )

        if self.max_string_size < 0:
            raise ValueError(
                f"max_string_size must be non-negative, got {self.max_string_size}"
            )

        if self.max_array_items < 0:
            raise ValueError(
                f"max_array_items must be non-negative, got {self.max_array_items}"
            )


class DataCompressor:
    """Universal data compression using tree-sitter AST analysis.

    This class provides language-agnostic detection and compression of data-heavy
    functions. It identifies functions where data (strings, arrays, objects) comprises
    a significant portion of the content and compresses that data while preserving
    the code structure and logic patterns.
    """

    def __init__(
        self, config: DataCompressionConfig | None = None, language: str | None = None
    ):
        """Initialize the DataCompressor.

        Args:
            config: Data compression configuration (uses defaults if None)
            language: Language name for language-specific optimizations
        """
        self.config = config or DataCompressionConfig()
        self.language = language

        # Get language-specific data node types or fall back to universal set
        if language and language in LANGUAGE_DATA_NODES:
            self.data_node_types = LANGUAGE_DATA_NODES[language]
            logger.debug(
                f"Using language-specific data nodes for {language}: {len(self.data_node_types)} types"
            )
        else:
            self.data_node_types = DATA_NODE_TYPES
            logger.debug(
                f"Using universal data nodes: {len(self.data_node_types)} types"
            )

    def detect_data_heavy_function(self, function_node: Node) -> bool:
        """Detect if a function is data-heavy using AST analysis.

        A function is considered data-heavy if:
        1. Data nodes comprise more than threshold_ratio of the total bytes, AND
        2. The total data size exceeds min_data_size bytes

        Args:
            function_node: Tree-sitter function node

        Returns:
            True if function is data-heavy, False otherwise
        """
        if not TREE_SITTER_AVAILABLE:
            return False

        if not self.config.enabled:
            return False

        try:
            node_text = function_node.text
            if node_text is None:
                return False

            total_bytes = len(node_text)
            if total_bytes == 0:
                return False

            data_bytes = self._count_data_bytes(function_node)
            data_ratio = data_bytes / total_bytes

            is_data_heavy = (
                data_ratio > self.config.threshold_ratio
                and data_bytes >= self.config.min_data_size
            )

            if is_data_heavy:
                logger.debug(
                    f"Data-heavy function detected: {data_bytes}/{total_bytes} bytes "
                    f"({data_ratio:.2%} data ratio)"
                )

            return is_data_heavy

        except Exception as e:
            logger.warning(f"Failed to detect data-heavy function: {e}")
            return False

    def _count_data_bytes(self, node: Node) -> int:
        """Recursively count bytes in data nodes.

        Args:
            node: Tree-sitter node

        Returns:
            Total bytes in data nodes
        """
        if node.type in self.data_node_types:
            node_text = node.text
            if node_text is None:
                return 0
            return len(node_text)

        total = 0
        for child in node.children:
            total += self._count_data_bytes(child)

        return total

    def compress_data_nodes(self, function_node: Node, original_text: bytes) -> str:
        """Compress data nodes in a function while preserving structure.

        This method replaces large data nodes with appropriate placeholders,
        keeping the code structure intact so that coding patterns remain visible.

        Args:
            function_node: Tree-sitter function node
            original_text: Original source code as bytes

        Returns:
            Modified source code with data compressed
        """
        if not TREE_SITTER_AVAILABLE:
            return original_text.decode("utf-8")

        if not self.config.enabled:
            return original_text.decode("utf-8")

        try:
            # Collect all data nodes that need compression
            replacements: list[dict[str, Any]] = []
            self._find_data_nodes(function_node, replacements)

            if not replacements:
                return original_text.decode("utf-8")

            # Sort by position (reverse order for safe replacement)
            replacements.sort(key=lambda x: x["start"], reverse=True)

            # Apply replacements (work with bytes to preserve UTF-8 byte offsets)
            modified_bytes = original_text
            for repl in replacements:
                start, end = repl["start"], repl["end"]
                original = repl["original"]

                # Create appropriate placeholder based on type
                placeholder = self._create_placeholder(original, repl["type"])
                placeholder_bytes = placeholder.encode("utf-8")

                # Replace in bytes (byte offsets from tree-sitter are UTF-8 byte positions)
                modified_bytes = (
                    modified_bytes[:start] + placeholder_bytes + modified_bytes[end:]
                )

            logger.debug(f"Compressed {len(replacements)} data nodes")
            return modified_bytes.decode("utf-8")

        except Exception as e:
            logger.warning(f"Failed to compress data nodes: {e}")
            return original_text.decode("utf-8")

    def _find_data_nodes(self, node: Node, replacements: list[dict[str, Any]]) -> None:
        """Find all data nodes that need compression.

        Args:
            node: Tree-sitter node to analyze
            replacements: List to append replacement info to
        """
        if node.type in self.data_node_types:
            node_text = node.text
            if node_text is None:
                return

            node_size = len(node_text)

            # Check if this data node needs compression
            if self._should_compress_node(node, node_size):
                replacements.append(
                    {
                        "start": node.start_byte,
                        "end": node.end_byte,
                        "original": node_text.decode("utf-8"),
                        "type": node.type,
                    }
                )
                # Don't recurse into children of data nodes we're compressing
                return

        # Recurse into children
        for child in node.children:
            self._find_data_nodes(child, replacements)

    def _should_compress_node(self, node: Node, node_size: int) -> bool:
        """Determine if a data node should be compressed.

        Args:
            node: Tree-sitter node
            node_size: Size of node in bytes

        Returns:
            True if node should be compressed
        """
        # Compress strings longer than max_string_size
        if "string" in node.type:
            return node_size > self.config.max_string_size

        # Compress large arrays/objects
        if any(t in node.type for t in ["array", "list", "object", "dict"]):
            return node_size > self.config.max_string_size

        # Default: compress if larger than threshold
        return node_size > self.config.max_string_size

    def _create_placeholder(self, original: str, node_type: str) -> str:
        """Create appropriate placeholder for data type.

        Args:
            original: Original data content
            node_type: Tree-sitter node type

        Returns:
            Placeholder string that preserves structure
        """
        # String types: keep delimiters, replace content
        if "string" in node_type:
            return self._create_string_placeholder(original)

        # Array/List types: keep first few items
        if any(t in node_type for t in ["array", "list"]):
            return self._create_array_placeholder(original)

        # Object/Dict types: indicate omitted data
        if any(t in node_type for t in ["object", "dict", "struct", "composite"]):
            return self._create_object_placeholder(original)

        # JSX text: preserve structure
        if "jsx" in node_type:
            return "..."

        # Default fallback
        return "/* data omitted */"

    def _create_string_placeholder(self, original: str) -> str:
        """Create placeholder for string data.

        Args:
            original: Original string content

        Returns:
            Compressed string placeholder
        """
        # Detect delimiter type
        if original.startswith('"""') or original.startswith("'''"):
            # Python triple-quoted string
            delim = original[:3]
            return f"{delim}...{delim}"
        elif original.startswith('"'):
            return '"..."'
        elif original.startswith("'"):
            return "'...'"
        elif original.startswith("`"):
            return "`...`"
        else:
            # No clear delimiter, use generic
            return '"..."'

    def _create_array_placeholder(self, original: str) -> str:
        """Create placeholder for array/list data.

        Args:
            original: Original array content

        Returns:
            Compressed array placeholder with first few items
        """
        # Try to extract first few items
        items = self._extract_array_items(original)

        if len(items) > self.config.max_array_items:
            # Keep first N items and add ellipsis
            kept_items = items[: self.config.max_array_items]
            items_str = ", ".join(kept_items)

            # Detect bracket style
            if original.strip().startswith("["):
                return f"[{items_str}, /* ... */]"
            elif original.strip().startswith("{"):
                return f"{{{items_str}, /* ... */}}"
            else:
                return f"[{items_str}, /* ... */]"

        # Array is small enough, keep as-is
        return original

    def _create_object_placeholder(self, original: str) -> str:
        """Create placeholder for object/dict data.

        Args:
            original: Original object content

        Returns:
            Compressed object placeholder
        """
        # Detect bracket style
        if original.strip().startswith("{"):
            return "{/* ... */}"
        elif original.strip().startswith("("):
            return "(/* ... */)"
        else:
            return "{/* ... */}"

    def _extract_array_items(self, array_text: str) -> list[str]:
        """Extract items from array literal.

        This is a simple heuristic-based approach that splits by comma.
        For more complex cases, full AST parsing would be better, but this
        is sufficient for compression purposes.

        Args:
            array_text: Array/list literal text

        Returns:
            List of extracted items (as strings)
        """
        # Remove brackets
        content = array_text.strip()
        for bracket_pair in (("[]", "[", "]"), ("{}", "{", "}")):
            if content.startswith(bracket_pair[1]) and content.endswith(
                bracket_pair[2]
            ):
                content = content[1:-1]
                break

        # Simple comma split (not perfect for nested structures, but good enough)
        items = re.split(r",\s*", content)

        # Filter out empty items (but don't limit length here - let caller decide)
        items = [item.strip() for item in items if item.strip()]
        return items


def get_data_node_types(language: str) -> set[str]:
    """Get data node types for a specific language.

    Args:
        language: Language name (e.g., 'python', 'typescript')

    Returns:
        Set of data node type names for the language
    """
    return LANGUAGE_DATA_NODES.get(language, DATA_NODE_TYPES)
