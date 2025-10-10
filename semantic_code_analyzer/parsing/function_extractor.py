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

# MIT License
#
# Copyright (c) 2024 Semantic Code Analyzer Contributors
#
# Permission is hereby granted, free of charge to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Function-level code extraction using Tree-sitter for universal AST parsing.

This module provides functionality to extract functions from source code files
across multiple programming languages using tree-sitter parsers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tree-sitter dependencies
try:
    from tree_sitter import Language, Parser
    from tree_sitter_javascript import language as javascript_language
    from tree_sitter_python import language as python_language
    from tree_sitter_typescript import language_typescript as typescript_language

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
    """Extract functions from source code using tree-sitter AST parsing.

    This class provides universal function extraction across multiple programming
    languages, extracting both standalone functions and class methods.
    """

    def __init__(self) -> None:
        """Initialize the FunctionExtractor with language parsers."""
        # Initialize parsers dict (will be empty if tree-sitter not available)
        self.parsers: dict[str, Any] = (
            {}
        )  # Use Any for Parser type to avoid mypy issues

        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "FunctionExtractor initialized without tree-sitter support. "
                "Function extraction will fall back to simple chunking."
            )
            return

        # Python parser
        try:
            python_parser = Parser()  # type: ignore[misc]
            python_parser.set_language(Language(python_language()))  # type: ignore[attr-defined]
            self.parsers[".py"] = python_parser
            logger.debug("Initialized Python parser")
        except Exception as e:
            logger.warning(f"Failed to initialize Python parser: {e}")

        # JavaScript parser
        try:
            js_parser = Parser()  # type: ignore[misc]
            js_parser.set_language(Language(javascript_language()))  # type: ignore[attr-defined]
            self.parsers[".js"] = js_parser
            self.parsers[".jsx"] = js_parser
            logger.debug("Initialized JavaScript parser")
        except Exception as e:
            logger.warning(f"Failed to initialize JavaScript parser: {e}")

        # TypeScript parser
        try:
            ts_parser = Parser()  # type: ignore[misc]
            ts_parser.set_language(Language(typescript_language()))  # type: ignore[attr-defined]
            self.parsers[".ts"] = ts_parser
            self.parsers[".tsx"] = ts_parser
            logger.debug("Initialized TypeScript parser")
        except Exception as e:
            logger.warning(f"Failed to initialize TypeScript parser: {e}")

        logger.info(
            f"FunctionExtractor initialized with {len(self.parsers)} language parsers"
        )

    def extract_functions(self, file_path: str, content: str) -> list[FunctionChunk]:
        """Extract functions from source code file.

        Args:
            file_path: Path to the file being analyzed
            content: Content of the file

        Returns:
            List of FunctionChunk objects containing functions with their metadata
        """
        if not TREE_SITTER_AVAILABLE or not self.parsers:
            logger.debug(
                f"Tree-sitter not available for {file_path}, using fallback chunking"
            )
            return self._fallback_extraction(file_path, content)

        # Get file extension
        file_ext = Path(file_path).suffix.lower()

        if file_ext not in self.parsers:
            logger.debug(f"No parser available for {file_ext}, using fallback chunking")
            return self._fallback_extraction(file_path, content)

        try:
            parser = self.parsers[file_ext]

            # Parse the content
            tree = parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node

            # Extract imports
            imports = self._extract_imports(root_node, content, file_ext)

            # Extract functions and methods
            functions = self._extract_function_nodes(root_node, content, file_ext)

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
                f"Extracted {len(chunks)} functions from {file_path} using tree-sitter"
            )
            return chunks

        except Exception as e:
            logger.warning(f"Failed to extract functions from {file_path}: {e}")
            return self._fallback_extraction(file_path, content)

    def _extract_imports(self, root_node: Any, content: str, file_ext: str) -> str:
        """Extract all import statements from the file.

        Args:
            root_node: Tree-sitter root node
            content: File content
            file_ext: File extension

        Returns:
            String containing all import statements
        """
        import_lines = []

        if file_ext == ".py":
            # Python imports
            import_types = ["import_statement", "import_from_statement"]
        elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
            # JavaScript/TypeScript imports
            import_types = ["import_statement"]
        else:
            return ""

        # Find all import nodes
        for child in root_node.children:
            if child.type in import_types:
                import_text = content[child.start_byte : child.end_byte]
                import_lines.append(import_text)

        return "\n".join(import_lines)

    def _extract_function_nodes(
        self, root_node: Any, content: str, file_ext: str
    ) -> list[dict[str, Any]]:
        """Extract function nodes from the AST.

        Args:
            root_node: Tree-sitter root node
            content: File content
            file_ext: File extension

        Returns:
            List of function information dictionaries
        """
        functions = []

        if file_ext == ".py":
            functions = self._extract_python_functions(root_node, content)
        elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
            functions = self._extract_js_ts_functions(root_node, content)

        return functions

    def _extract_python_functions(
        self, root_node: Any, content: str
    ) -> list[dict[str, Any]]:
        """Extract Python functions and methods.

        Args:
            root_node: Tree-sitter root node
            content: File content

        Returns:
            List of function information dictionaries
        """
        functions = []

        def visit_node(node: Any, is_in_class: bool = False) -> None:
            """Recursively visit nodes to find functions."""
            if node.type == "function_definition":
                func_name = self._get_function_name(node, content)
                func_code = content[node.start_byte : node.end_byte]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                functions.append(
                    {
                        "name": func_name,
                        "code": func_code,
                        "is_method": is_in_class,
                        "start_line": start_line,
                        "end_line": end_line,
                    }
                )

            # Check if entering a class
            entering_class = node.type == "class_definition"

            # Recursively visit children
            for child in node.children:
                visit_node(child, is_in_class=entering_class or is_in_class)

        visit_node(root_node)
        return functions

    def _extract_js_ts_functions(
        self, root_node: Any, content: str
    ) -> list[dict[str, Any]]:
        """Extract JavaScript/TypeScript functions and methods.

        Args:
            root_node: Tree-sitter root node
            content: File content

        Returns:
            List of function information dictionaries
        """
        functions = []

        def visit_node(node: Any, is_in_class: bool = False) -> None:
            """Recursively visit nodes to find functions."""
            # Function declarations and expressions
            if node.type in [
                "function_declaration",
                "function",
                "arrow_function",
                "method_definition",
            ]:
                func_name = self._get_function_name(node, content)
                func_code = content[node.start_byte : node.end_byte]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Skip empty names (anonymous functions without clear context)
                if func_name:
                    functions.append(
                        {
                            "name": func_name,
                            "code": func_code,
                            "is_method": is_in_class
                            or node.type == "method_definition",
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )

            # Check if entering a class
            entering_class = node.type in ["class_declaration", "class"]

            # Recursively visit children
            for child in node.children:
                visit_node(child, is_in_class=entering_class or is_in_class)

        visit_node(root_node)
        return functions

    def _get_function_name(self, node: Any, content: str) -> str:
        """Extract function name from a function node.

        Args:
            node: Tree-sitter function node
            content: File content

        Returns:
            Function name or placeholder
        """
        # Try to find identifier node
        for child in node.children:
            if child.type == "identifier":
                return content[child.start_byte : child.end_byte]
            elif child.type == "property_identifier":
                return content[child.start_byte : child.end_byte]

        # For arrow functions, try to find the variable name
        if node.type == "arrow_function" and node.parent:
            parent = node.parent
            if parent.type == "variable_declarator":
                for sibling in parent.children:
                    if sibling.type == "identifier":
                        return content[sibling.start_byte : sibling.end_byte]

        # Fallback: use line number as identifier
        return f"func_line_{node.start_point[0] + 1}"

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
