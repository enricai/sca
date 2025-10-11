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

"""Tests for the data compressor module."""

from __future__ import annotations

import pytest

from semantic_code_analyzer.parsing.data_compressor import (
    DATA_NODE_TYPES,
    LANGUAGE_DATA_NODES,
    DataCompressionConfig,
    DataCompressor,
    get_data_node_types,
)

# Try to import tree-sitter for tests
try:
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Try to import language parsers for testing
LANGUAGES_AVAILABLE: dict[str, bool] = {}

if TREE_SITTER_AVAILABLE:
    for lang_name in ["python", "typescript", "javascript", "go"]:
        try:
            if lang_name == "python":
                from tree_sitter_python import language as python_lang  # noqa: F401

                LANGUAGES_AVAILABLE["python"] = True
            elif lang_name == "typescript":
                from tree_sitter_typescript import language_typescript  # noqa: F401

                LANGUAGES_AVAILABLE["typescript"] = True
            elif lang_name == "javascript":
                from tree_sitter_javascript import language as js_lang  # noqa: F401

                LANGUAGES_AVAILABLE["javascript"] = True
            elif lang_name == "go":
                from tree_sitter_go import language as go_lang  # noqa: F401

                LANGUAGES_AVAILABLE["go"] = True
        except ImportError:
            LANGUAGES_AVAILABLE[lang_name] = False


class TestDataCompressionConfig:
    """Test cases for DataCompressionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DataCompressionConfig()

        assert config.enabled is True
        assert config.threshold_ratio == 0.7
        assert config.min_data_size == 1000
        assert config.max_string_size == 100
        assert config.max_array_items == 5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DataCompressionConfig(
            enabled=False,
            threshold_ratio=0.8,
            min_data_size=500,
            max_string_size=50,
            max_array_items=3,
        )

        assert config.enabled is False
        assert config.threshold_ratio == 0.8
        assert config.min_data_size == 500
        assert config.max_string_size == 50
        assert config.max_array_items == 3

    def test_config_validation_threshold_ratio_too_high(self) -> None:
        """Test that threshold_ratio > 1.0 raises ValueError."""
        with pytest.raises(
            ValueError, match="threshold_ratio must be between 0.0 and 1.0"
        ):
            DataCompressionConfig(threshold_ratio=1.5)

    def test_config_validation_threshold_ratio_negative(self) -> None:
        """Test that negative threshold_ratio raises ValueError."""
        with pytest.raises(
            ValueError, match="threshold_ratio must be between 0.0 and 1.0"
        ):
            DataCompressionConfig(threshold_ratio=-0.1)

    def test_config_validation_negative_min_data_size(self) -> None:
        """Test that negative min_data_size raises ValueError."""
        with pytest.raises(ValueError, match="min_data_size must be non-negative"):
            DataCompressionConfig(min_data_size=-100)

    def test_config_validation_negative_max_string_size(self) -> None:
        """Test that negative max_string_size raises ValueError."""
        with pytest.raises(ValueError, match="max_string_size must be non-negative"):
            DataCompressionConfig(max_string_size=-50)

    def test_config_validation_negative_max_array_items(self) -> None:
        """Test that negative max_array_items raises ValueError."""
        with pytest.raises(ValueError, match="max_array_items must be non-negative"):
            DataCompressionConfig(max_array_items=-5)


class TestDataCompressor:
    """Test cases for DataCompressor."""

    def test_initialization_universal(self) -> None:
        """Test initialization with universal data nodes."""
        compressor = DataCompressor()

        assert compressor.config.enabled is True
        assert compressor.language is None
        assert compressor.data_node_types == DATA_NODE_TYPES

    def test_initialization_language_specific(self) -> None:
        """Test initialization with language-specific data nodes."""
        compressor = DataCompressor(language="python")

        assert compressor.language == "python"
        assert compressor.data_node_types == LANGUAGE_DATA_NODES["python"]

    def test_initialization_unknown_language(self) -> None:
        """Test initialization with unknown language falls back to universal."""
        compressor = DataCompressor(language="unknown")

        assert compressor.language == "unknown"
        assert compressor.data_node_types == DATA_NODE_TYPES

    def test_disabled_compression(self) -> None:
        """Test that compression is disabled when config is disabled."""
        config = DataCompressionConfig(enabled=False)
        compressor = DataCompressor(config=config)

        # Mock a function node
        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        # Even with a data-heavy function, should return False
        assert compressor.config.enabled is False

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("python", False),
        reason="tree-sitter or tree-sitter-python not available",
    )
    def test_detect_data_heavy_python_svg_icon(self) -> None:
        """Test detection of data-heavy Python function with SVG data."""
        from tree_sitter_python import language as python_lang

        parser = Parser(Language(python_lang()))

        # Python function with large SVG path (simulated)
        python_code = """
def render_icon():
    svg_path = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" * 50
    return f'<svg><path d="{svg_path}"/></svg>'
"""

        tree = parser.parse(bytes(python_code, "utf8"))
        root_node = tree.root_node

        # Find function definition
        function_node = root_node.children[0]

        compressor = DataCompressor(language="python")
        is_data_heavy = compressor.detect_data_heavy_function(function_node)

        assert is_data_heavy is True

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("python", False),
        reason="tree-sitter or tree-sitter-python not available",
    )
    def test_detect_normal_function(self) -> None:
        """Test that normal functions are not detected as data-heavy."""
        from tree_sitter_python import language as python_lang

        parser = Parser(Language(python_lang()))

        # Normal Python function with logic
        python_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total
"""

        tree = parser.parse(bytes(python_code, "utf8"))
        root_node = tree.root_node

        # Find function definition
        function_node = root_node.children[0]

        compressor = DataCompressor(language="python")
        is_data_heavy = compressor.detect_data_heavy_function(function_node)

        assert is_data_heavy is False

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("typescript", False),
        reason="tree-sitter or tree-sitter-typescript not available",
    )
    def test_compress_typescript_jsx_icon(self) -> None:
        """Test compression of TypeScript React icon component with SVG."""
        from tree_sitter_typescript import language_typescript

        parser = Parser(Language(language_typescript()))

        # TypeScript icon component with large SVG path
        tsx_code = """
export function AnswerIcon(props: SVGProps<SVGSVGElement>) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
    );
}
"""

        tree = parser.parse(bytes(tsx_code, "utf8"))
        root_node = tree.root_node

        # Find function declaration
        function_node = None
        for child in root_node.children:
            if child.type in ["function_declaration", "export_statement"]:
                # Navigate to find the actual function
                for subchild in child.children:
                    if subchild.type == "function_declaration":
                        function_node = subchild
                        break
                if function_node:
                    break
                if child.type == "function_declaration":
                    function_node = child
                    break

        if function_node is None:
            # If we couldn't find function_declaration, use root
            function_node = root_node

        compressor = DataCompressor(language="typescript")
        compressed = compressor.compress_data_nodes(
            function_node, bytes(tsx_code, "utf8")
        )

        # Check that string was compressed
        assert "..." in compressed
        assert len(compressed) < len(tsx_code)

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("python", False),
        reason="tree-sitter or tree-sitter-python not available",
    )
    def test_compress_python_sql_query(self) -> None:
        """Test compression of Python function with large SQL query."""
        from tree_sitter_python import language as python_lang

        parser = Parser(Language(python_lang()))

        # Python function with large SQL query
        python_code = '''
def get_user_report():
    query = """
        SELECT users.id, users.name, users.email, users.age,
               orders.id, orders.total, orders.date, orders.status,
               products.name, products.price, products.category
        FROM users
        JOIN orders ON users.id = orders.user_id
        JOIN order_items ON orders.id = order_items.order_id
        JOIN products ON order_items.product_id = products.id
        WHERE users.active = TRUE
        GROUP BY users.id, orders.id, products.id
        ORDER BY orders.date DESC
    """
    return db.execute(query)
'''

        tree = parser.parse(bytes(python_code, "utf8"))
        root_node = tree.root_node

        # Find function definition
        function_node = root_node.children[0]

        compressor = DataCompressor(language="python")
        compressed = compressor.compress_data_nodes(
            function_node, bytes(python_code, "utf8")
        )

        # Check that SQL query was compressed
        assert '"""..."""' in compressed
        assert len(compressed) < len(python_code)
        # Logic should remain
        assert "db.execute(query)" in compressed

    def test_create_string_placeholder(self) -> None:
        """Test string placeholder creation."""
        compressor = DataCompressor()

        # Double quotes
        assert compressor._create_string_placeholder('"test"') == '"..."'

        # Single quotes
        assert compressor._create_string_placeholder("'test'") == "'...'"

        # Backticks
        assert compressor._create_string_placeholder("`test`") == "`...`"

        # Triple quotes
        assert compressor._create_string_placeholder('"""test"""') == '"""..."""'
        assert compressor._create_string_placeholder("'''test'''") == "'''...'''"

    def test_create_array_placeholder_small(self) -> None:
        """Test array placeholder for small arrays."""
        compressor = DataCompressor()

        small_array = "[1, 2, 3]"
        result = compressor._create_array_placeholder(small_array)

        # Small arrays should be kept as-is
        assert result == small_array

    def test_create_array_placeholder_large(self) -> None:
        """Test array placeholder for large arrays."""
        compressor = DataCompressor()

        # Array with more than max_array_items (5)
        large_array = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
        result = compressor._create_array_placeholder(large_array)

        # Should keep first 5 items and add ellipsis
        assert "1, 2, 3, 4, 5" in result
        assert "/* ... */" in result
        assert "12" not in result

    def test_create_object_placeholder(self) -> None:
        """Test object placeholder creation."""
        compressor = DataCompressor()

        # Curly braces
        assert compressor._create_object_placeholder("{}") == "{/* ... */}"
        assert compressor._create_object_placeholder("{a: 1, b: 2}") == "{/* ... */}"

    def test_extract_array_items(self) -> None:
        """Test array item extraction."""
        compressor = DataCompressor()

        # Simple array - should extract all items
        items = compressor._extract_array_items("[1, 2, 3, 4, 5, 6]")
        assert len(items) == 6
        assert items == ["1", "2", "3", "4", "5", "6"]

        # Array with brackets
        items = compressor._extract_array_items("[a, b, c]")
        assert items == ["a", "b", "c"]

        # Object-style array - should extract all items
        items = compressor._extract_array_items("{a, b, c, d, e, f}")
        assert len(items) == 6
        assert items == ["a", "b", "c", "d", "e", "f"]


class TestDataNodeTypes:
    """Test cases for data node type management."""

    def test_universal_data_node_types(self) -> None:
        """Test that universal data node types are defined."""
        assert isinstance(DATA_NODE_TYPES, set)
        assert len(DATA_NODE_TYPES) > 0

        # Check for common types
        assert "string" in DATA_NODE_TYPES
        assert "array" in DATA_NODE_TYPES
        assert "object" in DATA_NODE_TYPES

    def test_language_specific_data_nodes(self) -> None:
        """Test language-specific data node mappings."""
        assert isinstance(LANGUAGE_DATA_NODES, dict)

        # Check for common languages
        assert "python" in LANGUAGE_DATA_NODES
        assert "javascript" in LANGUAGE_DATA_NODES
        assert "typescript" in LANGUAGE_DATA_NODES

        # Verify Python-specific nodes
        python_nodes = LANGUAGE_DATA_NODES["python"]
        assert "string" in python_nodes
        assert "list" in python_nodes
        assert "dictionary" in python_nodes

    def test_get_data_node_types(self) -> None:
        """Test get_data_node_types function."""
        # Known language
        python_nodes = get_data_node_types("python")
        assert python_nodes == LANGUAGE_DATA_NODES["python"]

        # Unknown language should return universal set
        unknown_nodes = get_data_node_types("unknown_language")
        assert unknown_nodes == DATA_NODE_TYPES


class TestIntegrationScenarios:
    """Integration test scenarios for real-world use cases."""

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("python", False),
        reason="tree-sitter or tree-sitter-python not available",
    )
    def test_python_html_template(self) -> None:
        """Test compression of Python function with large HTML template."""
        from tree_sitter_python import language as python_lang

        parser = Parser(Language(python_lang()))

        python_code = '''
def render_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <div class="container">
            <h1>Welcome</h1>
            <p>This is a long HTML template with many elements.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html
'''

        tree = parser.parse(bytes(python_code, "utf8"))
        function_node = tree.root_node.children[0]

        compressor = DataCompressor(language="python")

        # Should detect as data-heavy
        is_data_heavy = compressor.detect_data_heavy_function(function_node)
        assert is_data_heavy is True

        # Should compress the HTML
        compressed = compressor.compress_data_nodes(
            function_node, bytes(python_code, "utf8")
        )
        assert '"""..."""' in compressed
        assert "return html" in compressed

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("javascript", False),
        reason="tree-sitter or tree-sitter-javascript not available",
    )
    def test_javascript_base64_data(self) -> None:
        """Test compression of JavaScript function with base64 data."""
        from tree_sitter_javascript import language as js_lang

        parser = Parser(Language(js_lang()))

        js_code = """
function getEmbeddedImage() {
    const base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" +
                   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" +
                   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    return `data:image/png;base64,${base64}`;
}
"""

        tree = parser.parse(bytes(js_code, "utf8"))
        function_node = tree.root_node.children[0]

        compressor = DataCompressor(language="javascript")

        # Should detect as data-heavy
        is_data_heavy = compressor.detect_data_heavy_function(function_node)
        assert is_data_heavy is True

        # Should compress the base64 strings
        compressed = compressor.compress_data_nodes(
            function_node, bytes(js_code, "utf8")
        )
        assert '"..."' in compressed
        assert "return" in compressed

    @pytest.mark.skipif(
        not TREE_SITTER_AVAILABLE or not LANGUAGES_AVAILABLE.get("python", False),
        reason="tree-sitter or tree-sitter-python not available",
    )
    def test_unicode_handling(self) -> None:
        """Test that compression correctly handles multi-byte Unicode characters."""
        from tree_sitter_python import language as python_lang

        parser = Parser(Language(python_lang()))

        # Python function with Unicode characters and large data
        python_code = '''
def render_message():
    # Unicode characters: 你好世界 😀🎉🚀
    message = """
        Hello 世界! This is a test with emojis 😀🎉🚀
        And some more text to make it large enough to compress.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Unicode: 你好世界 مرحبا בעולם Привет мир
    """ * 10
    return f"Message: {message}"
'''

        tree = parser.parse(bytes(python_code, "utf8"))
        function_node = tree.root_node.children[0]

        compressor = DataCompressor(language="python")

        # Compress the data
        compressed = compressor.compress_data_nodes(
            function_node, bytes(python_code, "utf8")
        )

        # Verify compression worked and didn't corrupt Unicode
        assert '"""..."""' in compressed
        assert "render_message" in compressed
        assert "return" in compressed
        # Verify we can decode it (no UTF-8 errors)
        assert isinstance(compressed, str)
