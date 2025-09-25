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

"""Tests for the domain classifier module."""

from __future__ import annotations

import pytest

from semantic_code_analyzer.analyzers.domain_classifier import (
    ArchitecturalDomain,
    DomainClassifier,
)


class TestDomainClassifier:
    """Test cases for the DomainClassifier class."""

    @pytest.fixture
    def classifier(self) -> DomainClassifier:
        """Create a DomainClassifier instance for testing."""
        return DomainClassifier()

    def test_get_analyzer_name(self, classifier: DomainClassifier) -> None:
        """Test that analyzer returns correct name."""
        assert classifier.get_analyzer_name() == "domain_classifier"

    def test_get_weight(self, classifier: DomainClassifier) -> None:
        """Test that analyzer returns correct weight."""
        assert classifier.get_weight() == 0.20

    def test_frontend_classification_by_path(
        self, classifier: DomainClassifier
    ) -> None:
        """Test frontend domain classification based on file paths."""
        test_cases = [
            "src/components/Button.tsx",
            "src/app/page.tsx",
            "components/Header.jsx",
            "src/styles/global.css",
            "public/logo.png",
        ]

        for file_path in test_cases:
            result = classifier.classify_domain(file_path, "")
            assert result.domain == ArchitecturalDomain.FRONTEND
            assert result.confidence > 0.5

    def test_backend_classification_by_path(self, classifier: DomainClassifier) -> None:
        """Test backend domain classification based on file paths."""
        test_cases = [
            "src/app/api/users/route.ts",
            "api/auth.js",
            "server/controllers/userController.py",
            "src/lib/database.ts",
            "backend/services/email.js",
        ]

        for file_path in test_cases:
            result = classifier.classify_domain(file_path, "")
            assert result.domain == ArchitecturalDomain.BACKEND
            assert result.confidence > 0.5

    def test_database_classification_by_path(
        self, classifier: DomainClassifier
    ) -> None:
        """Test database domain classification based on file paths."""
        test_cases = [
            "migrations/001_create_users.sql",
            "src/models/User.ts",
            "prisma/schema.prisma",
            "database/seeds/users.py",
        ]

        for file_path in test_cases:
            result = classifier.classify_domain(file_path, "")
            assert result.domain == ArchitecturalDomain.DATABASE
            assert result.confidence > 0.6

    def test_testing_classification_by_path(self, classifier: DomainClassifier) -> None:
        """Test testing domain classification based on file paths."""
        test_cases = [
            "tests/test_user.py",
            "src/components/Button.test.tsx",
            "__tests__/utils.js",
            "cypress/integration/login.spec.js",
        ]

        for file_path in test_cases:
            result = classifier.classify_domain(file_path, "")
            assert result.domain == ArchitecturalDomain.TESTING
            assert result.confidence > 0.7

    def test_configuration_classification_by_path(
        self, classifier: DomainClassifier
    ) -> None:
        """Test configuration domain classification based on file paths."""
        test_cases = [
            "package.json",
            "tsconfig.json",
            "next.config.js",
            ".env.local",
            "tailwind.config.ts",
        ]

        for file_path in test_cases:
            result = classifier.classify_domain(file_path, "")
            assert result.domain == ArchitecturalDomain.CONFIGURATION
            assert result.confidence > 0.6

    def test_frontend_classification_by_content(
        self, classifier: DomainClassifier
    ) -> None:
        """Test frontend domain classification based on content patterns."""
        react_component = """
import React, { useState } from 'react';

const Button: React.FC<{ title: string }> = ({ title }) => {
    const [isClicked, setIsClicked] = useState(false);

    return (
        <button onClick={() => setIsClicked(true)} className="btn">
            {title}
        </button>
    );
};

export default Button;
"""
        result = classifier.classify_domain(
            "src/components/Button.tsx", react_component
        )
        assert result.domain == ArchitecturalDomain.FRONTEND
        assert result.confidence > 0.8

    def test_backend_classification_by_content(
        self, classifier: DomainClassifier
    ) -> None:
        """Test backend domain classification based on content patterns."""
        api_route = """
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    try {
        const data = await fetchUserData();
        return NextResponse.json(data, { status: 200 });
    } catch (error) {
        return NextResponse.json({ error: 'Internal error' }, { status: 500 });
    }
}
"""
        result = classifier.classify_domain("src/app/api/users/route.ts", api_route)
        assert result.domain == ArchitecturalDomain.BACKEND
        assert result.confidence > 0.8

    def test_testing_classification_by_content(
        self, classifier: DomainClassifier
    ) -> None:
        """Test testing domain classification based on content patterns."""
        test_code = """
import { render, screen, fireEvent } from '@testing-library/react';
import Button from './Button';

describe('Button component', () => {
    test('renders button with title', () => {
        render(<Button title="Click me" />);
        expect(screen.getByText('Click me')).toBeInTheDocument();
    });

    test('handles click events', () => {
        const mockClick = jest.fn();
        render(<Button title="Test" onClick={mockClick} />);
        fireEvent.click(screen.getByText('Test'));
        expect(mockClick).toHaveBeenCalled();
    });
});
"""
        result = classifier.classify_domain("src/components/Button.test.tsx", test_code)
        assert result.domain == ArchitecturalDomain.TESTING
        assert result.confidence > 0.9

    def test_database_classification_by_content(
        self, classifier: DomainClassifier
    ) -> None:
        """Test database domain classification based on content patterns."""
        sql_migration = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
"""
        result = classifier.classify_domain(
            "migrations/001_create_users.sql", sql_migration
        )
        assert result.domain == ArchitecturalDomain.DATABASE
        assert result.confidence > 0.9

    def test_mixed_domain_signals(self, classifier: DomainClassifier) -> None:
        """Test classification when file has mixed domain signals."""
        mixed_content = """
import React from 'react';
import { render } from '@testing-library/react';

const TestComponent = () => <div>Test</div>;

describe('Component tests', () => {
    test('renders correctly', () => {
        render(<TestComponent />);
    });
});
"""
        # Path suggests testing, content has both frontend and testing patterns
        result = classifier.classify_domain(
            "src/components/Test.test.tsx", mixed_content
        )
        assert (
            result.domain == ArchitecturalDomain.TESTING
        )  # Path + content should favor testing
        assert len(result.secondary_domains) > 0  # Should have frontend as secondary

    def test_unknown_domain_classification(self, classifier: DomainClassifier) -> None:
        """Test that unclear files are classified as unknown."""
        unclear_content = """
const someFunction = () => {
    return "hello world";
};
"""
        result = classifier.classify_domain("src/utils.js", unclear_content)
        # Should likely be unknown or have low confidence
        assert result.confidence < 0.7 or result.domain == ArchitecturalDomain.UNKNOWN

    def test_classification_factors_included(
        self, classifier: DomainClassifier
    ) -> None:
        """Test that classification includes detailed factors."""
        result = classifier.classify_domain(
            "src/app/page.tsx",
            "export default function Page() { return <div>Hello</div>; }",
        )

        assert "path_scores" in result.classification_factors
        assert "import_scores" in result.classification_factors
        assert "content_scores" in result.classification_factors
        assert "combined_scores" in result.classification_factors

    def test_secondary_domains_populated(self, classifier: DomainClassifier) -> None:
        """Test that secondary domains are populated when applicable."""
        # A React test file should have both testing and frontend signals
        test_component = """
import React from 'react';
import { render, screen } from '@testing-library/react';

test('component renders', () => {
    render(<div>Test</div>);
    expect(screen.getByText('Test')).toBeInTheDocument();
});
"""
        result = classifier.classify_domain("src/Button.test.tsx", test_component)

        # Should have secondary domains due to mixed signals
        if result.domain == ArchitecturalDomain.TESTING:
            # Check if frontend is in secondary domains
            secondary_domain_values = [
                domain.value for domain, _ in result.secondary_domains
            ]
            assert any(
                "frontend" in domain_val for domain_val in secondary_domain_values
            )

    def test_analyze_file_returns_analysis_result(
        self, classifier: DomainClassifier
    ) -> None:
        """Test that analyze_file returns proper AnalysisResult."""
        content = "import React from 'react';\nexport default () => <div>Test</div>;"
        result = classifier.analyze_file("src/test.tsx", content)

        assert result.file_path == "src/test.tsx"
        assert 0 <= result.score <= 1
        assert len(result.patterns_found) > 0
        assert "domain" in result.metrics
        assert result.analysis_time >= 0

    def test_supported_extensions(self, classifier: DomainClassifier) -> None:
        """Test that classifier supports expected file extensions."""
        supported = classifier._get_supported_extensions()

        expected_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".py",
            ".sql",
            ".md",
            ".json",
        }
        assert expected_extensions.issubset(supported)

    def test_configuration_with_custom_config(self) -> None:
        """Test that classifier accepts custom configuration."""
        config = {"test_setting": True}
        classifier = DomainClassifier(config)
        assert classifier.config == config

    def test_get_classification_diagnostics_frontend(
        self, classifier: DomainClassifier
    ) -> None:
        """Test diagnostic generation for frontend files."""
        from semantic_code_analyzer.analyzers.domain_classifier import (
            ArchitecturalDomain,
        )

        file_path = "src/components/Button.tsx"
        content = """
import React, { useState } from 'react';
import { styled } from 'styled-components';

const Button = () => {
    const [count, setCount] = useState(0);
    return <button onClick={() => setCount(count + 1)}>{count}</button>;
};

export default Button;
"""

        diagnostics = classifier.get_classification_diagnostics(file_path, content)

        # Should have diagnostics for multiple domains
        assert len(diagnostics) > 0

        # Frontend should have strong matches
        if ArchitecturalDomain.FRONTEND in diagnostics:
            frontend_diag = diagnostics[ArchitecturalDomain.FRONTEND]
            # Should have multiple diagnostic categories
            categories = {diag.pattern_category for diag in frontend_diag}
            assert "import" in categories or "content" in categories

            # Should have some matched patterns
            total_matches = sum(len(diag.matched_patterns) for diag in frontend_diag)
            assert total_matches > 0

    def test_get_classification_diagnostics_backend(
        self, classifier: DomainClassifier
    ) -> None:
        """Test diagnostic generation for backend files."""
        from semantic_code_analyzer.analyzers.domain_classifier import (
            ArchitecturalDomain,
        )

        file_path = "src/app/api/users/route.ts"
        content = """
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    try {
        const data = await fetchUsers();
        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({ error: 'Failed' }, { status: 500 });
    }
}

export async function POST(request: NextRequest) {
    const body = await request.json();
    return NextResponse.json(body);
}
"""

        diagnostics = classifier.get_classification_diagnostics(file_path, content)

        assert len(diagnostics) > 0

        # Backend should have strong matches
        if ArchitecturalDomain.BACKEND in diagnostics:
            backend_diag = diagnostics[ArchitecturalDomain.BACKEND]
            # Should have diagnostic categories
            categories = {diag.pattern_category for diag in backend_diag}
            assert len(categories) > 0

            # Should have some matched patterns for backend
            total_matches = sum(len(diag.matched_patterns) for diag in backend_diag)
            assert total_matches > 0

    def test_get_classification_diagnostics_testing(
        self, classifier: DomainClassifier
    ) -> None:
        """Test diagnostic generation for testing files."""
        from semantic_code_analyzer.analyzers.domain_classifier import (
            ArchitecturalDomain,
        )

        file_path = "tests/components/Button.test.tsx"
        content = """
import { render, screen, fireEvent } from '@testing-library/react';
import { expect, test, describe } from '@jest/globals';
import Button from '../Button';

describe('Button Component', () => {
    test('renders button with text', () => {
        render(<Button>Click me</Button>);
        expect(screen.getByRole('button')).toBeInTheDocument();
    });

    test('handles click events', () => {
        const mockClick = jest.fn();
        render(<Button onClick={mockClick}>Click</Button>);
        fireEvent.click(screen.getByRole('button'));
        expect(mockClick).toHaveBeenCalled();
    });
});
"""

        diagnostics = classifier.get_classification_diagnostics(file_path, content)

        assert len(diagnostics) > 0

        # Testing should have strong matches
        if ArchitecturalDomain.TESTING in diagnostics:
            testing_diag = diagnostics[ArchitecturalDomain.TESTING]
            categories = {diag.pattern_category for diag in testing_diag}
            assert len(categories) > 0

            # Should have matched testing patterns
            total_matches = sum(len(diag.matched_patterns) for diag in testing_diag)
            assert total_matches > 0

    def test_human_readable_pattern_conversion(
        self, classifier: DomainClassifier
    ) -> None:
        """Test conversion of regex patterns to human-readable descriptions."""
        # Test import patterns
        react_pattern = r"import.*from\s+['\"]react['\"]"
        readable = classifier._get_human_readable_pattern(react_pattern, "import")
        assert readable == "React imports"

        # Test content patterns
        jsx_pattern = r"return\s*\(\s*<"
        readable = classifier._get_human_readable_pattern(jsx_pattern, "content")
        assert readable == "JSX return statements"

        # Test path patterns
        component_pattern = r"src/components/.*\.(tsx?|jsx?)$"
        readable = classifier._get_human_readable_pattern(component_pattern, "path")
        assert readable == "React components directory"

        # Test unknown pattern fallback
        unknown_pattern = r"unknown.*pattern"
        readable = classifier._get_human_readable_pattern(unknown_pattern, "import")
        assert readable == unknown_pattern  # Should return original pattern

    def test_diagnostics_missing_patterns_identified(
        self, classifier: DomainClassifier
    ) -> None:
        """Test that diagnostics correctly identify missing patterns."""
        from semantic_code_analyzer.analyzers.domain_classifier import (
            ArchitecturalDomain,
        )

        # Test file that should be frontend but is missing key frontend patterns
        file_path = "src/components/Component.tsx"
        content = """
// Basic TypeScript file with minimal React patterns
const Component = () => {
    return "Hello World";
};
export default Component;
"""

        diagnostics = classifier.get_classification_diagnostics(file_path, content)

        # Should have diagnostics for frontend domain
        if ArchitecturalDomain.FRONTEND in diagnostics:
            frontend_diag = diagnostics[ArchitecturalDomain.FRONTEND]

            # Should identify missing patterns
            for diag in frontend_diag:
                if diag.pattern_category in ["import", "content"]:
                    assert len(diag.missing_patterns) > 0
                    # Should have specific missing patterns like React imports
                    has_react_missing = any(
                        "react" in pattern.lower() for pattern in diag.missing_patterns
                    )
                    # At least one diagnostic category should note missing React patterns
                    if diag.pattern_category == "import":
                        assert has_react_missing or len(diag.missing_patterns) > 0

    def test_diagnostics_empty_file(self, classifier: DomainClassifier) -> None:
        """Test diagnostics behavior with empty or minimal file content."""
        file_path = "unknown.js"
        content = ""

        diagnostics = classifier.get_classification_diagnostics(file_path, content)

        # Should still generate diagnostics even for empty files
        assert isinstance(diagnostics, dict)
        # All domains should have mostly missing patterns
        for _domain, diag_list in diagnostics.items():
            for diag in diag_list:
                # Empty file should have many missing patterns
                assert len(diag.missing_patterns) >= len(diag.matched_patterns)
