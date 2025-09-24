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

"""
Pytest configuration and shared fixtures for the test suite.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import git
import numpy as np
import pytest

from semantic_code_analyzer import EnhancedScorerConfig

# Set environment variables for safe testing before any imports
os.environ["SCA_DISABLE_MODEL_LOADING"] = "1"
os.environ["SCA_TEST_MODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Set random seeds for reproducible tests
np.random.seed(42)


@pytest.fixture(scope="session")
def test_repo() -> Any:
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Initialize git repository
        repo = git.Repo.init(repo_path)

        # Configure git user for testing
        with repo.config_writer() as git_config:
            git_config.set_value("user", "name", "Test User")
            git_config.set_value("user", "email", "test@example.com")

        # Create initial commit
        test_file = repo_path / "README.md"
        test_file.write_text("# Test Repository\n\nThis is a test repository.")

        repo.index.add([str(test_file)])
        initial_commit = repo.index.commit("Initial commit")

        yield {"repo": repo, "repo_path": repo_path, "initial_commit": initial_commit}


@pytest.fixture
def sample_typescript_files() -> dict[str, str]:
    """Sample TypeScript files for testing analyzers."""
    return {
        "src/components/GoodComponent.tsx": """
'use client';

import React, { useState, useCallback } from 'react';

interface GoodComponentProps {
    title: string;
    onUpdate?: (value: string) => void;
}

const GoodComponent: React.FC<GoodComponentProps> = ({ title, onUpdate }) => {
    const [value, setValue] = useState<string>('');

    const handleChange = useCallback((newValue: string) => {
        setValue(newValue);
        onUpdate?.(newValue);
    }, [onUpdate]);

    return (
        <div role="region" aria-labelledby="component-title">
            <h2 id="component-title">{title}</h2>
            <input
                type="text"
                value={value}
                onChange={(e) => handleChange(e.target.value)}
                aria-label="Text input"
            />
        </div>
    );
};

export default GoodComponent;
""",
        "src/app/api/test/route.ts": """
import { NextRequest, NextResponse } from 'next/server';

interface TestResponse {
    message: string;
    timestamp: Date;
}

export async function GET(request: NextRequest): Promise<NextResponse> {
    try {
        const response: TestResponse = {
            message: 'Hello World',
            timestamp: new Date()
        };

        return NextResponse.json(response, { status: 200 });
    } catch (error) {
        console.error('Error in GET handler:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
""",
        "src/app/[locale]/layout.tsx": """
import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
    title: "Test App",
    description: "Test application for analysis"
};

interface LayoutProps {
    children: React.ReactNode;
    params: { locale: string };
}

export default function Layout({ children, params: { locale } }: LayoutProps): React.ReactElement {
    return (
        <html lang={locale}>
            <body>
                <main role="main">
                    {children}
                </main>
            </body>
        </html>
    );
}
""",
    }


@pytest.fixture
def sample_poor_quality_files() -> dict[str, str]:
    """Sample poor quality files for testing."""
    return {
        "components/poor.js": """
import React from 'react';

function Thing(props) {
    const [stuff, setStuff] = React.useState();

    function doThing() {
        fetch('/api/thing').then(res => res.json()).then(data => {
            setStuff(data);
        }).catch(err => {
            console.log(err);
        });
    }

    return (
        <div onClick={doThing}>
            <h1>{props.title}</h1>
            <div>{stuff && stuff.name}</div>
        </div>
    );
}

export default Thing;
""",
        "api/route.js": """
export default function handler(req, res) {
    if (req.method === 'GET') {
        res.status(200).json({ data: 'test' });
    }
}
""",
    }


@pytest.fixture
def enhanced_scorer_config() -> EnhancedScorerConfig:
    """Default enhanced scorer configuration for testing."""
    import os

    # Disable domain adherence during tests to prevent model loading segfaults
    disable_models = os.getenv("SCA_DISABLE_MODEL_LOADING", "0") == "1"

    if disable_models:
        return EnhancedScorerConfig(
            architectural_weight=0.30,
            quality_weight=0.30,
            typescript_weight=0.25,
            framework_weight=0.15,
            domain_adherence_weight=0.0,
            enable_domain_adherence_analysis=False,
            include_actionable_feedback=True,
            include_pattern_details=True,
            max_recommendations_per_file=5,
        )
    else:
        return EnhancedScorerConfig(
            architectural_weight=0.25,
            quality_weight=0.25,
            typescript_weight=0.20,
            framework_weight=0.15,
            domain_adherence_weight=0.15,
            enable_domain_adherence_analysis=True,
            include_actionable_feedback=True,
            include_pattern_details=True,
            max_recommendations_per_file=5,
        )


@pytest.fixture
def enhanced_scorer_config_with_models() -> EnhancedScorerConfig:
    """Enhanced scorer configuration with model loading enabled for specific tests."""

    return EnhancedScorerConfig(
        architectural_weight=0.25,
        quality_weight=0.25,
        typescript_weight=0.20,
        framework_weight=0.15,
        domain_adherence_weight=0.15,
        enable_domain_adherence_analysis=True,
        include_actionable_feedback=True,
        include_pattern_details=True,
        max_recommendations_per_file=5,
    )


@pytest.fixture
def mock_scorer(enhanced_scorer_config: EnhancedScorerConfig) -> Any:
    """Mock MultiDimensionalScorer for testing without git dependency."""

    from semantic_code_analyzer import MultiDimensionalScorer

    # Mock the git repo to avoid requiring actual git repository
    with patch("git.Repo") as mock_repo:
        mock_repo.return_value = Mock()
        scorer = MultiDimensionalScorer(enhanced_scorer_config, repo_path=".")
        yield scorer
