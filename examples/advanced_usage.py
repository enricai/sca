#!/usr/bin/env python3
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
Advanced usage examples for the Multi-Dimensional Code Analyzer.

This script demonstrates advanced features including custom configurations,
commit comparison, detailed pattern analysis, and integration patterns.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add the package to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_code_analyzer import EnhancedScorerConfig, MultiDimensionalScorer


def example_commit_comparison() -> None:
    """Advanced example of comparing multiple commits."""
    print("🔍 Advanced Commit Comparison Example")
    print("=" * 50)

    # Configuration optimized for comparison
    config = EnhancedScorerConfig(
        architectural_weight=0.25,
        quality_weight=0.35,  # Emphasize quality differences
        typescript_weight=0.25,
        framework_weight=0.15,
        include_pattern_details=True,
        max_recommendations_per_file=5,
    )

    scorer = MultiDimensionalScorer(config, repo_path=".")

    # Get recent commits for comparison
    try:
        import git

        repo = git.Repo(".")
        commits = [str(commit) for commit in repo.iter_commits(max_count=3)]

        if len(commits) >= 2:
            base_commit = commits[0]  # Latest commit
            compare_commits = commits[1:3]  # Previous commits

            print("🔍 Comparing commits:")
            print(f"   Base: {base_commit[:8]}")
            for i, commit in enumerate(compare_commits, 1):
                print(f"   Compare {i}: {commit[:8]}")

            # Perform comparison
            comparison_results = scorer.compare_commits(base_commit, compare_commits)

            # Display results
            print("\n📊 Comparison Results:")
            summary = comparison_results.get("comparison_summary", {})

            base_score = summary.get("base_score", 0)
            print(f"   Base score: {base_score:.3f}")

            comparisons = summary.get("comparisons", [])
            for comp in comparisons:
                commit_id = comp["commit"][:8]
                score = comp["score"]
                improvement = comp["improvement"]
                percentage = comp["improvement_percentage"]

                print(
                    f"   {commit_id}: {score:.3f} (base +{improvement:.3f}, {percentage:.1f}% better)"
                )

                # Show dimensional improvements
                dim_improvements = comp.get("dimensional_improvements", {})
                if dim_improvements:
                    print("      Dimensional improvements:")
                    for dim, improvement in dim_improvements.items():
                        print(f"        {dim}: +{improvement:.1f}%")

        else:
            print("⚠️  Not enough commits for comparison")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def example_custom_configurations() -> None:
    """Example of different configuration strategies."""
    print("\n\n⚖️  Custom Configuration Strategies")
    print("=" * 50)

    configurations = [
        {
            "name": "Architecture Focus",
            "description": "Emphasizes file structure and imports",
            "config": EnhancedScorerConfig(
                architectural_weight=0.50,
                quality_weight=0.20,
                typescript_weight=0.20,
                framework_weight=0.10,
            ),
        },
        {
            "name": "Quality Focus",
            "description": "Emphasizes best practices and security",
            "config": EnhancedScorerConfig(
                architectural_weight=0.15,
                quality_weight=0.50,
                typescript_weight=0.25,
                framework_weight=0.10,
            ),
        },
        {
            "name": "TypeScript Focus",
            "description": "Emphasizes type safety and TS patterns",
            "config": EnhancedScorerConfig(
                architectural_weight=0.20,
                quality_weight=0.20,
                typescript_weight=0.50,
                framework_weight=0.10,
            ),
        },
        {
            "name": "Framework Focus",
            "description": "Emphasizes Next.js and React patterns",
            "config": EnhancedScorerConfig(
                architectural_weight=0.20,
                quality_weight=0.20,
                typescript_weight=0.15,
                framework_weight=0.45,
            ),
        },
    ]

    # Test file with various patterns
    test_file = {
        "src/app/api/users/route.ts": """
import { NextRequest, NextResponse } from 'next/server';

interface User {
    id: string;
    name: string;
    email: string;
}

export async function GET(request: NextRequest): Promise<NextResponse> {
    try {
        const users: User[] = [
            { id: '1', name: 'John', email: 'john@example.com' },
            { id: '2', name: 'Jane', email: 'jane@example.com' }
        ];

        return NextResponse.json(users, { status: 200 });
    } catch (error) {
        console.error('Error fetching users:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}

export async function POST(request: NextRequest): Promise<NextResponse> {
    try {
        const body = await request.json();

        if (!body.name || !body.email) {
            return NextResponse.json(
                { error: 'Name and email are required' },
                { status: 400 }
            );
        }

        const newUser: User = {
            id: Math.random().toString(36),
            name: body.name,
            email: body.email
        };

        return NextResponse.json(newUser, { status: 201 });
    } catch (error) {
        console.error('Error creating user:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
"""
    }

    print("📊 Comparing different focus configurations:")
    print(
        f"{'Configuration':<18} {'Overall':<10} {'Arch':<8} {'Quality':<8} {'TS':<8} {'Framework':<10}"
    )
    print("-" * 72)

    for config_info in configurations:
        config_name = config_info["name"]
        config = config_info["config"]
        if not isinstance(config, EnhancedScorerConfig):
            raise TypeError(f"Expected EnhancedScorerConfig, got {type(config)}")

        scorer = MultiDimensionalScorer(config, repo_path=".")
        results = scorer.analyze_files(test_file)

        overall = results["overall_adherence"]
        dims = results.get("dimensional_scores", {})

        print(
            f"{config_name:<18} {overall:<10.3f} {dims.get('architectural', 0):<8.3f} "
            f"{dims.get('quality', 0):<8.3f} {dims.get('typescript', 0):<8.3f} "
            f"{dims.get('framework', 0):<8.3f}"
        )

    print("\n💡 Key insight: Weight configuration significantly impacts scoring")
    print("   Choose weights based on your project's priorities and standards.")


def example_pattern_analysis() -> None:
    """Example of detailed pattern analysis and insights."""
    print("\n\n🔬 Detailed Pattern Analysis Example")
    print("=" * 50)

    # Enable all detailed options
    config = EnhancedScorerConfig(
        include_pattern_details=True,
        include_actionable_feedback=True,
        max_recommendations_per_file=15,
    )

    # Complex example showcasing many patterns
    complex_files = {
        "src/components/UserDashboard.tsx": """
'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';

interface User {
    id: string;
    name: string;
    email: string;
    role: 'admin' | 'user';
    avatar?: string;
}

interface UserDashboardProps {
    initialUser?: User;
    onUserUpdate?: (user: User) => void;
}

const UserDashboard: React.FC<UserDashboardProps> = ({ initialUser, onUserUpdate }) => {
    const [user, setUser] = useState<User | null>(initialUser || null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const router = useRouter();

    // Fetch user data
    const fetchUser = useCallback(async (userId: string) => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch(`/api/users/${userId}`);

            if (!response.ok) {
                throw new Error('Failed to fetch user');
            }

            const userData = await response.json();
            setUser(userData);
            onUserUpdate?.(userData);

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    }, [onUserUpdate]);

    // Memoized computed values
    const displayName = useMemo(() => {
        return user?.name || 'Unknown User';
    }, [user?.name]);

    const isAdmin = useMemo(() => {
        return user?.role === 'admin';
    }, [user?.role]);

    // Handle navigation
    const handleProfileEdit = useCallback(() => {
        if (user) {
            router.push(`/users/${user.id}/edit`);
        }
    }, [user, router]);

    // Loading state
    if (loading) {
        return (
            <div role="status" aria-label="Loading user dashboard">
                <div className="loading-spinner" />
                <span className="sr-only">Loading...</span>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div role="alert" className="error-container">
                <h2>Error Loading Dashboard</h2>
                <p>{error}</p>
                <button onClick={() => setError(null)}>
                    Try Again
                </button>
            </div>
        );
    }

    // No user state
    if (!user) {
        return (
            <div className="no-user">
                <h2>No User Data</h2>
                <p>Please log in to view your dashboard.</p>
            </div>
        );
    }

    return (
        <main role="main" className="user-dashboard">
            <header className="dashboard-header">
                <div className="user-info">
                    {user.avatar && (
                        <img
                            src={user.avatar}
                            alt={`Profile picture for ${displayName}`}
                            className="user-avatar"
                            loading="lazy"
                        />
                    )}
                    <div>
                        <h1>{displayName}</h1>
                        <p className="user-email">{user.email}</p>
                        {isAdmin && (
                            <span className="admin-badge" role="status">
                                Administrator
                            </span>
                        )}
                    </div>
                </div>

                <nav className="dashboard-actions" role="navigation">
                    <button
                        onClick={handleProfileEdit}
                        className="edit-profile-btn"
                        aria-label={`Edit profile for ${displayName}`}
                    >
                        Edit Profile
                    </button>
                </nav>
            </header>

            <section className="dashboard-content">
                <h2>Dashboard Overview</h2>
                <p>Welcome to your dashboard, {displayName}!</p>
            </section>
        </main>
    );
};

export default UserDashboard;
""",
        "src/app/[locale]/layout.tsx": """
import type { Metadata } from "next";
import { NextIntlClientProvider } from "next-intl";
import { notFound } from "next/navigation";
import React from "react";

export const metadata: Metadata = {
    title: {
        template: '%s | MyApp',
        default: 'MyApp - User Management'
    },
    description: 'Advanced user management application',
    keywords: ['users', 'management', 'dashboard'],
    authors: [{ name: 'Development Team' }],
    openGraph: {
        title: 'MyApp User Management',
        description: 'Advanced user management application',
        type: 'website'
    }
};

interface RootLayoutProps {
    children: React.ReactNode;
    params: { locale: string };
}

const locales = ['en', 'es', 'fr'];

export default async function RootLayout({
    children,
    params: { locale }
}: RootLayoutProps): Promise<React.ReactElement> {
    // Validate locale
    if (!locales.includes(locale)) {
        notFound();
    }

    let messages;
    try {
        messages = (await import(`../../messages/${locale}.json`)).default;
    } catch (error) {
        console.error(`Failed to load messages for locale ${locale}:`, error);
        notFound();
    }

    return (
        <html lang={locale}>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </head>
            <body>
                <NextIntlClientProvider
                    locale={locale}
                    messages={messages}
                    timeZone="UTC"
                >
                    <div id="app-root">
                        {children}
                    </div>
                </NextIntlClientProvider>
            </body>
        </html>
    );
}

export function generateStaticParams() {
    return locales.map((locale) => ({ locale }));
}
""",
    }

    scorer = MultiDimensionalScorer(config, repo_path=".")
    results = scorer.analyze_files(complex_files)

    print("📊 Detailed Analysis Results:")
    print(f"   Overall adherence: {results['overall_adherence']:.3f}")
    print(f"   Confidence: {results['confidence']:.3f}")

    # Show pattern breakdown
    pattern_analysis = results.get("pattern_analysis", {})
    if pattern_analysis:
        print("\n🔍 Pattern Analysis:")
        print(f"   Total patterns: {pattern_analysis.get('total_patterns_found', 0)}")
        print(
            f"   Avg confidence: {pattern_analysis.get('pattern_confidence_avg', 0):.3f}"
        )

        patterns_by_type = pattern_analysis.get("patterns_by_type", {})
        if patterns_by_type:
            print("   Patterns by type:")
            for pattern_type, count in patterns_by_type.items():
                print(f"      {pattern_type}: {count}")

    # Show detailed recommendations
    feedback = results.get("actionable_feedback", [])
    if feedback:
        print(f"\n💡 Actionable Feedback ({len(feedback)} items):")

        # Group by severity
        by_severity: dict[str, list[dict[str, Any]]] = {}
        for rec in feedback:
            severity = rec["severity"]
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(rec)

        for severity in ["critical", "error", "warning", "info"]:
            if severity in by_severity:
                print(f"   {severity.upper()} ({len(by_severity[severity])}):")
                for rec in by_severity[severity][:3]:  # Show top 3 per severity
                    print(f"      • {rec['message']}")
                    if rec.get("suggested_fix"):
                        print(f"        Fix: {rec['suggested_fix']}")


def example_configuration_tuning() -> None:
    """Example of tuning configuration for specific project needs."""
    print("\n\n⚙️  Configuration Tuning Example")
    print("=" * 50)

    # Different project types need different configurations
    project_configs = {
        "Frontend Heavy": EnhancedScorerConfig(
            architectural_weight=0.25,
            quality_weight=0.25,
            typescript_weight=0.25,
            framework_weight=0.25,  # Equal emphasis on React/Next.js
        ),
        "API Heavy": EnhancedScorerConfig(
            architectural_weight=0.35,  # API structure important
            quality_weight=0.35,  # Security/error handling critical
            typescript_weight=0.25,  # Type safety for APIs
            framework_weight=0.05,  # Less framework-specific patterns
        ),
        "Type-Safe Focus": EnhancedScorerConfig(
            architectural_weight=0.20,
            quality_weight=0.25,
            typescript_weight=0.45,  # Heavy emphasis on TypeScript
            framework_weight=0.10,
        ),
        "Prototype/MVP": EnhancedScorerConfig(
            architectural_weight=0.40,  # Structure important for growth
            quality_weight=0.40,  # Basic quality standards
            typescript_weight=0.15,  # Less strict typing for speed
            framework_weight=0.05,  # Basic framework usage
        ),
    }

    # Test with a typical component
    test_component = {
        "src/components/UserCard.tsx": """
'use client';

import React from 'react';

interface UserCardProps {
    user: {
        id: string;
        name: string;
        email: string;
        avatar?: string;
    };
    onEdit?: () => void;
}

const UserCard: React.FC<UserCardProps> = ({ user, onEdit }) => {
    return (
        <div className="user-card">
            <div className="user-info">
                <h3>{user.name}</h3>
                <p>{user.email}</p>
            </div>
            {onEdit && (
                <button onClick={onEdit} className="edit-btn">
                    Edit
                </button>
            )}
        </div>
    );
};

export default UserCard;
"""
    }

    print("📊 Configuration comparison for the same code:")
    print(
        f"{'Project Type':<15} {'Overall':<10} {'Arch':<8} {'Quality':<8} {'TS':<8} {'Framework':<10}"
    )
    print("-" * 70)

    for project_type, config in project_configs.items():
        scorer = MultiDimensionalScorer(config, repo_path=".")
        results = scorer.analyze_files(test_component)

        overall = results["overall_adherence"]
        dims = results.get("dimensional_scores", {})

        print(
            f"{project_type:<15} {overall:<10.3f} {dims.get('architectural', 0):<8.3f} "
            f"{dims.get('quality', 0):<8.3f} {dims.get('typescript', 0):<8.3f} "
            f"{dims.get('framework', 0):<8.3f}"
        )

    print("\n🎯 Insight: Same code scores differently based on project priorities")


def example_ci_cd_integration() -> None:
    """Example of integrating with CI/CD pipelines."""
    print("\n\n🔗 CI/CD Integration Example")
    print("=" * 50)

    def quality_gate_check(commit_hash: str, min_score: float = 0.7) -> dict[str, Any]:
        """
        Quality gate check for CI/CD pipeline.

        Args:
            commit_hash: Commit to check
            min_score: Minimum acceptable score

        Returns:
            Check results with pass/fail status
        """
        config = EnhancedScorerConfig(
            architectural_weight=0.30,
            quality_weight=0.40,  # Emphasize quality for gates
            typescript_weight=0.25,
            framework_weight=0.05,
            max_recommendations_per_file=3,  # Limit for CI output
        )

        try:
            scorer = MultiDimensionalScorer(config, repo_path=".")
            results = scorer.analyze_commit(commit_hash)

            overall_score = results["overall_adherence"]
            passed = overall_score >= min_score

            return {
                "commit_hash": commit_hash,
                "overall_score": overall_score,
                "min_score": min_score,
                "passed": passed,
                "dimensional_scores": results.get("dimensional_scores", {}),
                "confidence": results.get("confidence", 0),
                "critical_issues": [
                    rec
                    for rec in results.get("actionable_feedback", [])
                    if rec["severity"] in ["critical", "error"]
                ],
                "files_analyzed": len(results.get("file_level_analysis", {})),
                "processing_time": results.get("processing_time", 0),
            }

        except Exception as e:
            return {"commit_hash": commit_hash, "error": str(e), "passed": False}

    # Example usage in CI/CD
    try:
        import git

        repo = git.Repo(".")
        latest_commit = str(repo.head.commit)

        print(f"🔍 Running quality gate for commit: {latest_commit[:8]}")

        check_result = quality_gate_check(latest_commit, min_score=0.7)

        print("\n📊 Quality Gate Results:")
        if "error" in check_result:
            print(f"   ❌ ERROR: {check_result['error']}")
            print("   Exit code: 1")
        else:
            status = "✅ PASSED" if check_result["passed"] else "❌ FAILED"
            print(f"   Status: {status}")
            print(f"   Overall Score: {check_result['overall_score']:.3f}")
            print(f"   Required Score: {check_result['min_score']}")
            print(f"   Confidence: {check_result['confidence']:.3f}")

            # Show dimensional breakdown
            dims = check_result["dimensional_scores"]
            print("   Dimensional Scores:")
            for dim, score in dims.items():
                print(f"      {dim}: {score:.3f}")

            # Show critical issues
            critical_issues = check_result["critical_issues"]
            if critical_issues:
                print(f"   ⚠️  Critical Issues ({len(critical_issues)}):")
                for issue in critical_issues:
                    print(f"      • {issue['message']}")

            exit_code = 0 if check_result["passed"] else 1
            print(f"   Exit code: {exit_code}")

        # Save results for CI/CD system
        ci_results = {
            "quality_gate": check_result,
            "pipeline_metadata": {
                "timestamp": time.time(),
                "commit": latest_commit,
                "repository": str(Path.cwd()),
            },
        }

        with open("quality_gate_results.json", "w") as f:
            json.dump(ci_results, f, indent=2, default=str)

        print("   Results saved to: quality_gate_results.json")

    except Exception as e:
        print(f"❌ CI/CD integration failed: {e}")


def example_batch_analysis() -> None:
    """Example of analyzing multiple commits in batch."""
    print("\n\n📦 Batch Analysis Example")
    print("=" * 50)

    config = EnhancedScorerConfig(
        include_actionable_feedback=False,  # Reduce output for batch
        include_pattern_details=False,
    )

    scorer = MultiDimensionalScorer(config, repo_path=".")

    try:
        import git

        repo = git.Repo(".")
        commits = [str(commit) for commit in repo.iter_commits(max_count=5)]

        print(f"📊 Batch analyzing {len(commits)} commits:")

        batch_results = []
        for i, commit_hash in enumerate(commits, 1):
            try:
                print(f"   {i}/{len(commits)}: {commit_hash[:8]}...", end=" ")

                start_time = time.time()
                result = scorer.analyze_commit(commit_hash)
                analysis_time = time.time() - start_time

                batch_results.append(
                    {
                        "commit": commit_hash,
                        "overall_score": result["overall_adherence"],
                        "dimensional_scores": result["dimensional_scores"],
                        "analysis_time": analysis_time,
                        "patterns_found": result.get("pattern_analysis", {}).get(
                            "total_patterns_found", 0
                        ),
                    }
                )

                print(f"✅ {result['overall_adherence']:.3f} ({analysis_time:.1f}s)")

            except Exception as e:
                print(f"❌ Failed: {e}")

        # Analyze batch results
        if batch_results:
            print("\n📈 Batch Analysis Summary:")

            overall_scores = [r["overall_score"] for r in batch_results]
            avg_score = sum(overall_scores) / len(overall_scores)
            max_score = max(overall_scores)
            min_score = min(overall_scores)

            print(f"   Average score: {avg_score:.3f}")
            print(f"   Score range: {min_score:.3f} - {max_score:.3f}")

            # Find best and worst commits
            best_commit = max(batch_results, key=lambda r: r["overall_score"])
            worst_commit = min(batch_results, key=lambda r: r["overall_score"])

            print(
                f"   Best commit: {best_commit['commit'][:8]} ({best_commit['overall_score']:.3f})"
            )
            print(
                f"   Worst commit: {worst_commit['commit'][:8]} ({worst_commit['overall_score']:.3f})"
            )

            # Save batch results
            with open("batch_analysis_results.json", "w") as f:
                json.dump(
                    {
                        "summary": {
                            "commits_analyzed": len(batch_results),
                            "average_score": avg_score,
                            "score_range": [min_score, max_score],
                        },
                        "detailed_results": batch_results,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            print("   Results saved to: batch_analysis_results.json")

    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")


if __name__ == "__main__":
    try:
        example_commit_comparison()
        example_custom_configurations()
        example_pattern_analysis()
        example_configuration_tuning()
        example_ci_cd_integration()
        example_batch_analysis()

        print("\n🎉 All advanced examples completed successfully!")
        print("\n🚀 Advanced Features Demonstrated:")
        print("   ✅ Multi-commit comparison")
        print("   ✅ Custom configuration strategies")
        print("   ✅ Detailed pattern analysis")
        print("   ✅ CI/CD quality gates")
        print("   ✅ Batch processing")
        print("   ✅ Project-specific tuning")

    except Exception as e:
        print(f"❌ Advanced example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
