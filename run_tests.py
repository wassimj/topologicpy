#!/usr/bin/env python
"""
Local TopologicPy test runner for src-layout repository.

Expected layout:

    GitHub/
    └── topologicpy/
        ├── src/
        │   └── topologicpy/
        │       ├── __init__.py
        │       ├── Vertex.py
        │       └── ...
        ├── tests/
        └── run_tests.py

Examples:

    python run_tests.py
    python run_tests.py -v --failfast
    python run_tests.py --test Vertex
    python run_tests.py --test Edge
    python run_tests.py --test test_01Vertex.py
    python run_tests.py --test tests/test_01Vertex.py
    python run_tests.py --test test_01Vertex.py::test_main
    python run_tests.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_matching_test(tests_dir: Path, test_name: str) -> str:
    """
    Resolve a flexible test selector to a pytest target.

    Accepted examples:
        Vertex
        test_01Vertex.py
        tests/test_01Vertex.py
        test_01Vertex.py::test_main
        tests/test_01Vertex.py::test_main
    """

    test_name = test_name.strip().replace("\\", "/")

    if not test_name:
        return str(tests_dir)

    # Preserve pytest node selectors, e.g. test_01Vertex.py::test_main
    if "::" in test_name:
        file_part, node_part = test_name.split("::", 1)
        resolved_file = _find_matching_test(tests_dir, file_part)
        return f"{resolved_file}::{node_part}"

    candidate = Path(test_name)

    # Direct path supplied.
    if candidate.exists():
        return str(candidate)

    # Path relative to repository root, e.g. tests/test_01Vertex.py
    repo_relative = Path.cwd() / candidate
    if repo_relative.exists():
        return str(repo_relative)

    # Path relative to tests folder, e.g. test_01Vertex.py
    tests_relative = tests_dir / test_name
    if tests_relative.exists():
        return str(tests_relative)

    # If user typed Vertex, match test files containing Vertex.
    lowered = test_name.lower()

    if not lowered.endswith(".py"):
        matches = sorted(
            p for p in tests_dir.glob("test_*.py")
            if lowered in p.stem.lower()
        )
    else:
        matches = sorted(
            p for p in tests_dir.glob(test_name)
        )

    if len(matches) == 1:
        return str(matches[0])

    if len(matches) > 1:
        print()
        print(f"Ambiguous test selector: {test_name}")
        print("Matching files:")
        for p in matches:
            print(f"  {p.name}")
        print()
        raise SystemExit(1)

    print()
    print(f"Could not find a matching test for: {test_name}")
    print()
    print("Available test files:")
    for p in sorted(tests_dir.glob("test_*.py")):
        print(f"  {p.name}")
    print()
    raise SystemExit(1)


def _list_tests(tests_dir: Path) -> None:
    print("Available test files:")
    print()
    for p in sorted(tests_dir.glob("test_*.py")):
        print(f"  {p.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TopologicPy unit tests locally before pushing to GitHub."
    )

    parser.add_argument(
        "--tests",
        default="tests",
        help="Path to the tests folder. Default: tests",
    )

    parser.add_argument(
        "--test",
        default=None,
        help=(
            "Specific test file, partial name, or pytest node to run. "
            "Examples: Vertex, test_01Vertex.py, tests/test_01Vertex.py, "
            "test_01Vertex.py::test_main"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test files and exit.",
    )

    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help='Test filename pattern. Default: "test_*.py"',
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run tests in verbose mode.",
    )

    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop after the first failure.",
    )

    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Capture printed output instead of showing it live.",
    )

    parser.add_argument(
        "--disable-warnings",
        action="store_true",
        help="Suppress pytest warning summary.",
    )

    parser.add_argument(
        "--use-project-config",
        action="store_true",
        help=(
            "Use pytest options from pyproject.toml. "
            "By default, this script ignores project addopts such as -nauto."
        ),
    )

    parser.add_argument(
        "--xdist",
        action="store_true",
        help=(
            "Run tests in parallel using pytest-xdist. "
            "Requires: pip install pytest-xdist"
        ),
    )

    args = parser.parse_args()

    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest is not installed.")
        print("Install it with:")
        print()
        print("    pip install pytest")
        print()
        return 1

    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    package_dir = src_dir / "topologicpy"
    tests_dir = root / args.tests

    if not src_dir.exists():
        print(f"Could not find src folder: {src_dir}")
        return 1

    if not package_dir.exists():
        print(f"Could not find topologicpy package folder: {package_dir}")
        return 1

    if not tests_dir.exists():
        print(f"Could not find tests folder: {tests_dir}")
        return 1

    if args.list:
        _list_tests(tests_dir)
        return 0

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    pytest_target = (
        _find_matching_test(tests_dir, args.test)
        if args.test
        else str(tests_dir)
    )

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        pytest_target,
        "--tb=short",
        "-o",
        f"python_files={args.pattern}",
    ]

    # Ignore problematic pytest addopts from pyproject.toml by default.
    # This avoids errors from options such as -nauto when pytest-xdist is absent.
    if not args.use_project_config:
        cmd.extend(["-o", "addopts="])

    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if args.failfast:
        cmd.append("-x")

    if not args.capture_output:
        cmd.append("-s")

    if args.disable_warnings:
        cmd.append("--disable-warnings")

    if args.xdist:
        cmd.extend(["-n", "auto"])

    print("Running TopologicPy tests")
    print(f"Repository: {root}")
    print(f"Source:     {src_dir}")
    print(f"Package:    {package_dir}")
    print(f"Tests:      {tests_dir}")
    print(f"Target:     {pytest_target}")
    print(f"Pattern:    {args.pattern}")
    print()

    result = subprocess.run(cmd, cwd=root, env=env)

    print()
    if result.returncode == 0:
        print("All tests passed.")
    else:
        print("Some tests failed.")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())