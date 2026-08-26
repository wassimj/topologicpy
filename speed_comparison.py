#!/usr/bin/env python3

"""
Compare TopologicPy unit-test execution time using the TopologicCore and
PythonOCC backends.

Run from the root of the TopologicPy repository:

    python benchmark_backends.py

Optionally repeat each backend several times:

    python benchmark_backends.py --runs 3

Additional pytest arguments can be supplied after "--":

    python benchmark_backends.py -- -n auto
    python benchmark_backends.py --runs 3 -- -n auto
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


BACKENDS = [
    ("TopologicCore", "topologic_core"),
    ("PythonOCC", "pythonocc"),
]


def run_tests(
    name: str,
    backend: str,
    repo_root: Path,
    pytest_args: list[str],
    run_number: int,
    total_runs: int,
) -> tuple[float, int]:
    """
    Runs the complete test suite using the specified backend.

    Parameters
    ----------
    name : str
        The display name of the backend.
    backend : str
        The value assigned to TOPOLOGICPY_CORE_BACKEND.
    repo_root : pathlib.Path
        The root directory of the TopologicPy repository.
    pytest_args : list
        Additional arguments passed to pytest.
    run_number : int
        The current run number.
    total_runs : int
        The total number of runs for this backend.

    Returns
    -------
    tuple
        A tuple containing the elapsed wall-clock time in seconds and the
        pytest process return code.
    """

    env = os.environ.copy()
    env["TOPOLOGICPY_CORE_BACKEND"] = backend

    # Ensure the local source tree is used rather than an installed copy.
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else src_path + os.pathsep + existing_pythonpath
    )

    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        *pytest_args,
    ]

    print()
    print("=" * 80)
    print(f"{name}  |  Run {run_number}/{total_runs}")
    print("=" * 80)
    print(f"Backend : {backend}")
    print(f"Command : {' '.join(command)}")
    print()

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
    )

    elapsed = time.perf_counter() - start

    print()
    print(f"{name} completed in {elapsed:.3f} seconds.")
    print(f"pytest return code: {result.returncode}")

    return elapsed, result.returncode


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds.

    Parameters
    ----------
    seconds : float
        The input duration in seconds.

    Returns
    -------
    str
        The formatted duration.
    """

    minutes, seconds = divmod(seconds, 60.0)

    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.2f}s"

    return f"{seconds:.2f}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full TopologicPy pytest execution time using "
            "TopologicCore and PythonOCC."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of complete test-suite runs per backend. Default is 1.",
    )

    args, pytest_args = parser.parse_known_args()

    # Allow the conventional:
    #
    #     benchmark_backends.py -- -n auto
    #
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    runs = max(1, args.runs)

    repo_root = Path(__file__).resolve().parent

    if not (repo_root / "src" / "topologicpy").is_dir():
        print(
            "Error: This script must be located in the root of the "
            "TopologicPy repository."
        )
        return 1

    results: dict[str, list[float]] = {
        name: [] for name, _ in BACKENDS
    }

    failures: dict[str, int] = {
        name: 0 for name, _ in BACKENDS
    }

    overall_start = time.perf_counter()

    # Alternate backend order between repetitions to reduce systematic
    # first-run / filesystem-cache bias.
    for run_index in range(runs):
        backends = (
            BACKENDS
            if run_index % 2 == 0
            else list(reversed(BACKENDS))
        )

        for name, backend in backends:
            elapsed, return_code = run_tests(
                name=name,
                backend=backend,
                repo_root=repo_root,
                pytest_args=pytest_args,
                run_number=run_index + 1,
                total_runs=runs,
            )

            results[name].append(elapsed)

            if return_code != 0:
                failures[name] += 1

    overall_elapsed = time.perf_counter() - overall_start

    print()
    print()
    print("=" * 80)
    print("BACKEND PERFORMANCE COMPARISON")
    print("=" * 80)
    print()

    for name, _ in BACKENDS:
        times = results[name]

        print(f"{name}")
        print("-" * 40)

        for i, elapsed in enumerate(times, start=1):
            print(f"Run {i:<3}: {format_time(elapsed)}")

        print()

        if len(times) == 1:
            print(f"Time   : {format_time(times[0])}")
        else:
            print(f"Mean   : {format_time(statistics.mean(times))}")
            print(f"Median : {format_time(statistics.median(times))}")
            print(f"Min    : {format_time(min(times))}")
            print(f"Max    : {format_time(max(times))}")

        if failures[name]:
            print(f"FAILED : {failures[name]} run(s)")
        else:
            print("Tests  : ALL PASSED")

        print()

    topologic_core_time = statistics.median(results["TopologicCore"])
    pythonocc_time = statistics.median(results["PythonOCC"])

    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"TopologicCore : {format_time(topologic_core_time)}"
    )
    print(
        f"PythonOCC     : {format_time(pythonocc_time)}"
    )

    difference = pythonocc_time - topologic_core_time

    if topologic_core_time > 0:
        ratio = pythonocc_time / topologic_core_time
    else:
        ratio = float("inf")

    print()

    if difference > 0:
        percent = (
            difference / topologic_core_time * 100.0
            if topologic_core_time > 0
            else float("inf")
        )

        print(
            f"PythonOCC is {format_time(difference)} slower "
            f"({percent:.1f}% slower)."
        )
        print(
            f"PythonOCC / TopologicCore ratio: {ratio:.3f}x"
        )

    elif difference < 0:
        difference = abs(difference)

        percent = (
            difference / topologic_core_time * 100.0
            if topologic_core_time > 0
            else 0.0
        )

        print(
            f"PythonOCC is {format_time(difference)} faster "
            f"({percent:.1f}% faster)."
        )
        print(
            f"PythonOCC / TopologicCore ratio: {ratio:.3f}x"
        )

    else:
        print("Both backends completed in the same time.")

    print()
    print(f"Total benchmark time: {format_time(overall_elapsed)}")

    print()

    if any(failures.values()):
        print(
            "WARNING: At least one test run failed. "
            "Performance results should not be considered valid."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())