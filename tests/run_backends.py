"""Run the shared TopologicPy public API suite against installed backends.

Each available backend runs in a separate pytest subprocess so backend imports,
global state, and caches cannot leak between engines.  Missing backends are
reported and skipped rather than treated as test failures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


BACKENDS = {
    "pythonocc": {
        "class_name": "PythonOCCBackend",
        "probe_module": "OCC.Core.TopoDS",
    },
    "topologic_core": {
        "class_name": "TopologicCoreBackend",
        "probe_module": "topologic_core",
    },
}


def _locations():
    tests = Path(__file__).resolve().parent
    root = tests.parent
    src = root / "src"
    return tests, root, src


def _env(src: Path, backend: str):
    env = os.environ.copy()
    env["TOPOLOGICPY_CORE_BACKEND"] = backend
    if src.is_dir():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + current if current else "")
    return env


def _available(backend: str, spec: dict[str, str], src: Path):
    """Probe both the backend runtime and TopologicPy's backend selection."""
    code = (
        "import importlib; "
        f"importlib.import_module({spec['probe_module']!r}); "
        "from topologicpy.Core import Core; "
        "b=Core.ResetBackend(); "
        f"assert b.__class__.__name__ == {spec['class_name']!r}, "
        "f'active={b.__class__.__name__}'; "
        "print(b.__class__.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=_env(src, backend),
        text=True,
        capture_output=True,
    )
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0, detail


def main():
    parser = argparse.ArgumentParser(
        description="Run TopologicPy tests against every installed topology backend."
    )
    parser.add_argument(
        "--only",
        choices=tuple(BACKENDS),
        help="Test only this backend. Missing requested backends are an error.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Probe backends and exit without running pytest.",
    )
    args, pytest_args = parser.parse_known_args()

    tests, root, src = _locations()
    names = [args.only] if args.only else list(BACKENDS)

    available = []
    unavailable = []

    for name in names:
        ok, detail = _available(name, BACKENDS[name], src)
        if ok:
            available.append(name)
            print(f"AVAILABLE   {name}: {detail}")
        else:
            tail = detail.splitlines()[-1] if detail else "probe failed"
            unavailable.append(name)
            print(f"UNAVAILABLE {name}: {tail}")

    if args.list_backends:
        return 0 if available else 2

    if args.only and args.only not in available:
        print(f"Requested backend {args.only!r} is not available.")
        return 2

    if not available:
        print("No supported TopologicPy backend is available.")
        return 2

    failures = []
    for name in available:
        print("\n" + "=" * 80)
        print(f"PUBLIC API CONTRACT: {name}")
        print("=" * 80)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(tests),
            "--backend",
            name,
            *pytest_args,
        ]
        result = subprocess.run(cmd, cwd=root, env=_env(src, name))
        if result.returncode:
            failures.append((name, result.returncode))

    print("\n" + "=" * 80)
    if failures:
        for name, code in failures:
            print(f"FAILED: {name} (pytest exit code {code})")
        return 1

    if unavailable and not args.only:
        print("Skipped unavailable backends: " + ", ".join(unavailable))
    print("All available backends passed the shared public API suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
