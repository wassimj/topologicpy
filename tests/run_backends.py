"""Run the same TopologicPy public API suite against every available backend.

Each backend runs in a separate pytest subprocess.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


BACKENDS = {
    "pythonocc": "PythonOCCBackend",
    "topologic_core": "TopologicCoreBackend",
}


def _locations():
    tests = Path(__file__).resolve().parent
    root = tests.parent
    src = root / "src"
    return tests, root, src


def _env(src, backend):
    env = os.environ.copy()
    env["TOPOLOGICPY_CORE_BACKEND"] = backend
    if src.is_dir():
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + current if current else "")
    return env


def _available(backend, expected_class, src):
    code = (
        "from topologicpy.Core import Core; "
        "b=Core.Backend(); "
        f"assert b.__class__.__name__ == {expected_class!r}, "
        "f'active={b.__class__.__name__}'; "
        "print(b.__class__.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=_env(src, backend),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=tuple(BACKENDS))
    parser.add_argument("--list-backends", action="store_true")
    args, pytest_args = parser.parse_known_args()

    tests, root, src = _locations()
    names = [args.only] if args.only else list(BACKENDS)

    available = []
    for name in names:
        ok, detail = _available(name, BACKENDS[name], src)
        if ok:
            available.append(name)
            print(f"AVAILABLE   {name}: {detail}")
        else:
            tail = detail.splitlines()[-1] if detail else "probe failed"
            print(f"UNAVAILABLE {name}: {tail}")

    if args.list_backends:
        return 0 if available else 2
    if not available:
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

    print("All available backends passed the shared public API suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
