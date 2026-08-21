"""Backend-neutral pytest configuration for the shared TopologicPy public API suite.

The suite does not require a particular topology kernel.  At startup it detects
which supported backend packages are importable and selects one automatically,
preferring PythonOCC when both are installed.  An explicit ``--backend`` (or
``TOPOLOGICPY_CORE_BACKEND`` environment variable) is still honoured and fails
clearly if that backend is not available.
"""

from __future__ import annotations

import fnmatch
import importlib
import json
import os
from pathlib import Path

import pytest


BACKEND_ENV = "TOPOLOGICPY_CORE_BACKEND"
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
BACKEND_PREFERENCE = ("pythonocc", "topologic_core")


def _backend_is_importable(name: str) -> bool:
    """Return True only when the backend's runtime module imports successfully."""
    module_name = BACKENDS[name]["probe_module"]
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError, OSError):
        return False


def _available_backends() -> tuple[str, ...]:
    """Return supported, importable backends in automatic-selection order."""
    return tuple(name for name in BACKEND_PREFERENCE if _backend_is_importable(name))


def _requested_backend(config) -> str:
    """Resolve CLI/environment selection, defaulting to automatic selection."""
    command_line = config.getoption("--backend")
    if command_line is not None:
        return command_line.strip().lower()
    return (os.environ.get(BACKEND_ENV, "auto") or "auto").strip().lower()


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        choices=("pythonocc", "topologic_core", "auto"),
        default=None,
        help=(
            "Topology backend to test. 'auto' selects the first available backend "
            "(PythonOCC preferred, then TopologicCore)."
        ),
    )


def pytest_configure(config):
    requested = _requested_backend(config)
    available = _available_backends()

    if requested == "auto":
        if not available:
            raise pytest.UsageError(
                "No supported TopologicPy topology backend is importable. "
                "Install pythonocc-core or topologic_core."
            )
        selected = available[0]
    else:
        if requested not in BACKENDS:
            raise pytest.UsageError(f"Unknown TopologicPy backend: {requested!r}")
        if requested not in available:
            raise pytest.UsageError(
                f"Requested TopologicPy backend {requested!r} is not importable. "
                f"Available backends: {', '.join(available) if available else 'none'}."
            )
        selected = requested

    # Set this before test modules are collected/imported so every TopologicPy
    # module sees the same backend from its first import onward.
    os.environ[BACKEND_ENV] = selected
    config._topologicpy_requested_backend = requested
    config._topologicpy_backend = selected
    config._topologicpy_available_backends = available


def pytest_sessionstart(session):
    from topologicpy.Core import Core

    selected = session.config._topologicpy_backend
    expected_class = BACKENDS[selected]["class_name"]

    backend = Core.ResetBackend()
    active_class = backend.__class__.__name__
    if active_class != expected_class:
        raise pytest.UsageError(
            "Backend selection failed: "
            f"selected {selected!r} ({expected_class}), active {active_class!r}."
        )

    session.config._topologicpy_backend_object = backend


def pytest_report_header(config):
    active = getattr(config, "_topologicpy_backend", "unknown")
    available = getattr(config, "_topologicpy_available_backends", ())
    available_text = ", ".join(available) if available else "none"
    return [
        f"TopologicPy backend: {active}",
        f"Available TopologicPy backends: {available_text}",
    ]


@pytest.fixture(scope="session")
def backend_name(request):
    """Name of the backend selected for this pytest process."""
    return request.config._topologicpy_backend


@pytest.fixture(scope="session")
def available_backends(request):
    """Names of all supported backends importable in this pytest process."""
    return request.config._topologicpy_available_backends


@pytest.fixture(autouse=True)
def _restore_selected_backend(request):
    """Prevent tests that call Core.SetBackend from leaking backend state."""
    from topologicpy.Core import Core

    selected = request.config._topologicpy_backend_object
    Core.SetBackend(selected)
    yield
    Core.SetBackend(selected)


def _exceptions():
    path = Path(__file__).with_name("backend_exceptions.json")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def pytest_collection_modifyitems(config, items):
    """Apply the small, explicit set of known backend-specific exceptions."""
    backend = getattr(config, "_topologicpy_backend", None)
    rules = _exceptions().get(backend, {})

    for item in items:
        for pattern, reason in rules.items():
            if item.nodeid.endswith(pattern) or fnmatch.fnmatch(item.nodeid, f"*{pattern}"):
                item.add_marker(pytest.mark.skip(reason=f"[{backend}] {reason}"))
                break
