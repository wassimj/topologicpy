"""Backend-neutral pytest configuration for the shared TopologicPy public API suite."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

import pytest

BACKEND_ENV = "TOPOLOGICPY_CORE_BACKEND"
BACKEND_CLASS = {"pythonocc": "PythonOCCBackend", "topologic_core": "TopologicCoreBackend"}

def pytest_addoption(parser):
    parser.addoption("--backend", choices=("pythonocc", "topologic_core", "auto"), default=None)

def pytest_configure(config):
    requested = config.getoption("--backend")
    if requested is not None:
        os.environ[BACKEND_ENV] = requested

def pytest_sessionstart(session):
    from topologicpy.Core import Core
    requested = (session.config.getoption("--backend") or os.environ.get(BACKEND_ENV, "auto") or "auto").strip().lower()
    backend = Core.ResetBackend()
    active_class = backend.__class__.__name__
    if requested in BACKEND_CLASS and active_class != BACKEND_CLASS[requested]:
        raise pytest.UsageError(f"Backend selection failed: requested {requested!r}, active {active_class!r}")
    active = "pythonocc" if active_class == "PythonOCCBackend" else "topologic_core" if active_class == "TopologicCoreBackend" else active_class
    session.config._topologicpy_backend = active
    session.config._topologicpy_backend_object = backend

def pytest_report_header(config):
    return f"TopologicPy backend: {getattr(config, '_topologicpy_backend', 'unknown')}"

@pytest.fixture(scope="session")
def backend_name(request):
    return request.config._topologicpy_backend

@pytest.fixture(autouse=True)
def _restore_selected_backend(request):
    from topologicpy.Core import Core
    selected = request.config._topologicpy_backend_object
    Core.SetBackend(selected)
    yield
    Core.SetBackend(selected)

def _exceptions():
    path = Path(__file__).with_name("backend_exceptions.json")
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f: return json.load(f)

def pytest_collection_modifyitems(config, items):
    rules = _exceptions().get(getattr(config, "_topologicpy_backend", None), {})
    for item in items:
        for pattern, reason in rules.items():
            if item.nodeid.endswith(pattern) or fnmatch.fnmatch(item.nodeid, f"*{pattern}"):
                item.add_marker(pytest.mark.skip(reason=f"[{config._topologicpy_backend}] {reason}"))
                break
