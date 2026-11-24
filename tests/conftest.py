"""Pytest configuration for the PythonOCC public API suite."""
import importlib
import pytest


def pytest_addoption(parser):
    parser.addoption("--backend", choices=("pythonocc",), default="pythonocc",
                     help="Compatibility option; PythonOCC is the sole supported kernel.")


def pytest_configure(config):
    config.addinivalue_line("markers", "pythonocc_only: native geometry contract")
    config.addinivalue_line("markers", "stress: computational stress/regression test")
    try:
        importlib.import_module("OCC.Core.TopoDS")
    except (ImportError, OSError) as exc:
        raise pytest.UsageError("PythonOCC is required. See README.md for installation.") from exc


def pytest_sessionstart(session):
    from topologicpy.Core import Core
    session.config._topologicpy_backend_object = Core.ResetBackend()


def pytest_report_header(config):
    return "TopologicPy geometry kernel: PythonOCC"


@pytest.fixture(scope="session")
def backend_name():
    return "pythonocc"


@pytest.fixture(scope="session")
def available_backends():
    return ("pythonocc",)


@pytest.fixture(autouse=True)
def _restore_selected_backend(request):
    """Isolate namespace test doubles used by the Core unit tests."""
    from topologicpy.Core import Core
    selected = request.config._topologicpy_backend_object
    Core.SetBackend(selected)
    yield
    Core.SetBackend(selected)
