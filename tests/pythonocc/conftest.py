# Copyright (C) 2026
# PythonOCC backend parity test configuration.

import os
import pytest

# Set backend before any topologicpy imports
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

# Force-set the backend so it works even when Core._backend was already
# initialized by a TopologicCore test running in the same pytest session.
from topologicpy.Core import Core
try:
    from topologicpy.pythonocc_backend import PythonOCCBackend
    Core.SetBackend(PythonOCCBackend())
except Exception:
    pass


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset backend after each test."""
    from topologicpy.Core import Core
    try:
        from topologicpy.pythonocc_backend import PythonOCCBackend
        Core.SetBackend(PythonOCCBackend())
    except Exception:
        pass
    yield
    Core.ResetBackend()


@pytest.fixture
def backend():
    """Get the PythonOCC backend instance."""
    from topologicpy.Core import Core
    return Core.Backend()
