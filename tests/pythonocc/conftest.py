# Copyright (C) 2026
# PythonOCC backend parity test configuration.

import os
import pytest

# Set backend before any topologicpy imports
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset backend after each test."""
    from topologicpy.Core import Core
    yield
    Core.ResetBackend()


@pytest.fixture
def backend():
    """Get the PythonOCC backend instance."""
    from topologicpy.Core import Core
    return Core.Backend()
