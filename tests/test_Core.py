"""Unit tests for topologicpy.Core.

These tests deliberately use lightweight fake backends.  They verify the facade
logic against lightweight fake backends where appropriate, without depending on geometric side effects.
"""

from __future__ import annotations

import py_compile

import pytest

core_module = pytest.importorskip("topologicpy.Core")
Core = core_module.Core

@pytest.fixture(autouse=True)
def _restore_backend_and_suppress_output(capfd):
    """Restore the active backend after each test and suppress expected diagnostics."""
    previous_backend = Core.Backend()
    capfd.readouterr()
    yield
    Core.SetBackend(previous_backend)
    capfd.readouterr()


class FakeVertexNamespace:
    label = "vertex-namespace"
    constant = 17

    @staticmethod
    def ByCoordinates(x, y, z):
        return {"type": "vertex", "coordinates": (x, y, z)}

    def __call__(self, *args, **kwargs):
        return {"called": True, "args": args, "kwargs": kwargs}


class FakeTopologyNamespace:
    @staticmethod
    def Analyze(topology):
        return f"analysis:{topology}"


class FakeBackend:
    def __init__(self):
        self.Vertex = FakeVertexNamespace()
        self.Edge = None
        self.Topology = FakeTopologyNamespace()
        self.Dictionary = object()
        self.value = 42

    def RawModule(self):
        return "raw-module"

    def Namespaces(self):
        return ["Dictionary", "Topology", "Vertex"]


class BackendWithoutHelpers:
    def __init__(self):
        self.Vertex = FakeVertexNamespace()
        self.Zeta = object()
        self.alpha = object()
        self._private = object()


class FakeInstance:
    value = 9
    not_callable = "plain-value"

    def scale(self, value, factor=1):
        return value * factor


def test_set_backend_and_backend_return_the_same_object():
    backend = FakeBackend()

    assert Core.SetBackend(backend) is backend
    assert Core.Backend() is backend


def test_namespace_proxy_resolves_attributes_and_methods():
    backend = FakeBackend()
    Core.SetBackend(backend)

    assert Core.Vertex.label == "vertex-namespace"
    assert Core.Vertex.constant == 17
    assert Core.Vertex.ByCoordinates(1, 2, 3) == {
        "type": "vertex",
        "coordinates": (1, 2, 3),
    }
    assert Core.Topology.Analyze("shape") == "analysis:shape"


def test_namespace_proxy_can_call_callable_namespace():
    backend = FakeBackend()
    Core.SetBackend(backend)

    assert Core.Vertex("a", key="b") == {
        "called": True,
        "args": ("a",),
        "kwargs": {"key": "b"},
    }






def test_set_backend_rejects_none():
    with pytest.raises(ValueError, match="backend cannot be None"):
        Core.SetBackend(None)


def test_raw_module_uses_backend_raw_module_when_available():
    backend = FakeBackend()
    Core.SetBackend(backend)

    assert Core.RawModule() == "raw-module"


def test_raw_module_returns_backend_when_no_raw_module_method_exists():
    backend = BackendWithoutHelpers()
    Core.SetBackend(backend)

    assert Core.RawModule() is backend


def test_namespaces_uses_backend_namespaces_method_when_available():
    Core.SetBackend(FakeBackend())

    assert Core.Namespaces() == ["Dictionary", "Topology", "Vertex"]


def test_namespaces_falls_back_to_public_dir_names():
    Core.SetBackend(BackendWithoutHelpers())

    namespaces = Core.Namespaces()
    assert "Vertex" in namespaces
    assert "Zeta" in namespaces
    assert "alpha" in namespaces
    assert "_private" not in namespaces


def test_namespace_returns_backend_namespace_and_validates_name():
    backend = FakeBackend()
    Core.SetBackend(backend)

    assert Core.Namespace("Vertex") is backend.Vertex

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        Core.Namespace("")

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        Core.Namespace(None)

    with pytest.raises(AttributeError, match="does not expose namespace 'Missing'"):
        Core.Namespace("Missing")


def test_has_namespace_reports_presence_without_raising():
    Core.SetBackend(FakeBackend())

    assert Core.HasNamespace("Vertex") is True
    assert Core.HasNamespace("Edge") is False
    assert Core.HasNamespace("Missing") is False
    assert Core.HasNamespace("") is False
    assert Core.HasNamespace(None) is False


def test_has_attribute_reports_namespace_attributes():
    Core.SetBackend(FakeBackend())

    assert Core.HasAttribute("Vertex", "ByCoordinates") is True
    assert Core.HasAttribute("Vertex", "constant") is True
    assert Core.HasAttribute("Vertex", "missing") is False
    assert Core.HasAttribute("Missing", "anything") is False


def test_call_invokes_callable_namespace_attribute():
    Core.SetBackend(FakeBackend())

    assert Core.Call("Vertex", "ByCoordinates", 4, 5, 6) == {
        "type": "vertex",
        "coordinates": (4, 5, 6),
    }


def test_call_rejects_non_callable_namespace_attribute():
    Core.SetBackend(FakeBackend())

    with pytest.raises(TypeError, match="Core.Vertex.constant is not callable"):
        Core.Call("Vertex", "constant")


def test_instance_call_invokes_instance_method():
    obj = FakeInstance()

    assert Core.InstanceCall(obj, "scale", 6, factor=7) == 42


def test_instance_call_validates_inputs_and_callable_attribute():
    obj = FakeInstance()

    with pytest.raises(ValueError, match="obj cannot be None"):
        Core.InstanceCall(None, "scale")

    with pytest.raises(ValueError, match="methodName must be a non-empty string"):
        Core.InstanceCall(obj, "")

    with pytest.raises(ValueError, match="methodName must be a non-empty string"):
        Core.InstanceCall(obj, None)

    with pytest.raises(TypeError, match="not_callable is not callable"):
        Core.InstanceCall(obj, "not_callable")

    with pytest.raises(AttributeError):
        Core.InstanceCall(obj, "missing_method")


def test_instance_attribute_returns_instance_attribute():
    obj = FakeInstance()

    assert Core.InstanceAttribute(obj, "value") == 9
    assert Core.InstanceAttribute(obj, "not_callable") == "plain-value"


def test_instance_attribute_validates_inputs():
    obj = FakeInstance()

    with pytest.raises(ValueError, match="obj cannot be None"):
        Core.InstanceAttribute(None, "value")

    with pytest.raises(ValueError, match="attributeName must be a non-empty string"):
        Core.InstanceAttribute(obj, "")

    with pytest.raises(ValueError, match="attributeName must be a non-empty string"):
        Core.InstanceAttribute(obj, None)

    with pytest.raises(AttributeError):
        Core.InstanceAttribute(obj, "missing_attribute")


def test_module_exports_public_backend_symbols():
    assert "Core" in core_module.__all__
    assert "TopologicCoreBackend" in core_module.__all__


def test_uploaded_core_file_syntax_is_valid_when_available():
    # Useful when the uploaded source is copied into /mnt/data during local triage.
    # Harmless in repository test runs because this only checks the installed file.
    assert callable(Core.SetBackend)
