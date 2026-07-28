# Auto-generated PythonOCC backend parity test
# Copied from test_Aperture.py

import os
os.environ["TOPOLOGICPY_CORE_BACKEND"] = "pythonocc"

"""Unit tests for topologicpy.Aperture.

These tests are intentionally compact because Aperture exposes only two public
methods. They verify valid construction/extraction and invalid-input handling
without relying on exact object identity from the geometry kernel.
"""

import pytest

Aperture = pytest.importorskip("topologicpy.Aperture").Aperture
Context = pytest.importorskip("topologicpy.Context").Context
Face = pytest.importorskip("topologicpy.Face").Face
Topology = pytest.importorskip("topologicpy.Topology").Topology
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex


@pytest.fixture(autouse=True)
def _suppress_expected_topologicpy_output(capfd):
    """Keep expected TopologicPy diagnostic prints out of normal pytest output."""
    capfd.readouterr()
    yield
    capfd.readouterr()


def _rectangle(*, width=1.0, length=1.0, origin=None):
    kwargs = {"width": width, "length": length}
    if origin is not None:
        kwargs["origin"] = origin

    try:
        return Face.Rectangle(**kwargs, silent=True)
    except TypeError:
        return Face.Rectangle(**kwargs)


def _context_for_topology(topology):
    """Create a context using common Context.ByTopologyParameters signatures."""
    attempts = [
        lambda: Context.ByTopologyParameters(topology, u=0.5, v=0.5, w=0.0),
        lambda: Context.ByTopologyParameters(topology, 0.5, 0.5, 0.0),
        lambda: Context.ByTopologyParameters(topology, u=0.5, v=0.5),
        lambda: Context.ByTopologyParameters(topology, 0.5, 0.5),
        lambda: Context.ByTopologyParameters(topology),
    ]

    for attempt in attempts:
        try:
            context = attempt()
        except TypeError:
            continue
        except Exception:
            continue

        if Topology.IsInstance(context, "Context"):
            return context

    return None


def test_topology_returns_none_for_invalid_aperture():
    assert Aperture.Topology(None) is None
    assert Aperture.Topology("not an aperture") is None


def test_by_topology_context_rejects_invalid_inputs():
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)

    assert Topology.IsInstance(host, "Face")
    assert Topology.IsInstance(context, "Context")

    aperture_topology = _rectangle(width=0.5, length=0.5)

    assert Aperture.ByTopologyContext(None, context) is None
    assert Aperture.ByTopologyContext("not a topology", context) is None
    assert Aperture.ByTopologyContext(aperture_topology, None) is None
    assert Aperture.ByTopologyContext(aperture_topology, "not a context") is None


def test_by_topology_context_creates_aperture():
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)

    assert Topology.IsInstance(host, "Face")
    assert Topology.IsInstance(context, "Context")

    aperture_origin = Vertex.ByCoordinates(0, 0, 0)
    aperture_topology = _rectangle(width=0.5, length=0.5, origin=aperture_origin)

    aperture = Aperture.ByTopologyContext(aperture_topology, context)

    assert Topology.IsInstance(aperture, "Aperture")


def test_topology_returns_aperture_representation():
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)
    aperture_topology = _rectangle(width=0.5, length=0.5)
    aperture = Aperture.ByTopologyContext(aperture_topology, context)

    returned_topology = Aperture.Topology(aperture)

    assert Topology.IsInstance(returned_topology, "Topology")
    assert Topology.IsInstance(returned_topology, "Face")

    try:
        area = Face.Area(returned_topology)
    except Exception:
        area = None

    if area is not None:
        assert area == pytest.approx(0.25, abs=1e-6)
