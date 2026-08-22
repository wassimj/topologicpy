"""Unit and integration tests for topologicpy.Aperture.

These tests cover:
- invalid-input handling,
- aperture construction and topology extraction,
- context/host round-tripping,
- retrieval of an aperture from its host topology.

The host-retrieval tests are important because successful construction of an
Aperture object alone does not prove that the relationship has been registered
and can later be discovered from the host.
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
    """Create a context using the canonical current TopologicPy API."""
    return Context.ByTopologyParameters(topology, u=0.5, v=0.5, w=0.5)


def _contains_same(topologies, target):
    """Return True if a list contains the same underlying topology."""
    if not isinstance(topologies, list):
        return False
    return any(Topology.IsSame(item, target, silent=True) for item in topologies)


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


def test_context_roundtrip_returns_host_topology():
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)

    returned_host = Context.Topology(context)

    assert Topology.IsInstance(returned_host, "Face")
    assert Topology.IsSame(returned_host, host, silent=True)


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

    area = Face.Area(returned_topology)
    assert area == pytest.approx(0.25, abs=1e-6)


def test_by_topology_context_registers_aperture_with_host():
    """Creating an Aperture must make it discoverable from its context host."""
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)
    aperture_topology = _rectangle(width=0.5, length=0.5)

    aperture = Aperture.ByTopologyContext(aperture_topology, context)

    assert Topology.IsInstance(aperture, "Aperture")

    host_apertures = Topology.Apertures(host, silent=True)

    assert isinstance(host_apertures, list)
    assert len(host_apertures) == 1
    assert _contains_same(host_apertures, aperture_topology)


def test_host_aperture_retrieval_is_repeatable():
    """Repeated host queries must not lose or duplicate the relationship."""
    host = _rectangle(width=2, length=2)
    context = _context_for_topology(host)
    aperture_topology = _rectangle(width=0.5, length=0.5)

    aperture = Aperture.ByTopologyContext(aperture_topology, context)
    assert Topology.IsInstance(aperture, "Aperture")

    first = Topology.Apertures(host, silent=True)
    second = Topology.Apertures(host, silent=True)

    assert isinstance(first, list)
    assert isinstance(second, list)
    assert len(first) == 1
    assert len(second) == 1
    assert _contains_same(first, aperture_topology)
    assert _contains_same(second, aperture_topology)
