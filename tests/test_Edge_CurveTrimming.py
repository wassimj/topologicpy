import pytest

from topologicpy.Core import Core
from topologicpy.Edge import Edge
from topologicpy.Vertex import Vertex


def _backend_name():
    backend = Core.Backend()
    for name in ("BackendName", "Name", "backend_name"):
        value = getattr(backend, name, None)
        if callable(value):
            try:
                return str(value())
            except Exception:
                pass
    return backend.__class__.__name__


def _is_pythonocc():
    return "pythonocc" in _backend_name().lower()


def _coords(vertex):
    return [
        Vertex.X(vertex, mantissa=9),
        Vertex.Y(vertex, mantissa=9),
        Vertex.Z(vertex, mantissa=9),
    ]


def _line_edge():
    a = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    b = Vertex.ByCoordinates(4.0, 0.0, 0.0)
    return Edge.ByStartVertexEndVertex(a, b)


def _wrap_occ_edge(shape):
    edge = Core.Edge.ByOcctShape(shape)
    assert edge is not None
    return edge


def _occ_quadratic_bezier_edge(reversed_orientation=False):
    if not _is_pythonocc():
        pytest.skip("Exact curved-edge trimming is specific to the PythonOCC backend.")

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BezierCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt

    poles = TColgp_Array1OfPnt(1, 3)
    poles.SetValue(1, gp_Pnt(0.0, 0.0, 0.0))
    poles.SetValue(2, gp_Pnt(1.0, 1.0, 0.0))
    poles.SetValue(3, gp_Pnt(2.0, 0.0, 0.0))
    shape = BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge()
    if reversed_orientation:
        shape = shape.Reversed()
    return _wrap_occ_edge(shape)


def test_trim_by_parameters_linear_forward():
    edge = _line_edge()
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([3.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(2.0, abs=1.0e-7)
    assert Edge.IsLinear(trimmed)


def test_trim_by_parameters_linear_reverse_direction():
    edge = _line_edge()
    trimmed = Edge.TrimByParameters(edge, 0.75, 0.25)

    assert trimmed is not None
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([3.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(2.0, abs=1.0e-7)


def test_trim_by_parameters_identity_full_reverse_and_validation():
    edge = _line_edge()

    assert Edge.TrimByParameters(edge, 0.0, 1.0) is edge

    reversed_edge = Edge.TrimByParameters(edge, 1.0, 0.0)
    assert reversed_edge is not None
    assert _coords(Edge.StartVertex(reversed_edge)) == pytest.approx([4.0, 0.0, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(reversed_edge)) == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-7)

    assert Edge.TrimByParameters(None, 0.25, 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, "bad", 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, -0.1, 0.75, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.25, 1.1, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.5, 0.5, silent=True) is None
    assert Edge.TrimByParameters(edge, 0.25, 0.75, tolerance=0, silent=True) is None


def test_pythonocc_trimmed_bezier_preserves_curve_geometry():
    edge = _occ_quadratic_bezier_edge()
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.Length(trimmed, mantissa=9) == pytest.approx(1.040228819, abs=1.0e-7)


def test_pythonocc_reverse_trimmed_bezier_preserves_curve_and_direction():
    edge = _occ_quadratic_bezier_edge()
    trimmed = Edge.TrimByParameters(edge, 0.75, 0.25)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.TangentAtParameter(trimmed, 0.5, mantissa=9) == pytest.approx([-1.0, 0.0, 0.0], abs=1.0e-7)


def test_pythonocc_trim_on_reversed_source_uses_topological_parameters():
    edge = _occ_quadratic_bezier_edge(reversed_orientation=True)
    trimmed = Edge.TrimByParameters(edge, 0.25, 0.75)

    assert trimmed is not None
    assert Edge.IsLinear(trimmed) is False
    assert _coords(Edge.StartVertex(trimmed)) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.EndVertex(trimmed)) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(Edge.VertexByParameter(trimmed, 0.5)) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
