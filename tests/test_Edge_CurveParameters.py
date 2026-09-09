import math

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


def _occ_bezier_shape(points):
    if not _is_pythonocc():
        pytest.skip("Exact curved-edge parameter test is specific to the PythonOCC backend.")

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BezierCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt

    poles = TColgp_Array1OfPnt(1, len(points))
    for index, (x, y, z) in enumerate(points, start=1):
        poles.SetValue(index, gp_Pnt(float(x), float(y), float(z)))
    curve = Geom_BezierCurve(poles)
    return BRepBuilderAPI_MakeEdge(curve).Edge()


def _occ_quadratic_bezier_edge(reversed_orientation=False):
    shape = _occ_bezier_shape([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    if reversed_orientation:
        shape = shape.Reversed()
    return _wrap_occ_edge(shape)


def test_linear_edge_parameter_round_trip_and_tangent():
    edge = _line_edge()
    vertex = Edge.VertexByParameter(edge, u=0.25)

    assert vertex is not None
    assert _coords(vertex) == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, vertex, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-7
    )


def test_parameter_endpoints_return_edge_endpoints():
    edge = _line_edge()
    start = Edge.StartVertex(edge)
    end = Edge.EndVertex(edge)

    assert _coords(Edge.VertexByParameter(edge, 0.0)) == pytest.approx(_coords(start), abs=1.0e-9)
    assert _coords(Edge.VertexByParameter(edge, 1.0)) == pytest.approx(_coords(end), abs=1.0e-9)
    assert Edge.ParameterAtVertex(edge, start) == pytest.approx(0.0, abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, end) == pytest.approx(1.0, abs=1.0e-7)


def test_parameter_methods_validate_inputs():
    edge = _line_edge()
    off_edge = Vertex.ByCoordinates(1.0, 1.0, 0.0)

    assert Edge.VertexByParameter(None, 0.5, silent=True) is None
    assert Edge.VertexByParameter(edge, "not-a-number", silent=True) is None
    assert Edge.VertexByParameter(edge, -0.1, silent=True) is None
    assert Edge.VertexByParameter(edge, 1.1, silent=True) is None
    assert Edge.ParameterAtVertex(None, off_edge, silent=True) is None
    assert Edge.ParameterAtVertex(edge, None, silent=True) is None
    assert Edge.ParameterAtVertex(edge, off_edge, silent=True) is None
    assert Edge.TangentAtParameter(None, silent=True) is None
    assert Edge.TangentAtParameter(edge, u="not-a-number", silent=True) is None


def test_pythonocc_quadratic_bezier_parameter_evaluation():
    edge = _occ_quadratic_bezier_edge()

    quarter = Edge.VertexByParameter(edge, u=0.25)
    middle = Edge.VertexByParameter(edge, u=0.5)

    assert _coords(quarter) == pytest.approx([0.5, 0.375, 0.0], abs=1.0e-7)
    assert _coords(middle) == pytest.approx([1.0, 0.5, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, quarter, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, middle, mantissa=9) == pytest.approx(0.5, abs=1.0e-7)


def test_pythonocc_quadratic_bezier_tangent_uses_curve_derivative():
    edge = _occ_quadratic_bezier_edge()

    expected_quarter = [2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0]
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        expected_quarter, abs=1.0e-7
    )
    assert Edge.TangentAtParameter(edge, u=0.5, mantissa=9) == pytest.approx(
        [1.0, 0.0, 0.0], abs=1.0e-7
    )


def test_pythonocc_reversed_curve_respects_topological_orientation():
    edge = _occ_quadratic_bezier_edge(reversed_orientation=True)

    quarter = Edge.VertexByParameter(edge, u=0.25)
    assert _coords(quarter) == pytest.approx([1.5, 0.375, 0.0], abs=1.0e-7)
    assert Edge.ParameterAtVertex(edge, quarter, mantissa=9) == pytest.approx(0.25, abs=1.0e-7)

    expected_tangent = [-2.0 / math.sqrt(5.0), 1.0 / math.sqrt(5.0), 0.0]
    assert Edge.TangentAtParameter(edge, u=0.25, mantissa=9) == pytest.approx(
        expected_tangent, abs=1.0e-7
    )
