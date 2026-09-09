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


def _line_edge():
    a = Vertex.ByCoordinates(0.0, 0.0, 0.0)
    b = Vertex.ByCoordinates(3.0, 4.0, 0.0)
    return Edge.ByStartVertexEndVertex(a, b)


def _wrap_occ_edge(shape):
    edge = Core.Edge.ByOcctShape(shape)
    assert edge is not None
    return edge


def _occ_circle_edge(radius=2.0):
    if not _is_pythonocc():
        pytest.skip("Exact curved-edge test is specific to the PythonOCC backend.")

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

    circle = gp_Circ(gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0)), radius)
    return _wrap_occ_edge(BRepBuilderAPI_MakeEdge(circle).Edge())


def _occ_bezier_edge(points):
    if not _is_pythonocc():
        pytest.skip("Bezier classification test is specific to the PythonOCC backend.")

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.Geom import Geom_BezierCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.gp import gp_Pnt

    poles = TColgp_Array1OfPnt(1, len(points))
    for index, (x, y, z) in enumerate(points, start=1):
        poles.SetValue(index, gp_Pnt(float(x), float(y), float(z)))
    curve = Geom_BezierCurve(poles)
    return _wrap_occ_edge(BRepBuilderAPI_MakeEdge(curve).Edge())


def test_edge_islinear_and_isclosed_for_straight_edge():
    edge = _line_edge()
    assert edge is not None
    assert Edge.IsLinear(edge) is True
    assert Edge.IsClosed(edge) is False
    assert math.isclose(Edge.Length(edge), 5.0, rel_tol=0.0, abs_tol=1.0e-6)


def test_edge_islinear_isclosed_validate_inputs():
    assert Edge.IsLinear(None, silent=True) is None
    assert Edge.IsClosed(None, silent=True) is None
    edge = _line_edge()
    assert Edge.IsLinear(edge, tolerance=0.0, silent=True) is None
    assert Edge.IsClosed(edge, tolerance=0.0, silent=True) is None
    assert Edge.IsLinear(edge, tolerance=float("inf"), silent=True) is None
    assert Edge.IsClosed(edge, tolerance=float("nan"), silent=True) is None


def test_pythonocc_closed_circle_is_closed_and_not_linear():
    edge = _occ_circle_edge(radius=2.0)
    assert Edge.IsClosed(edge) is True
    assert Edge.IsLinear(edge) is False
    assert math.isclose(Edge.Length(edge), 4.0 * math.pi, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_pythonocc_straight_bezier_is_geometrically_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is True
    assert math.isclose(Edge.Length(edge), 2.0, rel_tol=1.0e-6, abs_tol=1.0e-6)


def test_pythonocc_curved_bezier_is_not_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is False
    assert Edge.Length(edge) > 2.0


def test_pythonocc_collinear_backtracking_bezier_is_not_linear():
    edge = _occ_bezier_edge([
        (0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ])
    assert Edge.IsClosed(edge) is False
    assert Edge.IsLinear(edge) is False
    assert Edge.Length(edge) > 1.0
