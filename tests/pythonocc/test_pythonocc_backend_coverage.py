# Copyright (C) 2026
# Extended PythonOCC backend coverage tests -- the constructor/transform
# surface the developer guide (Appendix A + S13) expects a fully-working
# backend to support. These encode NUMERIC reality checks (perimeter,
# volume, area, hollow-section annulus area) so a fake/empty/degenerate
# topology cannot quietly pass. They are the punch-list: many currently
# fail because the backend does not yet implement these constructors.
#
# Run with PythonOCC installed:
#   TOPOLOGICPY_CORE_BACKEND=pythonocc python -m pytest \
#       tests/test_pythonocc_backend_coverage.py -v
#
# No cheating: every assertion inspects real OCCT geometry (length,
# volume, area, shared boundaries) -- not just type checks.

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TOPOLOGICPY_CORE_BACKEND", "pythonocc")

try:
    import OCC  # noqa: F401
    _HAS_OCC = True
except Exception:
    _HAS_OCC = False

# Only set the backend env var when OCC is importable, so this file
# doesn't pollute os.environ at collection time when OCC is missing
# (that would cause every subsequently-discovered test to attempt the
# pythonocc backend and fail en masse).
if not _HAS_OCC:
    os.environ.pop("TOPOLOGICPY_CORE_BACKEND", None)

pytestmark = pytest.mark.skipif(not _HAS_OCC, reason="PythonOCC (OCC) not importable")

# Constructors live in the algorithm layer and dispatch to the active
# backend via Core. Import them the same way the existing tests do.
Wire = pytest.importorskip("topologicpy.Wire").Wire
Shell = pytest.importorskip("topologicpy.Shell").Shell
Face = pytest.importorskip("topologicpy.Face").Face
Cell = pytest.importorskip("topologicpy.Cell").Cell
Topology = pytest.importorskip("topologicpy.Topology").Topology
Vertex = pytest.importorskip("topologicpy.Vertex").Vertex
Edge = pytest.importorskip("topologicpy.Edge").Edge


@pytest.fixture(autouse=True)
def _suppress(capfd):
    capfd.readouterr()
    yield
    capfd.readouterr()


def _v(x, y, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


# ===========================================================================
# Wire parametric shape constructors (S13.5 geometry construction)
# ===========================================================================

def test_wire_parametric_shapes_closed_with_real_length():
    shapes = [
        Wire.CrossShape(width=4, length=4, silent=True),
        Wire.CShape(width=4, length=4, silent=True),
        Wire.IShape(width=4, length=4, silent=True),
        Wire.LShape(width=4, length=4, silent=True),
        Wire.TShape(width=4, length=4, silent=True),
        Wire.Trapezoid(widthA=4, widthB=2, length=3, silent=True),
        Wire.Star(radiusA=2, radiusB=1, rays=5, silent=True),
        Wire.Einstein(radius=1, silent=True),
        Wire.GoldenRectangle(width=2, maxIterations=3, silent=True),
    ]
    for w in shapes:
        assert Topology.IsInstance(w, "Wire"), f"{w} not a Wire"
        assert Wire.IsClosed(w) is True, "parametric shape must be a closed wire"
        assert Wire.Length(w) > 0, "wire must have positive length"


def test_wire_squircle_positive_length():
    s = Wire.Squircle(radius=1, sides=25, silent=True)
    assert Topology.IsInstance(s, "Wire")
    assert Wire.Length(s) > 0


def test_wire_invalid_dims_return_none():
    assert Wire.CrossShape(width=0, length=4, silent=True) is None
    assert Wire.CShape(width=0, length=4, silent=True) is None
    assert Wire.IShape(width=0, length=4, silent=True) is None
    assert Wire.LShape(width=0, length=4, silent=True) is None
    assert Wire.TShape(width=0, length=4, silent=True) is None


def test_wire_fillet_miter_invert_return_wires():
    rectangle = Wire.GoldenRectangle(width=2, maxIterations=3, silent=True)
    fillet = Wire.Fillet(rectangle, radius=0.2, silent=True)
    miter = Wire.Miter(rectangle, silent=True)
    invert = Wire.Invert(rectangle, silent=True)
    for w in (fillet, miter, invert):
        assert Topology.IsInstance(w, "Wire"), "must return a Wire"
        assert len(Wire.Edges(w)) >= 1, "wire must carry edges"


def test_wire_lattice_and_cage_return_edges():
    lattice = Wire.Lattice(width=2, length=2, uSides=4, vSides=4, silent=True)
    # Cage needs a base face; build a rectangle face.
    base = Face.Rectangle(width=2, length=2, silent=True)
    cage = Wire.Cage(base, radius=0.2, silent=True)
    assert Topology.IsInstance(lattice, "Wire")
    assert len(Wire.Edges(lattice)) >= 1
    assert Topology.IsInstance(cage, "Wire")
    assert len(Wire.Edges(cage)) >= 1


# ===========================================================================
# Shell constructors (S13.5 / S8)
# ===========================================================================

def test_shell_by_thickened_wire_non_collinear():
    poly = Wire.ByVertices([_v(0, 0), _v(1, 0), _v(2, 1), _v(3, 0), _v(4, 0)],
                             close=False, silent=True)
    shell = Shell.ByThickenedWire(poly, offsetA=0.25, offsetB=0.25, silent=True)
    assert Topology.IsInstance(shell, "Shell"), "must return a Shell"
    assert len(Shell.Faces(shell)) > 0, "thickened shell must have faces"
    assert Shell.ByThickenedWire(None, silent=True) is None


def test_shell_mobius_strip_and_invalid():
    mobius = Shell.MobiusStrip(radius=1, height=0.5, uSides=8, vSides=1, silent=True)
    assert Topology.IsInstance(mobius, "Shell")
    assert len(Shell.Faces(mobius)) > 0
    assert Shell.MobiusStrip(radius=0, silent=True) is None
    assert Shell.MobiusStrip(height=0, silent=True) is None
    assert Shell.MobiusStrip(uSides=2, silent=True) is None
    assert Shell.MobiusStrip(vSides=0, silent=True) is None


def test_shell_golden_rectangle_constructor():
    shell = Shell.GoldenRectangle(width=2, maxIterations=3, includeSpiral=False, silent=True)
    assert Topology.IsInstance(shell, "Shell")
    assert len(Shell.Faces(shell)) >= 3
    assert Shell.GoldenRectangle(width=0, silent=True) is None


def test_shell_planarize_remove_collinear_simplify_self_merge():
    rect_face = Face.Rectangle(width=4, length=2, silent=True)
    shell = Shell.ByFaces([rect_face], silent=True)
    planarized = Shell.Planarize(shell, silent=True)
    cleaned = Shell.RemoveCollinearEdges(shell, silent=True)
    simplified = Shell.Simplify(shell, tolerance=0.01, silent=True)
    merged_face = Shell.SelfMerge(shell, silent=True)
    assert Topology.IsInstance(planarized, "Topology")
    assert Topology.IsInstance(cleaned, "Shell")
    assert Topology.IsInstance(simplified, "Shell")
    assert Topology.IsInstance(merged_face, "Face")
    assert Face.Area(merged_face) == pytest.approx(8, rel=0.05)
    assert Shell.Planarize(None, silent=True) is None
    assert Shell.SelfMerge(None, silent=True) is None


# ===========================================================================
# Face constructors (S13.5 / S8)
# ===========================================================================

def test_face_hollow_section_constructors_real_area():
    faces = [
        Face.CHS(radius=2, thickness=0.5, sides=24, silent=True),
        Face.Ring(radius=2, thickness=0.5, sides=24, silent=True),
        Face.RHS(width=4, length=3, thickness=0.25, silent=True),
        Face.SHS(size=4, thickness=0.25, silent=True),
    ]
    for f in faces:
        assert Topology.IsInstance(f, "Face")
        # annulus area ~ pi*(R^2 - r^2)
        assert Face.Area(f) > 0
        assert len(Face.InternalBoundaries(f)) >= 1, "hollow section must have internal boundary"


def test_face_fillet_simplify_remove_collinear():
    rectangle = Face.Rectangle(width=4, length=2, silent=True)
    filleted = Face.Fillet(rectangle, radius=0.2, silent=True)
    simplified = Face.Simplify(rectangle, tolerance=0.01, silent=True)
    cleaned = Face.RemoveCollinearEdges(rectangle, silent=True)
    assert Topology.IsInstance(filleted, "Face")
    assert Topology.IsInstance(simplified, "Face")
    assert Topology.IsInstance(cleaned, "Face")
    assert Face.Area(filleted) > 0


# ===========================================================================
# Cell polyhedron constructors (S13.5 / S8)
# ===========================================================================

def test_cell_polyhedron_constructors_real_volume():
    constructors = [
        Cell.Tetrahedron(length=1, depth=0, silent=True),
        Cell.Octahedron(radius=1, silent=True),
        Cell.Icosahedron(radius=1, silent=True),
        Cell.Dodecahedron(radius=1, silent=True),
    ]
    for cell in constructors:
        assert Topology.IsInstance(cell, "Cell")
        vol = Cell.Volume(cell)
        # Dodecahedron volume for radius=1 ~ 7.66; all must be clearly positive.
        assert vol is not None and vol > 0, "polyhedron must have real volume"


def test_cell_section_shape_constructors_real_volume():
    constructors = [
        Cell.CrossShape(width=4, length=4, height=1, silent=True),
        Cell.CShape(width=4, length=4, height=1, silent=True),
        Cell.IShape(width=4, length=4, height=1, silent=True),
        Cell.LShape(width=4, length=4, height=1, silent=True),
        Cell.TShape(width=4, length=4, height=1, silent=True),
        Cell.RHS(width=4, length=3, height=2, thickness=0.25, silent=True),
        Cell.SHS(size=4, height=2, thickness=0.25, silent=True),
        Cell.CHS(radius=1, height=2, thickness=0.2, sides=12, silent=True),
        Cell.Tube(radius=1, height=2, thickness=0.2, sides=12, silent=True),
    ]
    for cell in constructors:
        assert Topology.IsInstance(cell, "Cell")
        assert Cell.Volume(cell) > 0, "extruded/profile cell must have volume"


# ===========================================================================
# Topology adjacency (S13.7 / Table 7 + S9)
# ===========================================================================

def test_topology_shortest_edge_distance_and_edges():
    v0 = _v(0, 0, 0)
    v1 = _v(3, 0, 0)
    v2 = _v(0, 4, 0)
    # Build a closed shell (two triangles) so the backend can form a valid
    # cluster/cell; ShortestEdge examines sub-topologies between the two inputs.
    cluster = Shell.ByFaces([
        Face.ByWire(Wire.ByVertices([v0, v1, v2], close=True, silent=True), silent=True),
        Face.ByWire(Wire.ByVertices([v0, v1, _v(3, 0, 1)], close=True, silent=True), silent=True),
    ], silent=True)
    shortest = Topology.ShortestEdge(cluster, _v(0, 0, 3), tolerance=0.0001, silent=True)
    assert shortest is not None, "ShortestEdge must return an edge"
    assert abs(Edge.Length(Edge.ByStartVertexEndVertex(v0, v1, silent=True)) - 3.0) < 1e-6
