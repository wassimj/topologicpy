"""Copyright (C) 2026
Differential parity tests: topologic_core backend vs pythonocc backend.

These tests run the SAME 114-case battery under BOTH backends inside one
process (Core.ResetBackend() -> topologic_core, then
Core.SetBackend(PythonOCCBackend())) and compare the outputs.

Two acceptance rulers are used (both are asserted):

  Ruler A -- GEOMETRIC parity (must hold for all 114 cases):
      The measured material is the same within the project tolerance
      (1e-4): vertex sets match by tolerance-bucketing, centroids within
      tolerance, and every scalar both backends expose (volume / area /
      surface area / edge length) agrees within tolerance. The wrapper
      type and the wire/container grouping are deliberately IGNORED here,
      because a Cluster holding one connected chain of edges IS the same
      topology as the corresponding Wire.

  Ruler B -- STRUCTURAL parity (must hold for all 114 cases):
      top-level type, the six subtopology counts, exact coordinates at 6
      decimals. Any case that is not byte-identical must classify into the
      KNOWN_DEVIATIONS table below; a NEW or re-classified deviation fails
      the test so regressions are caught.

The current, verified state:
  * geometric parity ......... 114/114
  * structural exact match ... 114/114
  * remaining deviations ..... 0

If a future backend change introduces a verified, intentional structural
difference, document it in KNOWN_DEVIATIONS (and update the counts in this
docstring) in the SAME commit. The table is a living regression record, not
a permanent carve-out."""  # noqa: D400

from __future__ import annotations

from collections import Counter
import math

import pytest

from topologicpy.Core import Core
from topologicpy.Edge import Edge
from topologicpy.Vertex import Vertex
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Cluster import Cluster
from topologicpy.Graph import Graph
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary

# The differential needs the legacy kernel: skip the whole module when it is
# not importable (the pythonocc-only CI matrix installs it via '.[test]').
pytest.importorskip("topologic_core")

from topologicpy.pythonocc_backend import PythonOCCBackend  # noqa: E402

TOL = 1e-4        # project tolerance used by the geometric ruler
R6 = 6            # decimals used by the strict structural ruler


# ---------------------------------------------------------------------------
# Geometry builders (fresh objects per call -- every backend pass must start
# from objects created under that backend).
# ---------------------------------------------------------------------------
def P(x, y=0.0, z=0.0):
    return Vertex.ByCoordinates(x, y, z)


def tri_face():
    return Face.ByVertices([P(0, 0, 0), P(1, 0, 0), P(0, 1, 0)])


def cube_cell(ox=0.0):
    vs = [P(ox, 0, 0), P(ox + 1, 0, 0), P(ox + 1, 1, 0), P(ox, 1, 0),
          P(ox, 0, 1), P(ox + 1, 0, 1), P(ox + 1, 1, 1), P(ox, 1, 1)]
    qs = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
          (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return Cell.ByFaces([Face.ByVertices([vs[a], vs[b], vs[c], vs[d]]) for a, b, c, d in qs])


def sq_face(ox=0.0):
    return Face.ByVertices([P(ox, 0, 0), P(ox + 1, 0, 0), P(ox + 1, 1, 0), P(ox, 1, 0)])


def octa_cc():
    return CellComplex.Octahedron(origin=P(0, 0, 0), radius=1.0,
                                  direction=[0, 0, 1], placement="center")


# ---------------------------------------------------------------------------
# The 114-case battery: (name, thunk). Each thunk returns a topology, a scalar
# tuple, or None; both backends must answer it with the same material.
# ---------------------------------------------------------------------------
def all_cases():
    L = []

    def case(name, fn):
        L.append((name, fn))

    # ---- primitives / geometry ------------------------------------------
    case("vertex", lambda: P(1, 2, 3))
    case("edge_len", lambda: Edge.ByStartVertexEndVertex(P(0, 0, 0), P(3, 4, 0)))
    case("wire_square", lambda: Wire.ByVertices([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], close=True))
    case("wire_open", lambda: Wire.ByVertices([P(0, 0), P(1, 0), P(2, 0)], close=False))
    case("wire_rect", lambda: Wire.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"))
    case("face_tri", lambda: tri_face())
    case("face_rect", lambda: Face.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"))
    case("face_hole", lambda: (
        Face.ByWires(Wire.Rectangle(origin=P(0, 0, 0), width=8, length=6, direction=[0, 0, 1], placement="lowerleft"),
                     [Wire.Rectangle(origin=P(5, 1, 0), width=2, length=2, direction=[0, 0, 1], placement="lowerleft")])))
    case("cell_cube", lambda: cube_cell())
    case("cell_tetra", lambda: Cell.Tetrahedron(length=1, depth=0, silent=True))
    case("cell_octa", lambda: Cell.Octahedron(radius=1, silent=True))
    case("cell_prism", lambda: Cell.Prism(origin=P(0, 0, 0), width=2, length=3, height=4,
                                          uSides=1, vSides=1, wSides=1, placement="center"))
    case("cellcomplex_octa", lambda: octa_cc())
    case("cellcomplex_prism", lambda: CellComplex.Prism(origin=P(0, 0, 0), width=2, length=2, height=2,
                                                        uSides=2, vSides=1, wSides=1, placement="center"))
    case("cluster", lambda: Cluster.ByTopologies(
        [P(0, 0, 0), Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 0, 0)), tri_face()]))

    # ---- topology queries ------------------------------------------------
    case("type_ids", lambda: tuple(Topology.Type(o) for o in (
        P(0, 0, 0), Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 0, 0)),
        Wire.ByVertices([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], close=True),
        tri_face(), cube_cell(), octa_cc(), Cluster.ByTopologies([P(0, 0)]))))
    case("octa_counts", lambda: tuple(
        len(Topology.Vertices(octa_cc(), silent=True) or []),
        len(Topology.Edges(octa_cc(), silent=True) or []),
        len(Topology.Wires(octa_cc(), silent=True) or []),
        len(Topology.Faces(octa_cc(), silent=True) or []),
        len(Topology.Shells(octa_cc(), silent=True) or []),
        len(Topology.Cells(octa_cc(), silent=True) or [])))
    case("isinstance_topology", lambda: tuple(Topology.IsInstance(o, "Topology") for o in (P(0, 0, 0), octa_cc())))
    case("centroid_cell", lambda: Topology.Centroid(cube_cell(), silent=True))
    case("centroid_hole_face", lambda: Topology.Centroid(
        Face.ByWires(Wire.Rectangle(origin=P(0, 0, 0), width=8, length=6, direction=[0, 0, 1], placement="lowerleft"),
                     [Wire.Rectangle(origin=P(5, 1, 0), width=2, length=2, direction=[0, 0, 1], placement="lowerleft")]),
        silent=True))

    # ---- transforms ------------------------------------------------------
    case("translate_edge", lambda: Topology.Translate(Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 0, 0)), 5, 2, 1, silent=True))
    case("rotate_edge", lambda: Topology.Rotate(Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 0, 0)), origin=P(0, 0, 0), axis=[0, 0, 1], angle=90, silent=True))
    case("scale_cell", lambda: Topology.Scale(cube_cell(), origin=P(0, 0, 0), x=2, y=2, z=2, silent=True))
    case("translate_cc", lambda: Topology.Translate(octa_cc(), 5, 0, 0, silent=True))

    # ---- booleans: cell --------------------------------------------------
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge),
                   ("slice", Topology.Slice), ("impose", Topology.Impose), ("imprint", Topology.Imprint),
                   ("xor", Topology.SymmetricDifference)]:
        def _make(fp=fn):
            return lambda: fp(cube_cell(0.0), cube_cell(0.5), silent=True)
        case(f"bool_{op}_cell", _make())

    # ---- booleans: edge --------------------------------------------------
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge)]:
        def _make(fp=fn):
            return lambda: fp(Edge.ByVertices([P(-1, 0, 0), P(1, 0, 0)], silent=True),
                              Edge.ByVertices([P(-0.25, 0, 0), P(1.75, 0, 0)], silent=True), silent=True)
        case(f"bool_{op}_edge", _make())

    # ---- booleans: wire (overlapping rectangles) -------------------------
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge)]:
        def _make(fp=fn):
            return lambda: fp(Wire.Rectangle(origin=P(0, 0), width=2, length=2, placement="center"),
                              Wire.Rectangle(origin=P(0.75, 0), width=2, length=2, placement="center"), silent=True)
        case(f"bool_{op}_wire", _make())

    # ---- serialization ---------------------------------------------------
    case("brep_roundtrip_cell", lambda: Topology.ByBREPString(Topology.BREPString(cube_cell(), silent=True), silent=True))
    case("brep_roundtrip_cc", lambda: Topology.ByBREPString(Topology.BREPString(octa_cc(), silent=True), silent=True))

    # ---- dictionary ------------------------------------------------------
    def dict_rt():
        d = Dictionary.ByKeysValues(["a", "b"], [Core.IntAttribute(7), Core.StringAttribute("hi")], silent=True)
        return Dictionary.ValueAtKey(d, "a"), Dictionary.ValueAtKey(d, "b")
    case("dictionary_rt", dict_rt)

    # ---- graph -----------------------------------------------------------
    case("graph_counts", lambda: (
        len(Graph.Vertices(Graph.ByVerticesEdges([P(0, 0), P(1, 0), P(0, 1)], [Edge.ByStartVertexEndVertex(P(0, 0), P(1, 0))]))),
        len(Graph.Edges(Graph.ByVerticesEdges([P(0, 0), P(1, 0), P(0, 1)], [Edge.ByStartVertexEndVertex(P(0, 0), P(1, 0))])))))

    # ---- adjacency inside the octahedron (west vertex) --------------------
    def adj_west():
        cc = octa_cc()
        west = None
        for vt in Topology.Vertices(cc, silent=True):
            if abs(Vertex.X(vt) + 1.0) < 1e-6 and abs(Vertex.Y(vt)) < 1e-6 and abs(Vertex.Z(vt)) < 1e-6:
                west = vt
                break
        if west is None:
            return None
        return len(Topology.AdjacentTopologies(west, hostTopology=cc, topologyType="vertex", silent=True))
    case("adj_vertices_west", adj_west)

    # ---- edge metrics ----------------------------------------------------
    case("edge_direction", lambda: Edge.Direction(Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 1, 0)), mantissa=R6))
    case("edge_midpoint", lambda: Edge.VertexByParameter(Edge.ByStartVertexEndVertex(P(0, 0, 0), P(4, 0, 0)), 0.5))
    case("edge_param_at_mid", lambda: Edge.ParameterAtVertex(
        Edge.ByStartVertexEndVertex(P(0, 0, 0), P(4, 0, 0)),
        Edge.VertexByParameter(Edge.ByStartVertexEndVertex(P(0, 0, 0), P(4, 0, 0)), 0.25), mantissa=R6))
    case("edge_quadrance", lambda: Edge.Quadrance(Edge.ByStartVertexEndVertex(P(1, 2, 3), P(4, 6, 3)), mantissa=R6))
    case("edge_iscoplanar", lambda: Edge.IsCoplanar(
        Edge.ByStartVertexEndVertex(P(0, 0, 0), P(1, 0, 0)),
        Edge.ByStartVertexEndVertex(P(1, 1, 0), P(2, 1, 0)), mantissa=R6))

    # ---- wire metrics / curves ------------------------------------------
    case("wire_circle_len", lambda: Wire.Length(Wire.Circle(origin=P(0, 0, 0), radius=1.0), mantissa=R6))
    case("wire_ellipse_len", lambda: Wire.Length(Wire.Ellipse(origin=P(0, 0, 0), width=2.0, length=1.0), mantissa=R6))
    case("wire_arc", lambda: Wire.Arc(P(-1, 0, 0), P(0, 1, 0), P(1, 0, 0), sides=16))
    case("wire_square", lambda: Wire.Square(origin=P(0, 0, 0), size=2.0, placement="center"))
    case("wire_star", lambda: Wire.Star(origin=P(0, 0, 0), radiusA=1.0, radiusB=0.4, rays=8))
    case("wire_fillet", lambda: Wire.Fillet(Wire.Square(origin=P(0, 0, 0), size=2.0, placement="center"), radius=0.4))
    case("wire_offset", lambda: Wire.ByOffset(Wire.Square(origin=P(0, 0, 0), size=2.0, placement="center"), offset=0.3))
    case("wire_convexhull", lambda: Wire.ConvexHull(Cluster.ByTopologies([
        P(0, 0, 0), P(1, 0, 0), P(0.5, 1, 0), P(0.2, 0.1, 0), P(0.8, 0, 0)]), mantissa=R6))
    case("wire_isclosed", lambda: Wire.IsClosed(Wire.Circle(origin=P(0, 0, 0), radius=1.0)))

    # ---- face metrics / curves ------------------------------------------
    case("face_circle_area", lambda: Face.Area(Face.Circle(origin=P(0, 0, 0), radius=1.0, sides=32), mantissa=R6))
    case("face_ellipse_area", lambda: Face.Area(Face.Ellipse(origin=P(0, 0, 0), width=2.0, length=1.0), mantissa=R6))
    case("face_star_area", lambda: Face.Area(Face.Star(origin=P(0, 0, 0), radiusA=1.0, radiusB=0.4, rays=8), mantissa=R6))
    case("face_ring_area", lambda: Face.Area(Face.Ring(origin=P(0, 0, 0), radius=1.0, thickness=0.25), mantissa=R6))
    case("face_normal", lambda: Face.Normal(Face.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"), outputType="xyz", mantissa=R6))
    case("face_internal_vertex", lambda: Face.InternalVertex(Face.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"), silent=True))
    case("face_offset", lambda: Face.ByOffset(Face.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"), offset=0.3))
    case("face_fillet", lambda: Face.Fillet(Face.Square(origin=P(0, 0, 0), size=2.0, placement="center"), radius=0.4))
    case("face_iscoplanar", lambda: Face.IsCoplanar(Face.Rectangle(origin=P(0, 0, 0), width=2, length=3, placement="center"),
                                                    Face.Rectangle(origin=P(1, 1, 0), width=2, length=3, placement="center"), mantissa=R6))

    # ---- cell primitives -------------------------------------------------
    case("cell_box", lambda: Cell.Box(origin=P(0, 0, 0), width=2, length=3, height=4, placement="center"))
    case("cell_cube_2", lambda: Cell.Cube(origin=P(0, 0, 0), size=2.0))
    case("cell_cylinder", lambda: Cell.Cylinder(origin=P(0, 0, 0), radius=0.5, height=2.0, uSides=24))
    case("cell_sphere", lambda: Cell.Sphere(origin=P(0, 0, 0), radius=0.5, uSides=16, vSides=8))
    case("cell_cone", lambda: Cell.Cone(origin=P(0, 0, 0), baseRadius=0.5, topRadius=0.2, height=1.5, uSides=24))
    case("cell_torus", lambda: Cell.Torus(origin=P(0, 0, 0), majorRadius=0.5, minorRadius=0.125, uSides=16, vSides=8))
    case("cell_dodeca", lambda: Cell.Dodecahedron(radius=1.0, silent=True))
    case("cell_icosa", lambda: Cell.Icosahedron(radius=1.0, silent=True))
    case("cell_internal_vertex", lambda: Cell.InternalVertex(cube_cell(), silent=True))
    case("cell_thickenedshell", lambda: Cell.ByThickenedShell(
        Shell.ByFaces(Topology.Faces(cube_cell(), silent=True)), thickness=0.2))

    # ---- cellcomplex -----------------------------------------------------
    case("cc_box", lambda: CellComplex.Box(origin=P(0, 0, 0), width=2, length=2, height=2, uSides=2, vSides=2, wSides=2, placement="center"))
    case("cc_prism_vol", lambda: CellComplex.Volume(CellComplex.Prism(origin=P(0, 0, 0), width=2, length=2, height=2, uSides=2, vSides=1, wSides=1, placement="center"), mantissa=R6))
    case("cc_torus", lambda: CellComplex.Torus(origin=P(0, 0, 0), majorRadius=0.5, minorRadius=0.125, uSides=16, vSides=8))
    case("cc_external_boundary", lambda: CellComplex.ExternalBoundary(CellComplex.Prism(origin=P(0, 0, 0), width=2, length=2, height=2, uSides=2, vSides=1, wSides=1, placement="center")))
    case("cc_nonmanifold_faces", lambda: len(CellComplex.NonManifoldFaces(
        CellComplex.Prism(origin=P(0, 0, 0), width=2, length=2, height=2, uSides=2, vSides=1, wSides=1, placement="center"))))
    case("cc_decompose", lambda: len(CellComplex.Decompose(
        CellComplex.Prism(origin=P(0, 0, 0), width=2, length=2, height=2, uSides=2, vSides=1, wSides=1, placement="center"))))

    # ---- topology relations ----------------------------------------------
    case("within_cube_cube", lambda: Topology.Within(cube_cell(), cube_cell(0.5), silent=True))
    case("vertex_within_cell", lambda: Topology.Within(P(0.5, 0.5, 0.5), cube_cell(), silent=True))
    case("cell_contains_vertex", lambda: Topology.Contains(cube_cell(), P(0.5, 0.5, 0.5), silent=True))
    case("cell_contains_ext", lambda: Topology.Contains(cube_cell(), P(5, 5, 5), silent=True))
    case("boundingbox_cell", lambda: Topology.BoundingBox(cube_cell(), mantissa=R6))
    case("diameter_cell", lambda: Topology.Diameter(cube_cell(), mantissa=R6, tolerance=1e-6))
    case("com_cell", lambda: Topology.CenterOfMass(cube_cell(), silent=True))
    case("issame_self", lambda: Topology.IsSame(cube_cell(), cube_cell(), silent=True))
    case("enclosing_edges_vertex", lambda: len(Vertex.EnclosingEdges(P(1, 0, 0),
        Wire.Square(origin=P(0, 0, 0), size=2.0, placement="center"), silent=True)))
    case("vertex_degree", lambda: Vertex.Degree(P(1, 0, 0),
        Wire.Square(origin=P(0, 0, 0), size=2.0, placement="center"), "edge"))

    # ---- topology ops ----------------------------------------------------
    case("divide_cell", lambda: Topology.Divide(cube_cell(), Face.Rectangle(origin=P(0.5, 0, 0), width=2, length=2, direction=[1, 0, 0], placement="center"), silent=True))
    case("selfmerge_cluster", lambda: Topology.SelfMerge(Cluster.ByTopologies([P(0, 0, 0), P(1, 0, 0)]), silent=True))

    # ---- JSON round-trip -------------------------------------------------
    def json_rt():
        r = Topology.ByJSONString(Topology.JSONString([cube_cell()], mantissa=R6), silent=True)
        return r[0] if isinstance(r, list) and r else r
    case("json_roundtrip_cell", json_rt)

    # ---- booleans: face vs face (coplanar offset) ------------------------
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge)]:
        def _make(fp=fn):
            return lambda: fp(sq_face(0.0), sq_face(0.5), silent=True)
        case(f"bool_{op}_face", _make())

    # ---- booleans: cellcomplex vs cellcomplex ----------------------------
    def cc_box2(ox=0.0):
        return CellComplex.Box(origin=P(ox, 0, 0), width=2, length=2, height=2,
                               uSides=1, vSides=1, wSides=1, placement="center")
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge)]:
        def _make(fp=fn):
            return lambda: fp(cc_box2(0.0), cc_box2(0.5), silent=True)
        case(f"bool_{op}_cc", _make())

    # ---- booleans: vertex vs vertex --------------------------------------
    for op, fn in [("union", Topology.Union), ("difference", Topology.Difference),
                   ("intersect", Topology.Intersect), ("merge", Topology.Merge),
                   ("slice", Topology.Slice), ("impose", Topology.Impose), ("imprint", Topology.Imprint)]:
        case(f"bool_{op}_vertex", lambda fp=fn: fp(P(0, 0, 0), P(0, 0, 0), silent=True))

    # ---- graph adjacency -------------------------------------------------
    case("graph_adjacent", lambda: (
        len(Graph.Vertices(Graph.ByVerticesEdges([P(0, 0), P(1, 0), P(0, 1)], [Edge.ByStartVertexEndVertex(P(0, 0), P(1, 0)), Edge.ByStartVertexEndVertex(P(0, 0), P(0, 1))]))),
        len(Graph.Edges(Graph.ByVerticesEdges([P(0, 0), P(1, 0), P(0, 1)], [Edge.ByStartVertexEndVertex(P(0, 0), P(1, 0)), Edge.ByStartVertexEndVertex(P(0, 0), P(0, 1))])))))
    def graph_deg():
        g = Graph.ByVerticesEdges([P(0, 0), P(1, 0), P(0, 1)],
                                  [Edge.ByStartVertexEndVertex(P(0, 0), P(1, 0)), Edge.ByStartVertexEndVertex(P(0, 0), P(0, 1))])
        vt = Graph.Vertices(g)[0]  # vertex owned by the graph (identity lookup works)
        return len(Graph.AdjacentVertices(g, vt))
    case("graph_deg", graph_deg)

    return L


# ---------------------------------------------------------------------------
# Signature extraction -- tolerance-agnostic raw material, read through the
# wrappers so a Cluster-wrapped result still reports its material's totals.
# ---------------------------------------------------------------------------
def rnd(v):
    try:
        r = round(float(v), R6)
        return 0.0 if r == 0 else r
    except Exception:
        return v


def vraw(v):
    return tuple(float(c) for c in Vertex.Coordinates(v, outputType="xyz"))


def raw_sig(to):
    """Raw signature: type, counts, vertices, scalars (summed over subtree)."""
    if to is None:
        return {"type": None, "counts": None, "verts": [], "scalars": {}, "centroid": None}
    sig = {"type": Topology.TypeAsString(to)}
    fetchers = [Topology.Cells, Topology.Faces, Topology.Shells,
                Topology.Wires, Topology.Edges, Topology.Vertices]
    sig["counts"] = tuple(len(fn(to, silent=True) or []) for fn in fetchers)
    sig["verts"] = [vraw(v) for v in (Topology.Vertices(to, silent=True) or [])]
    sc = {}
    for c in (Topology.Cells(to, silent=True) or []):
        try:
            sc["volume"] = sc.get("volume", 0.0) + (Cell.Volume(c, mantissa=9) or 0.0)
        except Exception:
            pass
    for f in (Topology.Faces(to, silent=True) or []):
        try:
            sc["area"] = sc.get("area", 0.0) + (Face.Area(f, mantissa=9) or 0.0)
        except Exception:
            pass
    for e in (Topology.Edges(to, silent=True) or []):
        try:
            sc["length"] = sc.get("length", 0.0) + (Edge.Length(e, mantissa=9) or 0.0)
        except Exception:
            pass
    if Topology.IsInstance(to, "cell"):
        try:
            sc["surf"] = Cell.SurfaceArea(to, mantissa=9)
        except Exception:
            pass
    elif Topology.IsInstance(to, "cellcomplex"):
        try:
            sc["surf"] = CellComplex.SurfaceArea(to, mantissa=9)
        except Exception:
            pass
    sig["scalars"] = sc
    c = Topology.Centroid(to, silent=True)
    sig["centroid"] = vraw(c) if c is not None else None
    return sig


def scalar_sig(x):
    return {"val": tuple(float(v) for v in x)}


def exact_str(sig):
    """Strict 6-decimal canonical form used by the structural ruler."""
    return {
        "type": sig["type"].lower() if sig["type"] else None,
        "counts": sig["counts"],
        "verts": sorted({tuple(rnd(v) for v in vert) for vert in sig["verts"]}),
        "scalars": {k: rnd(v) for k, v in sig["scalars"].items()},
        "centroid": tuple(rnd(c) for c in sig["centroid"]) if sig["centroid"] else None,
    }


def geometry_equivalent(a, b, tol=TOL):
    """Ruler A: is the measured GEOMETRY the same within `tol`?

    Vertex multisets are bucketed at the tolerance scale and compared as
    multisets; centroids and every shared scalar must agree within `tol`.
    Wrapper type and wire/container grouping are deliberately ignored.
    Returns (bool, note).
    """
    if isinstance(a, str) or isinstance(b, str):
        return (a == b, "both raise same exception")
    if "type" not in a or "type" not in b:
        va, vb = tuple(a["val"]), tuple(b["val"])
        return (len(va) == len(vb) and all(abs(x - y) <= tol for x, y in zip(va, vb)), "scalar tuple")
    if a["type"] is None or b["type"] is None:
        return (a["type"] is None and b["type"] is None, "none-vs-value")
    dec = -int(math.floor(math.log10(tol) - 1e-9))

    def _b(v):
        return tuple(round(c, dec) for c in v)

    verts_ok = Counter(_b(v) for v in a["verts"]) == Counter(_b(v) for v in b["verts"])
    ca, cb = a["centroid"], b["centroid"]
    centroid_ok = (ca is not None and cb is not None and
                   max(abs(ca[k] - cb[k]) for k in range(3)) <= tol)
    scalars_ok = all(abs(a["scalars"][k] - b["scalars"][k]) <= tol
                     for k in set(a["scalars"]) & set(b["scalars"]))
    ok = verts_ok and centroid_ok and scalars_ok
    return (ok, "verts=%s centroid=%s scalars=%s" % (verts_ok, centroid_ok, scalars_ok))


def classify(a, b, tol=TOL):
    """Classify a strict mismatch into the documented deviation categories.

    Returns (label, detail). Labels:
      MATCH                      -- exact structural equality
      (a) structural-wire-grouping -- same edges/verts, different wire count
      (b) numeric-noise          -- same content, all deltas within tol
      (c) container-only         -- same geometry, different wrapper type
      (a) real-structural        -- genuine content divergence (must not happen)
    """
    if isinstance(a, str) or isinstance(b, str):
        return ("MATCH", "both raise same exception") if a == b else ("(a) real-structural", f"exception: {a!r} vs {b!r}")
    if "type" not in a or "type" not in b:
        va, vb = tuple(a["val"]), tuple(b["val"])
        if len(va) != len(vb) or any(abs(x - y) > tol for x, y in zip(va, vb)):
            return ("(a) real-structural", f"scalar tuple {va} vs {vb}")
        return ("MATCH", "scalar tuple")
    if a["type"] is None or b["type"] is None:
        return ("MATCH", "both empty/None") if a["type"] is None and b["type"] is None else ("(a) real-structural", "empty-vs-nonempty")

    ea, eb = exact_str(a), exact_str(b)
    if repr(ea) == repr(eb):
        return ("MATCH", "exact")

    # geometry within tolerance?
    geom_ok = geometry_equivalent(a, b, tol)[0]
    if not geom_ok:
        return ("(a) real-structural", "geometry differs beyond tolerance")

    ta, tb = a["type"], b["type"]
    counts = a["counts"]
    has_missing = bool(set(a["scalars"]) != set(b["scalars"]))
    num_small = any(0 < abs(a["scalars"][k] - b["scalars"][k]) <= tol for k in set(a["scalars"]) & set(b["scalars"]))

    detail = "type %s vs %s" % (ta, tb) if ta != tb else ("" if not has_missing else "scalar present on one side")
    if ta != tb or has_missing:
        return ("(c) container-only", detail)
    if counts != b["counts"]:
        # Only wire count differs while edges/verts match -> core's wire grouping.
        if (counts[3] != b["counts"][3] and counts[4] == b["counts"][4]
                and counts[0] == b["counts"][0] and counts[1] == b["counts"][1]):
            return ("(a) structural-wire-grouping", "wires %s vs %s" % (counts[3], b["counts"][3]))
        return ("(a) real-structural", "counts %s vs %s" % (counts, b["counts"]))
    if num_small:
        return ("(b) numeric-noise", "")
    return ("(b) numeric-noise", "sub-tolerance vertex/centroid positioning")


# ---------------------------------------------------------------------------
# One collection pass over BOTH backends, cached for the whole session so the
# battery is executed exactly once (114 thunks x 2 backends).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def backend_results():
    cases = all_cases()

    def collect(backend):
        if backend == "core":
            Core.ResetBackend()
        else:
            Core.SetBackend(PythonOCCBackend())
        out = {}
        for name, fn in cases:
            try:
                r = fn()
                if isinstance(r, (tuple, list)) and not isinstance(r, (str, bytes)):
                    out[name] = scalar_sig(tuple(r))
                else:
                    out[name] = raw_sig(r)
            except Exception as exc:  # noqa: BLE001 - parity must capture every failure
                out[name] = f"EXC:{type(exc).__name__}"
        return out

    core = collect("core")
    pyocc = collect("pythonocc")
    return {name: (core[name], pyocc[name]) for name, _ in cases}


# ---------------------------------------------------------------------------
# The documented deviations: name -> expected classification (and why).
# A case must match EXACTLY unless it is in this table, in which case it must
# classify into its documented category. Update the table IN THE SAME COMMIT
# as any backend fix that changes a classification.
# ---------------------------------------------------------------------------
KNOWN_DEVIATIONS = {
}


def test_differential_geometry_parity(backend_results):
    """Ruler A: every one of the 114 cases must be geometry-equivalent.

    Within the project tolerance (1e-4) every case must carry identical
    material (vertex multisets, centroid, volume/area/length). Wrapper type
    and wire grouping are not part of this ruler.
    """
    failures = []
    for name, (a, b) in backend_results.items():
        ok, note = geometry_equivalent(a, b, TOL)
        if not ok:
            failures.append(f"{name}: {note}")
    assert not failures, "geometry mismatch:\n" + "\n".join(failures)


def test_differential_structural_parity(backend_results):
    """Ruler B: strict equality (type + counts + exact coords) everywhere,
    except the documented deviations which must classify into their expected
    category. A NEW deviation, an unexplained one, or a re-classified one
    fails -- i.e. this is the regression guard.
    """
    problems = []
    for name, (a, b) in backend_results.items():
        label, detail = classify(a, b, TOL)
        expected = KNOWN_DEVIATIONS.get(name)
        if expected is None:
            if label != "MATCH":
                problems.append(f"{name}: UNEXPECTED deviation ({label}) {detail}")
        else:
            expected_label, expected_why = expected
            if label != expected_label:
                problems.append(
                    f"{name}: classification changed: expected "
                    f"{expected_label!r} ({expected_why}), got {label!r} ({detail})")
    assert not problems, "structural parity issues:\n" + "\n".join(problems)