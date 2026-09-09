"""Sentinel regression for the pre-workshop Curve/NURBS rollback defect.

Topology.Slice(Face, grid-of-Edges) must produce a coherent shared-topology Shell,
not a malformed collection of faces/edges.
"""

import math

from topologicpy.Cluster import Cluster
from topologicpy.Edge import Edge
from topologicpy.Face import Face
from topologicpy.Topology import Topology
from topologicpy.Vertex import Vertex


def _edge(x1, y1, x2, y2):
    return Edge.ByVertices(
        Vertex.ByCoordinates(float(x1), float(y1), 0.0),
        Vertex.ByCoordinates(float(x2), float(y2), 0.0),
        silent=True,
    )


def test_slice_face_with_grid_edges_produces_well_formed_shell():
    face = Face.Rectangle(width=6.0, length=6.0, placement="lowerleft", silent=True)
    assert Topology.IsInstance(face, "Face")

    # A 2 x 2 set of full-span interior cutters divides the face into a 3 x 3 grid.
    cutters = [
        _edge(2.0, -1.0, 2.0, 7.0),
        _edge(4.0, -1.0, 4.0, 7.0),
        _edge(-1.0, 2.0, 7.0, 2.0),
        _edge(-1.0, 4.0, 7.0, 4.0),
    ]
    assert all(Topology.IsInstance(edge, "Edge") for edge in cutters)
    grid = Cluster.ByTopologies(cutters, silent=True)
    assert Topology.IsInstance(grid, "Cluster")

    sliced = Topology.Slice(face, grid, tolerance=1.0e-4, silent=True)

    assert Topology.IsInstance(sliced, "Shell")
    faces = Topology.Faces(sliced, silent=True)
    edges = Topology.Edges(sliced, silent=True)
    vertices = Topology.Vertices(sliced, silent=True)

    # A coherent 3 x 3 planar subdivision has 9 faces, 24 shared edges and 16
    # shared vertices. These counts catch duplicated/unshared subtopologies.
    assert len(faces) == 9
    assert len(edges) == 24
    assert len(vertices) == 16

    areas = [Face.Area(f, mantissa=9) for f in faces]
    assert all(area is not None and area > 0.0 for area in areas)
    assert all(math.isclose(area, 4.0, rel_tol=1.0e-7, abs_tol=1.0e-7) for area in areas)
    assert math.isclose(sum(areas), 36.0, rel_tol=1.0e-7, abs_tol=1.0e-7)

    # Euler characteristic for a connected planar disk: V - E + F = 1.
    assert len(vertices) - len(edges) + len(faces) == 1
