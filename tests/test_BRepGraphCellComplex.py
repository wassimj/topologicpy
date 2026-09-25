import os

import pytest

pytest.importorskip("OCC.Core.BRepGraph")

if os.environ.get("TOPOLOGICPY_CORE_BACKEND", "").strip().lower() != "pythonocc":
    pytest.skip("BRepGraph Tranche 2 tests require the pythonocc backend", allow_module_level=True)

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_SHELL
from OCC.Core.TopExp import TopExp_Explorer

from topologicpy.CellComplex import CellComplex
from topologicpy.Topology import Topology
from topologicpy.pythonocc_backend._brepgraph import cached_index
from topologicpy.pythonocc_backend.topology import Topology as BackendTopology
from topologicpy.pythonocc_backend.shell import Shell as BackendShell


def _native_shape(topology):
    return getattr(topology, "shape", None)


def _same_shape_set(left, right):
    left_shapes = [_native_shape(item) for item in left]
    right_shapes = [_native_shape(item) for item in right]
    assert len(left_shapes) == len(right_shapes)
    unmatched = list(right_shapes)
    for shape in left_shapes:
        for i, candidate in enumerate(unmatched):
            try:
                same = bool(shape.IsSame(candidate))
            except Exception:
                same = False
            if same:
                unmatched.pop(i)
                break
        else:
            return False
    return not unmatched


def _octahedron():
    cc = CellComplex.Octahedron(radius=0.5)
    assert Topology.IsInstance(cc, "CellComplex")
    return cc


def test_face_solid_incidence_classifies_octahedron_boundaries():
    cc = _octahedron()
    index = cached_index(cc, _native_shape(cc))
    assert index is not None and index.valid

    external = index.subshapes_by_ancestor_count(
        TopAbs_FACE,
        TopAbs_SOLID,
        min_count=1,
        max_count=1,
    )
    internal = index.subshapes_by_ancestor_count(
        TopAbs_FACE,
        TopAbs_SOLID,
        min_count=2,
        max_count=None,
    )

    assert external is not None and len(external) == 8
    assert internal is not None and len(internal) == 1


def test_cellcomplex_nonmanifold_faces_use_native_incidence():
    cc = _octahedron()
    internal = CellComplex.NonManifoldFaces(cc)
    internal_boundaries = CellComplex.InternalFaces(cc)

    assert isinstance(internal, list)
    assert isinstance(internal_boundaries, list)
    assert len(internal) == 1
    assert len(internal_boundaries) == 1
    assert _same_shape_set(internal, internal_boundaries)


def test_cellcomplex_brepgraph_and_legacy_nonmanifold_semantics_match():
    cc = _octahedron()

    old = os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
    try:
        graph_result = CellComplex.NonManifoldFaces(cc)
        os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = "1"
        legacy_result = CellComplex.NonManifoldFaces(cc)
    finally:
        if old is None:
            os.environ.pop("TOPOLOGICPY_DISABLE_BREPGRAPH", None)
        else:
            os.environ["TOPOLOGICPY_DISABLE_BREPGRAPH"] = old

    assert isinstance(graph_result, list)
    assert isinstance(legacy_result, list)
    assert len(graph_result) == 1
    assert len(legacy_result) == 1
    assert _same_shape_set(graph_result, legacy_result)


def test_cell_adjacency_remains_exact_on_shared_face():
    cc = _octahedron()
    cells = CellComplex.Cells(cc)
    assert len(cells) == 2

    adjacent_0 = Topology.AdjacentTopologies(
        cells[0], hostTopology=cc, topologyType="cell", silent=True
    )
    adjacent_1 = Topology.AdjacentTopologies(
        cells[1], hostTopology=cc, topologyType="cell", silent=True
    )

    assert isinstance(adjacent_0, list) and len(adjacent_0) == 1
    assert isinstance(adjacent_1, list) and len(adjacent_1) == 1
    assert _native_shape(adjacent_0[0]).IsSame(_native_shape(cells[1]))
    assert _native_shape(adjacent_1[0]).IsSame(_native_shape(cells[0]))


def test_coedge_incidence_distinguishes_cylinder_seam_from_free_boundary():
    solid = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    explorer = TopExp_Explorer(solid, TopAbs_SHELL)
    assert explorer.More()
    shell_shape = explorer.Current()

    shell = BackendTopology.ByOcctShape(shell_shape)
    assert isinstance(shell, BackendShell)

    index = cached_index(shell, shell_shape)
    assert index is not None and index.valid

    seam_edges = index.edge_shapes_by_incidence(
        min_use_count=2,
        max_use_count=None,
        min_face_count=1,
        max_face_count=1,
    )
    free_edges = index.edge_shapes_by_incidence(
        min_use_count=1,
        max_use_count=1,
    )

    assert seam_edges is not None and len(seam_edges) >= 1
    assert free_edges is not None and len(free_edges) == 0

    seam_info = index.edge_incidence(seam_edges[0])
    assert seam_info is not None
    assert seam_info["use_count"] >= 2
    assert seam_info["face_count"] == 1

    # The complete cylindrical shell is closed.  A face-count-only algorithm
    # commonly mistakes the periodic seam for a free edge; CoEdge use count
    # makes the correct distinction.
    assert shell.IsClosed() is True
