from unittest.mock import patch
import pytest
from topologicpy.Cluster import Cluster
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.TGraph import TGraph
from topologicpy.Core import Core


@pytest.mark.pythonocc_only
def test_direct_members_preserve_intersecting_independent_cell():
    cc = CellComplex.Prism(uSides=2, vSides=1, wSides=1)
    independent = Cell.Prism(width=.5, length=.5, height=.5)
    cluster = Core.Cluster.ByTopologies([cc, independent, independent], False)
    with patch.object(Cluster, '_ReconstructTopologies', side_effect=AssertionError('descendant scan')):
        members = Cluster.Topologies(cluster)
        assert len(members) == 2
        assert Cluster.FreeCells(cluster) == [independent]
        assert Cluster.FreeFaces(cluster) == []


@pytest.mark.pythonocc_only
def test_shape_only_cluster_recovers_direct_members():
    from topologicpy.pythonocc_backend.cluster import Cluster as NativeCluster
    a = Cell.Prism()
    b = Vertex.ByCoordinates(10, 0, 0)
    original = Core.Cluster.ByTopologies([a, b], False)
    loaded = NativeCluster(shape=original.GetOcctShape(), topologies=[])
    with patch.object(Cluster, '_ReconstructTopologies', side_effect=AssertionError('descendant scan')):
        members = Cluster.Topologies(loaded)
    assert sorted(Topology.TypeAsString(x) for x in members) == ['Cell', 'Vertex']


@pytest.mark.pythonocc_only
def test_nested_cluster_preserves_hierarchy():
    inner = Core.Cluster.ByTopologies([Cell.Prism()], False)
    outer = Core.Cluster.ByTopologies([inner, Vertex.ByCoordinates(10, 0, 0)], False)
    assert Cluster.Topologies(outer)[0] is inner
    assert Cluster.FreeCells(outer) == []


@pytest.mark.pythonocc_only
@pytest.mark.parametrize('roundtrip', [False, True])
def test_graph_single_decomposition_preserves_adjacency(tmp_path, roundtrip):
    cc = CellComplex.Prism(uSides=3, vSides=2, wSides=1)
    expected = TGraph.ByTopology(cc)
    cluster = Core.Cluster.ByTopologies([cc], False)
    if roundtrip:
        target = str(tmp_path/'graph.tpy')
        assert Topology.ExportToTPY(cluster, target, overwrite=True)
        cluster = Topology.ByTPYPath(target)
    with patch.object(Cluster, 'Topologies', wraps=Cluster.Topologies) as query, \
         patch.object(Cluster, '_ReconstructTopologies', side_effect=AssertionError('descendant scan')):
        graph = TGraph.ByTopology(cluster)
    assert query.call_count == 1
    assert TGraph.Order(graph) == TGraph.Order(expected) == 6
    assert TGraph.Size(graph) == TGraph.Size(expected) == 7


def test_other_backend_retains_reconstruction():
    sentinel = object()
    with patch.object(Topology, 'IsInstance', return_value=True), \
         patch.object(Core, 'InstanceCall', side_effect=TypeError), \
         patch.object(Cluster, '_ReconstructTopologies', return_value=[sentinel]) as fallback:
        assert Cluster.Topologies(object()) == [sentinel]
    assert fallback.call_count == 1
