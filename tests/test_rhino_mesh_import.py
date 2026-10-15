"""Rhino Mesh facets are materialized through the TopologicPy public importer."""
import pytest
rhino3dm=pytest.importorskip('rhino3dm')
from topologicpy.Rhino import Rhino
from topologicpy.Topology import Topology
from topologicpy.Face import Face


def mesh():
    result=rhino3dm.Mesh()
    for point in [(0,0,0),(4,0,0),(4,3,0),(0,3,0)]:result.Vertices.Add(*point)
    result.Faces.AddFace(0,1,2,3)
    return result


def test_mesh_quad_import():
    topology=Rhino.ByMesh(mesh())
    assert Topology.IsInstance(topology,'Face')
    assert abs(Face.Area(topology)-12)<1e-6


def test_mesh_file_import_preserves_attributes_and_source(tmp_path):
    file=rhino3dm.File3dm()
    attributes=rhino3dm.ObjectAttributes();attributes.Name='floor_mesh'
    file.Objects.AddMesh(mesh(),attributes)
    path=tmp_path/'mesh.3dm';assert file.Write(str(path),8)
    items=Topology.By3DMPath(str(path))
    assert len(items)==1 and Topology.IsInstance(items[0],'Face')
    from topologicpy.Dictionary import Dictionary
    assert Dictionary.ValueAtKey(Topology.Dictionary(items[0]),'name')=='floor_mesh'
    exported=tmp_path/'roundtrip.3dm';assert Rhino.ExportTo3DM(items,str(exported))
    assert exported.read_bytes()==path.read_bytes()
