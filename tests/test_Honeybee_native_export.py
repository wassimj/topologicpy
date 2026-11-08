"""Native HBJSON regressions; skipped when optional energy packages are absent."""
import importlib.util
import re
import pytest

pytestmark=pytest.mark.skipif(any(importlib.util.find_spec(n) is None for n in ('honeybee','honeybee_energy','honeybee_radiance')),reason='Optional Honeybee dependencies are not installed')

def test_identifiers_and_display_names():
    from topologicpy.Cell import Cell
    from topologicpy.Cluster import Cluster
    from topologicpy.Topology import Topology
    from topologicpy.Honeybee import Honeybee
    from topologicpy.Vertex import Vertex
    cells=[]
    names=['Room with spaces','Room with spaces','ห้อง','A very long room name '*20]
    for i,name in enumerate(names):
        cell=Cell.Prism(origin=Vertex.ByCoordinates(i*6,0,0),width=4,length=4,height=3)
        Topology.SetDictionary(cell,{'TOPOLOGIC_name':name});cells.append(cell)
    model=Honeybee.ModelByTopology(Cluster.ByTopologies(cells),buildingName='Native_test')
    rooms=model.to_dict()['rooms'];identifiers=[r['identifier'] for r in rooms]
    assert len(set(identifiers))==4
    assert all(re.fullmatch(r'[.A-Za-z0-9_-]{1,100}',i) for i in identifiers)
    assert sorted(r['display_name'] for r in rooms)==sorted(names)
    assert all(len(f['identifier'])<=100 for r in rooms for f in r['faces'])

@pytest.mark.parametrize('kind',['door','untyped_floor'])
def test_single_cell_native_subfaces(kind):
    from topologicpy.Cell import Cell
    from topologicpy.Topology import Topology
    from topologicpy.Honeybee import Honeybee
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    from topologicpy.Context import Context
    from topologicpy.Aperture import Aperture
    cell=Cell.Prism(width=4,length=4,height=3)
    Topology.SetDictionary(cell,{'TOPOLOGIC_name':'Single room'})
    if kind=='door':
        host=next(f for f in Topology.Faces(cell) if abs(Face.Normal(f)[0])>.9)
        x,y,z=Vertex.Coordinates(Topology.Centroid(host))
        points=[[x,y-.4,z-1],[x,y+.4,z-1],[x,y+.4,z+1],[x,y-.4,z+1]]
    else:
        host=next(f for f in Topology.Faces(cell) if Face.Normal(f)[2]<-.9)
        x,y,z=Vertex.Coordinates(Topology.Centroid(host))
        points=[[x-.4,y-.4,z],[x+.4,y-.4,z],[x+.4,y+.4,z],[x-.4,y+.4,z]]
    subface=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points])
    if kind=='door':Topology.SetDictionary(subface,{'kind':'door'})
    assert Aperture.ByTopologyContext(subface,Context.ByTopologyParameters(host,.5,.5,.5)) is not None
    data=Honeybee.ModelByTopology(cell,buildingName='Single_cell').to_dict()
    assert len(data['rooms'])==1
    faces=data['rooms'][0]['faces']
    assert sum(len(f.get('doors' if kind=='door' else 'apertures',[])) for f in faces)==1
