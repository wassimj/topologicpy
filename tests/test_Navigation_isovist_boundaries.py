"""Navigation eye-level faces distinguish floor voids from visual obstructions."""
from pathlib import Path
import pytest
from topologicpy.Navigation import Navigation
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary


def rectangle(x=0,y=0,w=10,h=10,z=0):
    return Face.ByVertices([Vertex.ByCoordinates(a,b,z) for a,b in [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]])


def floor(z=0):
    return Face.ByWires(Face.ExternalBoundary(rectangle(z=z)),[Face.ExternalBoundary(rectangle(x=2,y=4,w=2,h=2,z=z))])


def roles(face,key='isovistType'):
    return [Dictionary.ValueAtKey(Topology.Dictionary(w),key,silent=True) for w in Face.InternalBoundaries(face)]


def sees(face,point):
    return Vertex.IsInternal(Vertex.ByCoordinates(*point),face,silent=True)


def test_floor_void_is_transparent_but_column_blocks_and_both_reject_sources():
    model=Navigation.ByFaces([floor()],obstacles=[rectangle(x=7,y=1,w=1,h=1)])
    eye=Navigation.IsovistFaces(model)[0]
    assert sorted(roles(eye))==['hole','obstacle']
    source=Vertex.ByCoordinates(1,5,1.6)
    iso=Face.Isovist(eye,source,metrics=True,silent=True)
    control=Face.Isovist(rectangle(z=1.6),source,obstacles=[Face.ExternalBoundary(rectangle(x=7,y=1,w=1,h=1,z=1.6))],metrics=True,silent=True)
    assert iso is not None and control is not None
    assert Face.Area(iso)==pytest.approx(Face.Area(control),abs=.001)
    assert sees(iso,[5,5,1.6]) and sees(iso,[3,5,1.6])
    assert Face.Isovist(eye,Vertex.ByCoordinates(3,5,1.6),silent=True) is None
    assert Face.Isovist(eye,Vertex.ByCoordinates(7.5,1.5,1.6),silent=True) is None


def test_classification_does_not_mutate_walking_faces_or_make_void_walkable():
    model=Navigation.ByFaces([floor()])
    original=Face.InternalBoundaries(model.faces[0])[0]
    assert Dictionary.ValueAtKey(Topology.Dictionary(original),'isovistType',silent=True) is None
    eye=Navigation.IsovistFaces(model)[0]
    assert roles(eye)==['hole']
    assert Face.Area(eye)==pytest.approx(96)
    iso=Face.Isovist(eye,Vertex.ByCoordinates(1,5,1.6),silent=True)
    assert Face.Area(iso)==pytest.approx(100,abs=.001)
    graph=Navigation.Graph(model)
    assert sum(Face.Area(f) for f in Navigation.Tiles(model,graph))==pytest.approx(96)
    assert Dictionary.ValueAtKey(Topology.Dictionary(original),'isovistType',silent=True) is None


def test_merged_boundary_is_opaque_even_when_obstacle_misses_its_centroid():
    walking=Face.ByWires(Face.ExternalBoundary(rectangle(w=12)),[Face.ExternalBoundary(rectangle(x=2,y=4,w=7,h=2))])
    model=Navigation.ByFaces([walking],obstacles=[rectangle(x=8.5,y=4.5,w=1,h=1)])
    eye=Navigation.IsovistFaces(model)[0]
    assert roles(eye)==['obstacle']
    loop=Face.ByWire(Face.InternalBoundaries(eye)[0])
    assert Vertex.X(Topology.Centroid(loop))<8.5


def test_void_filled_by_another_slab_is_not_an_internal_boundary():
    model=Navigation.ByFaces([floor(),rectangle(x=2,y=4,w=2,h=2)],storeyIDs=['a','a'])
    eye=Navigation.IsovistFaces(model)[0]
    assert roles(eye)==[]
    assert Face.Area(eye)==pytest.approx(100)


def test_obstacles_are_storey_specific_and_custom_key_is_consumed():
    model=Navigation.ByFaces([floor(),floor(z=3)],obstacles=[rectangle(x=7,y=1,w=1,h=1,z=3)],storeyIDs=['lower','upper'])
    low=Navigation.IsovistFaces(model,storey='lower',boundaryTypeKey='vision')[0]
    upper=Navigation.IsovistFaces(model,storey='upper',boundaryTypeKey='vision')[0]
    assert roles(low,'vision')==['hole']
    assert sorted(roles(upper,'vision'))==['hole','obstacle']
    assert roles(low)==[None]
    iso=Face.Isovist(low,Vertex.ByCoordinates(1,5,1.6),boundaryTypeKey='vision',silent=True)
    assert Face.Area(iso)==pytest.approx(100,abs=.001)


def solid_box(x,y,z,X,Y,Z):
    return {'vertices':[[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],
            'triangles':[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],
            'bounds':[[x,y,z],[X,Y,Z]],'source_id':1}


def test_retained_ifc_solids_are_resectioned_at_requested_eye_height():
    model=Navigation.ByFaces([floor()])
    # Same retained-solid structure supplied by IFC.NavigationModel.
    model._ifc={'_obstacle_meshes':[solid_box(7,1,0,8,2,1)],'_void_meshes':{},'_obstacle_faces':()}
    low=Navigation.IsovistFaces(model,eyeHeight=.5)[0]
    high=Navigation.IsovistFaces(model,eyeHeight=1.6)[0]
    assert sorted(roles(low))==['hole','obstacle']
    assert roles(high)==['hole']


def test_ifc_void_relations_are_subtracted_before_classification():
    model=Navigation.ByFaces([floor()])
    # A full-height opening consumes this obstacle section entirely.
    model._ifc={'_obstacle_meshes':[solid_box(7,1,0,8,2,3)],'_void_meshes':{1:[solid_box(7,1,0,8,2,3)]},'_obstacle_faces':()}
    assert roles(Navigation.IsovistFaces(model)[0])==['hole']


def test_real_ifc_boundaries_all_have_supported_roles():
    model=Navigation.ByIFCPath(Path(__file__).parent/'fixtures'/'navigation-three-run.ifc')
    eyes=Navigation.IsovistFaces(model)
    labels=[role for eye in eyes for role in roles(eye)]
    assert labels and 'obstacle' in labels
    assert set(labels).issubset({'hole','obstacle'})
    assert {round(Vertex.Z(v),3) for eye in eyes for v in Topology.Vertices(eye)}=={1.6,2.8,3.4,4.6,4.75}


@pytest.mark.parametrize('key',['',None,42])
def test_invalid_boundary_key_is_rejected(key):
    with pytest.raises(ValueError,match='boundaryTypeKey'):
        Navigation.IsovistFaces(Navigation.ByFaces([floor()]),boundaryTypeKey=key)
