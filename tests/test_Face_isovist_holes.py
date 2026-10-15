"""Visibility voids are transparent; both voids and solids exclude viewpoints."""
import pytest
from topologicpy.Face import Face
from topologicpy.Vertex import Vertex
from topologicpy.Topology import Topology
from topologicpy.Dictionary import Dictionary


def room():
    return Face.Rectangle(width=10,length=10)


def wire(x=0,y=0,width=2,length=2):
    return Face.ExternalBoundary(Face.Rectangle(origin=Vertex.ByCoordinates(x,y,0),width=width,length=length))


def value(topology,key):
    return Dictionary.ValueAtKey(Topology.Dictionary(topology),key,silent=True)


def inside(face,point):
    return Vertex.IsInternal(Vertex.ByCoordinates(*point),face,silent=True)


def test_explicit_hole_is_visible_through_and_visible_area_includes_void():
    boundary=room();void=wire();source=Vertex.ByCoordinates(-3,0,0)
    iso=Face.Isovist(boundary,source,holes=[void],metrics=True,silent=True)
    opaque=Face.Isovist(boundary,source,obstacles=[void],metrics=True,silent=True)
    assert iso is not None and opaque is not None
    assert Face.Area(iso)==pytest.approx(100,abs=.001)
    assert Face.Area(opaque)<Face.Area(iso)
    assert inside(iso,[4,0,0]) and inside(iso,[0,0,0])
    assert not inside(opaque,[4,0,0])
    assert not Face.InternalBoundaries(iso)
    assert value(iso,'occluded_length')==0
    assert value(iso,'closed_perimeter')==pytest.approx(40,abs=.001)


@pytest.mark.parametrize('role',['hole','obstacle'])
@pytest.mark.parametrize('point',[(0,0,0),(1,0,0)])
def test_explicit_exclusions_reject_sources_including_boundary(role,point):
    args={'holes' if role=='hole' else 'obstacles':[wire()]}
    assert Face.Isovist(room(),Vertex.ByCoordinates(*point),silent=True,**args) is None


@pytest.mark.parametrize('role',['hole','obstacle'])
def test_internal_boundary_default_and_source_rejection(role):
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire()])
    iso=Face.Isovist(boundary,Vertex.ByCoordinates(-3,0,0),defaultBoundaryType=role,silent=True)
    assert iso is not None
    assert (Face.Area(iso)==pytest.approx(100,abs=.001))==(role=='hole')
    assert Face.Isovist(boundary,Vertex.ByCoordinates(0,0,0),defaultBoundaryType=role,silent=True) is None
    assert Face.Isovist(boundary,Vertex.ByCoordinates(1,0,0),defaultBoundaryType=role,silent=True) is None


@pytest.mark.parametrize('default',['hole','obstacle'])
def test_mixed_wire_labels_override_default_with_custom_key(default):
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire(x=-1.5),wire(x=2,width=1,length=1)])
    for inner in Face.InternalBoundaries(boundary):
        role='HoLe' if Vertex.X(Topology.Centroid(inner))<0 else 'OBSTACLE'
        Topology.SetDictionary(inner,Dictionary.ByPythonDictionary({'vision':role}))
    iso=Face.Isovist(boundary,Vertex.ByCoordinates(-4,0,0),boundaryTypeKey='vision',defaultBoundaryType=default,silent=True)
    control=Face.Isovist(room(),Vertex.ByCoordinates(-4,0,0),obstacles=[wire(x=2,width=1,length=1)],silent=True)
    assert iso is not None and control is not None
    assert Face.Area(iso)==pytest.approx(Face.Area(control),abs=.001)
    assert inside(iso,[-1.5,0,0])
    assert not inside(iso,[4,0,0])


def test_labels_set_on_input_wire_survive_face_construction():
    void=wire()
    Topology.SetDictionary(void,Dictionary.ByPythonDictionary({'isovistType':'hole'}))
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[void]);before=Face.Area(boundary)
    iso=Face.Isovist(boundary,Vertex.ByCoordinates(-3,0,0),silent=True)
    assert iso is not None and Face.Area(iso)==pytest.approx(100,abs=.001)
    assert Face.Area(boundary)==before
    assert len(Face.InternalBoundaries(boundary))==1
    assert value(Face.InternalBoundaries(boundary)[0],'isovistType')=='hole'


def test_explicit_lists_determine_roles_and_obstacles_win_overlaps():
    void=wire();Topology.SetDictionary(void,Dictionary.ByPythonDictionary({'isovistType':'obstacle'}))
    source=Vertex.ByCoordinates(-3,0,0)
    transparent=Face.Isovist(room(),source,holes=[void],silent=True)
    mixed=Face.Isovist(room(),source,holes=[void],obstacles=[void],silent=True)
    opaque=Face.Isovist(room(),source,obstacles=[void],silent=True)
    assert Face.Area(transparent)==pytest.approx(100,abs=.001)
    assert Face.Area(mixed)==pytest.approx(Face.Area(opaque),abs=.001)


def test_transparent_edges_do_not_transfer_as_physical_walls():
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire()])
    for edge in Topology.Edges(Face.InternalBoundaries(boundary)[0]):
        Topology.SetDictionary(edge,Dictionary.ByPythonDictionary({'wall_tag':'void'}))
    iso=Face.Isovist(boundary,Vertex.ByCoordinates(-3,0,0),defaultBoundaryType='hole',metrics=True,transferDictionaries=True,silent=True)
    assert value(iso,'occlusivity')==0
    assert all(value(edge,'wall_tag') is None for edge in Topology.Edges(iso))


def test_tilted_face_uses_wire_labels_before_flattening():
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire()])
    boundary=Topology.Rotate(boundary,origin=Vertex.Origin(),axis=[1,0,0],angle=37)
    boundary=Topology.Translate(boundary,x=10,y=20,z=30)
    for inner in Face.InternalBoundaries(boundary):
        Topology.SetDictionary(inner,Dictionary.ByPythonDictionary({'isovistType':'hole'}))
    source=Topology.Rotate(Vertex.ByCoordinates(-3,0,0),origin=Vertex.Origin(),axis=[1,0,0],angle=37)
    source=Topology.Translate(source,x=10,y=20,z=30)
    iso=Face.Isovist(boundary,source,metrics=True,silent=True)
    assert iso is not None and Face.Area(iso)==pytest.approx(100,abs=.002)
    # Existing isovists may reverse winding; the geometric plane must agree.
    assert abs(sum(a*b for a,b in zip(Face.Normal(iso),Face.Normal(boundary))))==pytest.approx(1,abs=1e-4)
    normal=Face.Normal(boundary)
    assert all(abs(sum((c-o)*n for c,o,n in zip(Vertex.Coordinates(v),[10,20,30],normal)))<1e-4 for v in Topology.Vertices(iso))
    assert Face.Isovist(boundary,Vertex.ByCoordinates(10,20,30),silent=True) is None


def test_hole_with_range_and_field_of_view_matches_unperforated_room():
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire()]);source=Vertex.ByCoordinates(-3,0,0)
    options={'direction':[1,0,0],'fov':90,'radius':6,'metrics':True,'silent':True}
    iso=Face.Isovist(boundary,source,defaultBoundaryType='hole',**options)
    control=Face.Isovist(room(),source,**options)
    assert iso is not None and control is not None
    assert Face.Area(iso)==pytest.approx(Face.Area(control),abs=.001)
    assert value(iso,'range_limited_length')==pytest.approx(value(control,'range_limited_length'),abs=.001)


@pytest.mark.parametrize('options',[{'defaultBoundaryType':'void'},{'defaultBoundaryType':None},{'boundaryTypeKey':''},{'boundaryTypeKey':None}])
def test_invalid_boundary_type_options(options):
    assert Face.Isovist(room(),Vertex.ByCoordinates(-3,0,0),silent=True,**options) is None


def test_invalid_wire_label_is_reported(capsys):
    boundary=Face.ByWires(Face.ExternalBoundary(room()),[wire()])
    Topology.SetDictionary(Face.InternalBoundaries(boundary)[0],Dictionary.ByPythonDictionary({'isovistType':'typo'}))
    assert Face.Isovist(boundary,Vertex.ByCoordinates(-3,0,0)) is None
    assert 'boundary type' in capsys.readouterr().out


@pytest.mark.parametrize('explicit',[False,True])
@pytest.mark.parametrize('role',['hole','obstacle'])
def test_small_exclusion_still_rejects_viewpoint(explicit,role):
    tiny=wire(width=.005,length=.005)
    boundary=room() if explicit else Face.ByWires(Face.ExternalBoundary(room()),[tiny])
    options=({'holes' if role=='hole' else 'obstacles':[tiny]} if explicit else {'defaultBoundaryType':role})
    assert Face.Isovist(boundary,Vertex.ByCoordinates(0,0,0),silent=True,**options) is None
