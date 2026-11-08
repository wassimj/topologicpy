"""Exact mesh fast paths for wall voids and polygon-bounded roof cuts."""
import pytest
from topologicpy.IFC import IFCFastTopology as N, IFCFastEntity
from topologicpy.Topology import Topology
from topologicpy.Cell import Cell


def prism(outline, low, high):
    n = len(outline)
    vertices = [[x, y, z] for z in (low, high) for x, y in outline]
    faces = [list(reversed(range(n))), list(range(n, 2*n))]
    faces += [[i, (i+1)%n, (i+1)%n+n, i+n] for i in range(n)]
    return {"vertices": vertices, "faces": faces, "edges": []}


def box(x0, y0, z0, x1, y1, z1):
    return prism([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],z0,z1)


def volume(mesh):
    cell = Topology.ByGeometry(vertices=mesh["vertices"], faces=mesh["faces"], topologyType="cell", silent=True)
    assert Topology.IsInstance(cell, "Cell")
    return Cell.Volume(cell, mantissa=7)


def test_through_opening_avoids_native_boolean_and_preserves_volume(monkeypatch):
    host=box(0,0,0,10,.2,3)
    cutter=box(4,-.1,1,6,.3,2)
    with monkeypatch.context() as scope:
        scope.setattr(Topology,"ByGeometry",staticmethod(lambda *a,**k: (_ for _ in ()).throw(AssertionError("Native CSG used"))))
        mesh=N._boolean_mesh(host,cutter,"DIFFERENCE")
    assert volume(mesh)==pytest.approx(5.6,abs=1e-6)


def test_mitered_wall_opening_uses_its_actual_outline():
    host=prism([(0,0),(10,0),(10,.2),(.2,.2)],0,3)
    mesh=N._column_difference(host,box(4,-.1,1,6,.3,2))
    assert mesh is not None
    assert volume(mesh)==pytest.approx(5.54,abs=1e-6)


def test_partial_depth_recess_keeps_its_back_wall():
    host=box(0,0,0,10,.3,3)
    cutter=box(4,-.1,1,6,.1,2)
    assert N._prismatic_difference(host,cutter) is None
    mesh=N._column_difference(host,cutter)
    assert mesh is not None
    assert volume(mesh)==pytest.approx(8.8,abs=1e-6)


def test_convex_plane_clip_is_closed_and_has_correct_side():
    host=box(0,0,0,2,2,2)
    for negative in [False,True]:
        mesh=N._clip_convex_mesh(host,[0,0,1],[0,0,1],negative)
        assert volume(mesh)==pytest.approx(4.,abs=1e-6)


def test_bounded_roof_cut_does_not_clip_outside_its_polygon(monkeypatch):
    e={1:IFCFastEntity(1,"IFCPOLYGONALBOUNDEDHALFSPACE",[("REF",2),False,("REF",3),("REF",4)]),
       2:IFCFastEntity(2,"IFCPLANE",[("REF",5)]),
       3:IFCFastEntity(3,"IFCAXIS2PLACEMENT3D",[("REF",6),None,None]),
       5:IFCFastEntity(5,"IFCAXIS2PLACEMENT3D",[("REF",7),None,None]),
       6:IFCFastEntity(6,"IFCCARTESIANPOINT",[[0.,0.,0.]]),
       7:IFCFastEntity(7,"IFCCARTESIANPOINT",[[0.,0.,1.]]),
       4:IFCFastEntity(4,"IFCPOLYLINE",[[("REF",8),("REF",9),("REF",10),("REF",11),("REF",8)]]),
       8:IFCFastEntity(8,"IFCCARTESIANPOINT",[[-1.,-1.]]),
       9:IFCFastEntity(9,"IFCCARTESIANPOINT",[[2.,-1.]]),
       10:IFCFastEntity(10,"IFCCARTESIANPOINT",[[2.,1.]]),
       11:IFCFastEntity(11,"IFCCARTESIANPOINT",[[-1.,1.]])}
    host=prism([(0,0),(10,0),(10,.2),(.2,.2)],0,3)
    cutter=N._half_space_mesh(e[1],e,N._identity(),1.,host)
    mesh=N._column_difference(host,cutter)
    assert mesh is not None
    assert volume(mesh)==pytest.approx(5.18,abs=1e-6)


def test_enclosed_cavity_uses_the_general_path():
    assert N._column_difference(box(0,0,0,3,3,3),box(1,1,1,2,2,2)) is None


def test_repeated_window_cuts_remain_closed():
    mesh=prism([(0,0),(40,0),(40,.2),(.2,.2)],0,3)
    expected=7.98*3
    for i in range(10):
        mesh=N._boolean_mesh(mesh,box(1+i*3,-.1,1,2+i*3,.3,2),"DIFFERENCE")
        expected-=.2
    assert volume(mesh)==pytest.approx(expected,abs=1e-6)


def test_floor_touching_door_then_window_has_no_degenerate_cap_faces(monkeypatch):
    # Duplex wall #4087: its first cut touches z=0 within floating-point noise.
    # A microscopic cap triangle used to poison the following window cut.
    host=box(0,-17.383,0,.417,-.417,3.1)
    door=box(-.1,-2.247,4.440892098500626e-16,.6,-.997,2.01)
    window=box(-.1,-12.605,.1,.6,-11.855,2.3)
    with monkeypatch.context() as scope:
        scope.setattr(Topology,'ByGeometry',staticmethod(lambda *a,**k: (_ for _ in ()).throw(AssertionError('Native CSG used'))))
        mesh=N._boolean_mesh(host,door,'DIFFERENCE')
        assert N._prism_profile(mesh) is not None
        mesh=N._boolean_mesh(mesh,window,'DIFFERENCE')
    expected=.417*(16.966*3.1-1.25*2.01-.75*2.2)
    assert volume(mesh)==pytest.approx(expected,abs=1e-6)


def test_collinear_faces_alone_are_not_a_solid():
    mesh=N._closed_polygon_mesh([[[0,0,0],[1,0,0],[2,0,1e-16]]])
    assert not mesh['faces']
