"""Pattern geometry, partitions, projection and compatibility regressions."""
import math
import pytest
from topologicpy.Grid import Grid
from topologicpy.Face import Face
from topologicpy.Wire import Wire
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Topology import Topology


def coords(vertex):
    return Vertex.Coordinates(vertex,mantissa=10)


@pytest.mark.parametrize("pattern",["rectangular","hexagonal","diagrid","polar"])
def test_pattern_outputs_partition_area(pattern):
    args=dict(width=6,length=4,gridType=pattern,spacing=1.2,approximate=True)
    grid=Grid.OnPlane(**args)
    assert grid is not None
    assert Topology.Edges(grid)
    vertices=Grid.OnPlane(**args,output="vertices")
    assert vertices and all(Topology.IsInstance(v,"Vertex") for v in vertices)
    assert all(abs(coords(v)[2])<1e-7 for v in vertices)
    faces=Grid.OnPlane(**args,output="faces")
    assert len(faces)>1
    assert all(Topology.IsInstance(f,"Face") for f in faces)
    assert sum(Face.Area(f,mantissa=10) for f in faces)==pytest.approx(24,abs=1e-5)


def test_rectangular_independent_spacing_and_translation():
    verts=Grid.OnPlane(width=8,length=6,uSpacing=2,vSpacing=3,
                       origin=[100,200,5],output="vertices")
    points=[coords(v) for v in verts]
    assert {round(p[0]-100,5) for p in points}=={-4,-2,0,2,4}
    assert {round(p[1]-200,5) for p in points}=={-3,0,3}
    assert all(abs(p[2]-5)<1e-7 for p in points)


def test_hexagons_are_regular_and_both_orientations_work():
    for orientation in ("flat-top","pointy-top"):
        faces=Grid.OnPlane(width=9,length=9,gridType="hexagonal",cellSize=1,
                           orientation=orientation,output="faces")
        regular=[f for f in faces if abs(Face.Area(f)-3*math.sqrt(3)/2)<1e-4]
        assert regular
        edges=Topology.Edges(regular[0])
        assert len(edges)==6
        assert all(Edge.Length(e)==pytest.approx(1,abs=1e-4) for e in edges)


def test_diagrid_has_two_diagonal_families():
    grid=Grid.OnPlane(width=5,length=5,gridType="diagrid",angle=30)
    for edge in Topology.Edges(grid):
        a,b=map(coords,Edge.Vertices(edge))
        assert abs((b[1]-a[1])/(b[0]-a[0]))==pytest.approx(math.tan(math.pi/6),abs=1e-4)


def test_holes_clearance_and_faces_keep_domain_area():
    outer=Wire.Rectangle(width=8,length=6)
    hole=Wire.Rectangle(width=2,length=2)
    host=Face.ByWires(outer,[hole])
    faces=Grid.OnFace(host,gridType="hexagonal",cellSize=.8,output="faces",clearance=.25)
    expected=Face.ByOffset(host,offset=.25,smooth=False,silent=True)
    assert sum(Face.Area(f) for f in faces)==pytest.approx(Face.Area(expected),abs=1e-4)
    verts=Grid.OnFace(host,gridType="rectangular",output="vertices",clearance=.25)
    for v in verts:
        x,y,z=coords(v)
        assert abs(x)<=3.75+1e-5 and abs(y)<=2.75+1e-5
        assert not (abs(x)<1.25-1e-5 and abs(y)<1.25-1e-5)


def test_local_and_xy_projection_on_sloping_face():
    host=Face.ByVertices([Vertex.ByCoordinates(*p) for p in
                         [(-3,-2,-3),(3,-2,3),(3,2,3),(-3,2,-3)]])
    for projection in ("local","xy"):
        vertices=Grid.OnFace(host,gridType="rectangular",projection=projection,output="vertices",origin=[0,0,0])
        assert vertices
        assert all(abs(coords(v)[0]-coords(v)[2])<1e-5 for v in vertices)
    edges=Grid.OnFace(host,gridType="rectangular",projection="xy",origin=[0,0,0])
    vertical=[e for e in Topology.Edges(edges) if abs(coords(Edge.StartVertex(e))[0]-coords(Edge.EndVertex(e))[0])<1e-6]
    assert {round(coords(Edge.StartVertex(e))[0],5) for e in vertical}==set(range(-3,4))


def test_plane_normal_and_rotation():
    vertices=Grid.OnPlane(width=4,length=4,direction=[1,0,0],origin=[2,3,4],rotation=25,output="vertices")
    assert vertices and all(abs(coords(v)[0]-2)<1e-5 for v in vertices)


@pytest.mark.parametrize("kwargs",[
    {"gridType":"unknown"},{"spacing":0},{"spacing":float("nan")},
    {"clearance":-1},{"gridType":"diagrid","angle":90},
    {"gridType":"polar","angularSpacing":17},
    {"gridType":"hexagonal","orientation":"unknown"},
    {"output":"points"},{"output":"tiles"},{"direction":[0,0,0]},
    {"width":-1},{"gridType":"polar","angularSpacing":0},
])
def test_invalid_inputs_respect_silent(kwargs,capsys):
    assert Grid.OnPlane(**kwargs,silent=True) is None
    assert not capsys.readouterr().out
    assert Grid.OnPlane(**kwargs,silent=False) is None
    assert "Error" in capsys.readouterr().out


def test_existing_rectangular_calls_remain_edges():
    host=Face.Rectangle(width=4,length=4)
    grid=Grid.OnFace(host,spacing=1,uAlignment="start",vAlignment="start")
    assert grid is not None and Topology.Edges(grid)


def test_full_erosion_and_invalid_projection():
    assert Grid.OnPlane(width=1,length=1,clearance=2,output="faces",silent=True)==[]
    vertical=Face.Rectangle(width=3,length=3,direction=[1,0,0])
    assert Grid.OnFace(vertical,gridType="polar",projection="xy",silent=True) is None


def test_polar_vertices_include_ring_ray_intersections():
    vertices=Grid.OnPlane(width=8,length=8,gridType="polar",output="vertices")
    assert vertices
    unit=[v for v in vertices if abs(math.hypot(*coords(v)[:2])-1)<1e-5]
    assert len(unit)==12
    angles={round(math.degrees(math.atan2(coords(v)[1],coords(v)[0]))%360) for v in unit}
    assert angles==set(range(0,360,30))


def test_clearance_keeps_pattern_origin_fixed():
    host=Face.Rectangle(width=8,length=8)
    vertices=Grid.OnFace(host,gridType="rectangular",spacing=1,origin=[.2,.3,0],clearance=.4,output="vertices")
    assert vertices
    interior=[coords(v) for v in vertices if max(abs(x) for x in coords(v)[:2])<3.5]
    assert interior
    assert all(abs((p[0]-.2)-round(p[0]-.2))<1e-5 and abs((p[1]-.3)-round(p[1]-.3))<1e-5 for p in interior)


def test_xy_origin_projects_vertically_without_changing_xy():
    host=Face.ByVertices([Vertex.ByCoordinates(*p) for p in
                         [(-3,-2,7),(3,-2,13),(3,2,13),(-3,2,7)]])
    vertices=Grid.OnFace(host,gridType="rectangular",projection="xy",origin=[.2,.3,0],output="vertices")
    assert vertices
    interior=[coords(v) for v in vertices if abs(coords(v)[0])<2.9 and abs(coords(v)[1])<1.9]
    assert interior
    assert all(abs((p[0]-.2)-round(p[0]-.2))<1e-5 and abs((p[1]-.3)-round(p[1]-.3))<1e-5 for p in interior)
    assert all(abs(coords(v)[2]-coords(v)[0]-10)<1e-5 for v in vertices)
