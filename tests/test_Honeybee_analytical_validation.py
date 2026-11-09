"""Native HBJSON validity, precision and aperture-preservation regressions."""
import importlib.util,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(any(importlib.util.find_spec(n) is None for n in ('honeybee','honeybee_energy','honeybee_radiance')),reason='Optional Honeybee packages are not installed')

def test_boundary_touching_aperture_moves_without_losing_glazing_area():
    from ladybug_geometry.geometry3d.face import Face3D
    from ladybug_geometry.geometry3d.pointvector import Point3D
    from topologicpy.Honeybee import Honeybee
    host=Face3D(tuple(Point3D(*p) for p in [(0,0,0),(4,0,0),(4,3,0),(0,3,0)]))
    opening=Face3D(tuple(Point3D(*p) for p in [(0,1,0),(1,1,0),(1,2,0),(0,2,0)]))
    fitted,record=Honeybee._FitSubFaceGeometry(opening,host,1e-5)
    assert host.is_sub_face(fitted,1e-5,.0175)
    assert fitted.area==pytest.approx(opening.area,abs=1e-12)
    assert record['distance_m']==pytest.approx(2e-5)
    assert record['method']=='area_preserving_translation'

def test_roof_microsteps_survive_hbjson_save_load(tmp_path):
    from honeybee.model import Model
    from topologicpy.Honeybee import Honeybee
    from topologicpy.Cell import Cell
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    points=json.loads((Path(__file__).with_name('data')/'energy-curved-end-wall.json').read_text())
    face=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points],tolerance=1e-8)
    cell=Cell.ByThickenedFace(face,thickness=1,bothSides=False,tolerance=1e-8)
    exported=Honeybee.ModelByTopology(cell,addSensorGrid=False)
    path=tmp_path/'roof.hbjson';assert Honeybee.ExportToHBJSON(exported,str(path))
    restored=Model.from_hbjson(str(path))
    assert restored.tolerance==pytest.approx(1e-5)
    assert restored.rooms[0].volume==pytest.approx(Cell.Volume(cell),rel=1e-5)
    assert not restored.check_all(raise_exception=False,detailed=True)

def test_boundary_touching_native_aperture_records_its_adjustment():
    from topologicpy.Honeybee import Honeybee
    from topologicpy.Cell import Cell
    from topologicpy.Topology import Topology
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    from topologicpy.Context import Context
    from topologicpy.Aperture import Aperture
    cell=Cell.Prism(width=4,length=4,height=3,placement='bottom')
    host=next(f for f in Topology.Faces(cell) if Face.Normal(f)[0]>.9)
    x,y,z=Vertex.Coordinates(Topology.Centroid(host))
    points=[(x,y-2,z-.5),(x,y-1,z-.5),(x,y-1,z+.5),(x,y-2,z+.5)]
    opening=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points])
    assert Aperture.ByTopologyContext(opening,Context.ByTopologyParameters(host,.5,.5,.5)) is not None
    model=Honeybee.ModelByTopology(cell,addSensorGrid=False)
    apertures=[a for r in model.rooms for f in r.faces for a in f.apertures]
    assert len(apertures)==1 and apertures[0].area==pytest.approx(1.)
    assert apertures[0].user_data['topologicpy_geometry_correction']['distance_m']<=.001
    assert not model.check_all(raise_exception=False,detailed=True)
