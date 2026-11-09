"""Native export regression for elevated exterior undersides."""
import pytest
from pathlib import Path
pytest.importorskip('openstudio')
from topologicpy.Cell import Cell
from topologicpy.CellComplex import CellComplex
from topologicpy.Topology import Topology
from topologicpy.EnergyModel import EnergyModel


@pytest.mark.parametrize('offset',[0.,100.])
def test_cantilever_underside_is_exposed_floor_in_any_coordinate_frame(offset):
    lower=Topology.Translate(Cell.Prism(width=4,length=4,height=3,placement='bottom'),0,0,offset)
    upper=Topology.Translate(Cell.Prism(width=6,length=4,height=3,placement='bottom'),1,0,offset+3)
    building=CellComplex.ByCells([lower,upper])
    assets=Path(__file__).resolve().parents[1]/'assets/EnergyModel'
    model=EnergyModel.ByTopology(building,defaultSpaceType=None,glazingRatio=0,
        osModelPath=str(assets/'OSMTemplate-OfficeBuilding-3.10.0.osm'),
        weatherFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.epw'),
        designDayFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.ddy'))
    assert model is not None
    floors=[s for s in model.getSurfaces() if s.surfaceType()=='Floor' and s.outsideBoundaryCondition()=='Outdoors']
    assert sum(s.grossArea() for s in floors)==pytest.approx(8.)
    assert all(all(v.z()==pytest.approx(offset+3) for v in s.vertices()) for s in floors)
    roofs=[s for s in model.getSurfaces() if s.surfaceType()=='RoofCeiling' and s.outsideBoundaryCondition()=='Outdoors']
    assert roofs and all(s.outwardNormal().z()>0 for s in roofs)
    ground=[s for s in model.getSurfaces() if s.outsideBoundaryCondition()=='Ground']
    assert ground and all(s.surfaceType()=='Floor' and s.outwardNormal().z()<0 for s in ground)

def test_curved_end_wall_remains_renderable_after_osm_roundtrip(tmp_path):
    import json
    import openstudio
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    points=json.loads((Path(__file__).with_name('data')/'energy-curved-end-wall.json').read_text())
    face=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points],tolerance=1e-8)
    building=Cell.ByThickenedFace(face,thickness=1,bothSides=False,tolerance=1e-8)
    assets=Path(__file__).resolve().parents[1]/'assets/EnergyModel'
    model=EnergyModel.ByTopology(building,defaultSpaceType=None,glazingRatio=0,
        osModelPath=str(assets/'OSMTemplate-OfficeBuilding-3.10.0.osm'),
        weatherFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.epw'),
        designDayFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.ddy'))
    path=tmp_path/'curved.osm'
    assert EnergyModel.ExportToOSM(model,str(path))
    restored=openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(path))).get()
    end_walls=[]
    for surface in restored.getSurfaces():
        local=openstudio.Transformation.alignFace(surface.vertices()).inverse()*surface.vertices()
        assert openstudio.computeTriangulation(list(local)[::-1],[]),surface.nameString()
        if max(v.x() for v in surface.vertices())-min(v.x() for v in surface.vertices())<1e-6:end_walls.append(surface)
    assert len(end_walls)>=2
    assert sum(s.grossArea() for s in end_walls)==pytest.approx(2*20.93712692,abs=1e-5)

@pytest.mark.parametrize('direction',[-1,1])
def test_inward_aperture_keeps_host_wall_in_openstudio_scene(tmp_path,direction):
    import openstudio
    from openstudio.openstudiomodelgenerators import ThreeJSForwardTranslator
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    from topologicpy.Context import Context
    from topologicpy.Aperture import Aperture
    cell=Cell.Prism(width=4,length=4,height=3,placement='bottom')
    host=next(f for f in Topology.Faces(cell) if Face.Normal(f)[0]*direction>.9)
    x,y,z=Vertex.Coordinates(Topology.Centroid(host))
    points=[[x,y-.5,z-.75],[x,y+.5,z-.75],[x,y+.5,z+.75],[x,y-.5,z+.75]]
    face=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points])
    if sum(a*b for a,b in zip(Face.Normal(face),Face.Normal(host)))>0:
        face=Face.ByVertices([Vertex.ByCoordinates(*p) for p in reversed(points)])
    assert sum(a*b for a,b in zip(Face.Normal(face),Face.Normal(host)))<0
    assert Aperture.ByTopologyContext(face,Context.ByTopologyParameters(host,.5,.5,.5)) is not None
    assets=Path(__file__).resolve().parents[1]/'assets/EnergyModel'
    model=EnergyModel.ByTopology(cell,defaultSpaceType=None,glazingRatio=0,
        osModelPath=str(assets/'OSMTemplate-OfficeBuilding-3.10.0.osm'),
        weatherFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.epw'),
        designDayFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.ddy'))
    assert len(model.getSubSurfaces())==1
    assert all(a.outwardNormal().dot(s.outwardNormal())>.99 for s in model.getSurfaces() for a in s.subSurfaces())
    path=tmp_path/'aperture.osm';assert EnergyModel.ExportToOSM(model,str(path))
    restored=openstudio.osversion.VersionTranslator().loadModel(openstudio.path(str(path))).get()
    translator=ThreeJSForwardTranslator();translator.modelToThreeJS(restored,True)
    assert not translator.errors(),[e.logMessage() for e in translator.errors()]

@pytest.mark.parametrize('disabled',[False,True])
def test_vertical_seam_regularisation_is_bounded_and_keeps_xy(disabled):
    from topologicpy.EnergyModel import EnergyModel
    points=[(0,0,10),(0,0,10.0008),(0,0,10.0016),(1,0,10.0008)]
    mapping,record=EnergyModel._OSMVerticalSeams(points,0 if disabled else .001)
    if disabled:
        assert not mapping and record['maximum_displacement_m']==0
    else:
        assert mapping[(0,0,10)]==pytest.approx(10.0004)
        assert mapping[(0,0,10.0008)]==pytest.approx(10.0004)
        assert (0,0,10.0016) not in mapping
        assert (1,0,10.0008) not in mapping
        assert record['maximum_displacement_m']<=.0005


def test_regularised_curved_end_walls_are_single_polygons_and_space_is_closed():
    import json,openstudio
    from topologicpy.Face import Face
    from topologicpy.Vertex import Vertex
    from openstudio.openstudiomodelgenerators import ThreeJSForwardTranslator
    points=json.loads((Path(__file__).with_name('data')/'energy-curved-end-wall.json').read_text())
    face=Face.ByVertices([Vertex.ByCoordinates(*p) for p in points],tolerance=1e-8)
    cell=Cell.ByThickenedFace(face,thickness=1,bothSides=False,tolerance=1e-8)
    assets=Path(__file__).resolve().parents[1]/'assets/EnergyModel'
    model=EnergyModel.ByTopology(cell,defaultSpaceType=None,glazingRatio=0,seamTolerance=.001,
        osModelPath=str(assets/'OSMTemplate-OfficeBuilding-3.10.0.osm'),
        weatherFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.epw'),
        designDayFilePath=str(assets/'GBR_London.Gatwick.037760_IWEC.ddy'))
    caps=[s for s in model.getSurfaces() if max(v.x() for v in s.vertices())-min(v.x() for v in s.vertices())<1e-6]
    assert len(caps)==2 and all(s.grossArea()>20 for s in caps)
    assert all(s.isEnclosedVolume() for s in model.getSpaces())
    translator=ThreeJSForwardTranslator();translator.modelToThreeJS(model,True)
    assert not translator.errors()
    record=json.loads(model.getBuilding().additionalProperties().getFeatureAsString('topologicpy_export_regularisation').get())
    assert 0<record['maximum_displacement_m']<=.0005
