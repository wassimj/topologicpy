"""Component evidence and complete analytical assembly failure recovery."""
from copy import deepcopy
from pathlib import Path
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from topologicpy.IFC import IFC, IFCFastEntity
from topologicpy.Topology import Topology

SOURCE = Path(__file__).with_name('data')/'analytical-three-rooms.ifc'


def test_component_wall_body_restores_parent_evidence():
    model=IFC.Model(str(SOURCE),silent=True)
    entities=deepcopy(model.entities)
    parent=next(e for e in entities.values() if e.type=='IFCWALL')
    original=IFC._Analytical_evidence_read_evidence(model)
    expected=next(r for r in original.records if r['id']==parent.id)
    child_id=max(entities)+1
    args=deepcopy(parent.args);args[0]='component-wall';args[2]='Wall body'
    entities[child_id]=IFCFastEntity(child_id,'IFCBUILDINGELEMENTPART',args)
    parent.args[6]=None
    entities[child_id+1]=IFCFastEntity(child_id+1,'IFCRELAGGREGATES',['parts',None,None,None,('REF',parent.id),[('REF',child_id)]])
    evidence=IFC._Analytical_evidence_read_evidence(IFC.ModelByEntities(entities,silent=True))
    restored=next(r for r in evidence.records if r['id']==parent.id)
    assert restored['type']=='IFCWALL'
    assert restored['method']=='aggregate_children'
    assert restored['component_sources']==['component-wall']
    assert restored['footprint'].area==pytest.approx(expected['footprint'].area)
    assert restored['z1']-restored['z0']==pytest.approx(expected['z1']-expected['z0'])
    assert restored['faces']
    assert not any(m['id']==parent.id for m in evidence.statistics['unavailable_products'])


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('region',[
    Polygon([(0,0),(5,0),(5,1),(1,1),(1,5),(0,5)]),
    box(0,0,6,6).difference(box(1,1,5,5)),
])
def test_triangulation_preserves_concavity_and_holes(region,legacy,monkeypatch):
    if legacy:
        import shapely
        monkeypatch.delattr(shapely,'constrained_delaunay_triangles',raising=False)
    triangles=IFC._Analytical_geometry_triangulate_polygon(region)
    assert triangles
    assert region.symmetric_difference(unary_union(triangles)).area<1e-8
    assert sum(t.area for t in triangles)==pytest.approx(region.area)
    assert all(len(t.exterior.coords)==4 for t in triangles)


def test_failed_exterior_and_shared_assembly_retain_native_rooms(monkeypatch):
    source=IFC.AnalyticalModel(str(SOURCE),coordinateFrame='local',verify=False)
    assert source.spaces
    plan=deepcopy(source.editable_plan)
    if plan.get('buildings'):plan=plan['buildings'][0]['plan']
    plan['exterior_envelope']={'facets':[]}
    def fail_exterior(*args,**kwargs):raise ValueError('Exterior fit failed')
    def fail_shared(*args,**kwargs):raise ValueError('Shared face assembly failed')
    monkeypatch.setattr(IFC,'_Analytical_envelope_shaped_complex',fail_exterior)
    monkeypatch.setattr(IFC,'_Analytical_cellular_shared_complex',fail_shared)
    result=IFC._Analytical_redraw_assemble(plan)
    assert len(result.spaces)==len(plan['rooms'])
    assert len(Topology.Cells(result.topology))==len(plan['rooms'])
    assert {'EXTERIOR_FIT_REQUIRES_REVIEW','SHARED_ASSEMBLY_REQUIRES_REVIEW'} <= {d.code for d in result.diagnostics}


def test_sewing_retains_small_real_room_void():
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    from topologicpy.Core import Core
    first=box(0,0,4,4).difference(box(2,2,2.1,2.1))
    second=box(4,0,8,4)
    cells=[IFC._Analytical_redraw_prism(p,0.,3.,1e-5) for p in [first,second]]
    expected=sum(Cell.Volume(c,mantissa=10) for c in cells)
    sewn=IFC._Analytical_redraw_sew_cells(cells,1e-5)
    result=Topology.Cells(sewn)
    assert len(result)==2
    assert sum(Cell.Volume(c,mantissa=10) for c in result)==pytest.approx(expected,abs=1e-6)
    point=Vertex.ByCoordinates(2.05,2.05,1.5)
    assert all(int(Core.CellUtility.Contains(c,point,1e-7))!=0 for c in result)


@pytest.mark.parametrize('protect_tiny_room',[False,True])
@pytest.mark.pythonocc_only
def test_sewing_filters_only_unowned_microscopic_residuals(monkeypatch,protect_tiny_room):
    from topologicpy.Cell import Cell
    from topologicpy.Vertex import Vertex
    from topologicpy.Dictionary import Dictionary
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    import OCC.Core.BOPAlgo as bop
    room=Cell.Prism(width=2,length=2,height=3)
    tiny=Cell.Prism(origin=Vertex.ByCoordinates(20,20,20),width=.001,length=.001,height=.001,tolerance=1e-7)
    compound=TopoDS_Compound();builder=BRep_Builder();builder.MakeCompound(compound)
    builder.Add(compound,room.shape);builder.Add(compound,tiny.shape)
    class NativeResult:
        def SetArguments(self,*args):pass
        def SetFuzzyValue(self,*args):pass
        def SetRunParallel(self,*args):pass
        def Perform(self):pass
        def HasErrors(self):return False
        def AddAllToResult(self):pass
        def MakeContainers(self):pass
        def Shape(self):return compound
    monkeypatch.setattr(bop,'BOPAlgo_CellsBuilder',NativeResult)
    inputs=[room,tiny] if protect_tiny_room else [room]
    result=IFC._Analytical_redraw_sew_cells(inputs,1e-5)
    assert len(Topology.Cells(result))==len(inputs)
    assert Dictionary.ValueAtKey(Topology.Dictionary(result),'numerical_residual_regions_removed')==(0 if protect_tiny_room else 1)
    assert sum(Cell.Volume(c,mantissa=12) for c in Topology.Cells(result))==pytest.approx(sum(Cell.Volume(c,mantissa=12) for c in inputs),abs=1e-10)


@pytest.mark.pythonocc_only
def test_native_cleanup_removes_empty_wire_without_filling_real_hole():
    from topologicpy.Cell import Cell
    from topologicpy.Face import Face
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Wire
    polygon=box(0,0,4,4).difference(box(2,2,2.1,2.1))
    cell=IFC._Analytical_redraw_prism(polygon,0.,3.,1e-5)
    face=next(f for f in Topology.Faces(cell) if Face.InternalBoundaries(f))
    wire=TopoDS_Wire();builder=BRep_Builder();builder.MakeWire(wire);face.shape.Free(True);builder.Add(face.shape,wire)
    volume=Cell.Volume(cell,mantissa=12)
    cleaned=IFC._Analytical_native_remove_empty_wires(cell)
    assert Cell.Volume(cleaned,mantissa=12)==pytest.approx(volume,abs=1e-8)
    holes=[w for f in Topology.Faces(cleaned) for w in Face.InternalBoundaries(f) or []]
    assert holes
    assert all(Topology.Edges(w) for w in holes)


@pytest.mark.parametrize('bad_datum',[False,True])
def test_inferred_floor_layers_do_not_create_overlapping_storeys(bad_datum):
    import numpy as np
    from shapely.geometry import shape
    records=[]
    for identity,z,storey,kind in [(1,0.,'lower','FLOOR'),(2,3.,'upper','FLOOR'),(3,3.2,'upper','FLOOR'),(4,6.,'roof','ROOF')]:
        polygon=box(0,0,5,4)
        points=np.array([[x,y,height] for height in [z,z+.15] for x,y in list(polygon.exterior.coords)[:-1]])
        records.append(dict(id=identity,source=str(identity),name='floor',type='IFCSLAB',predefined_type=kind,storey=storey,footprint=polygon,z0=z,z1=z+.15,points=points,faces=[],method='projected_faces'))
    evidence=IFC._Analytical_evidence_Evidence(records=records,context={'storeys':{'lower':{'elevation':0.},'upper':{'elevation':3.},'roof':{'elevation':6.}},'children':{},'materials':{}})
    if bad_datum:evidence.context['storeys']['upper']['elevation']=-30.
    plan=IFC._Analytical_redraw_fit_layout(evidence)
    rooms=plan['rooms']
    assert rooms
    assert plan['fidelity']['source_space_coverage_applicable'] is False
    assert plan['fidelity']['source_space_coverage']==1.
    assert 'inferred_reference_coverage' in plan['fidelity']
    IFC._Analytical_redraw_refresh_fidelity(plan)
    assert plan['fidelity']['source_space_coverage']==1.
    assert min(r['bottom'] for r in rooms)>=0
    assert len({(r['bottom'],r['top']) for r in rooms})==2
    assert max(r['top'] for r in rooms)<6.3
    for i,room in enumerate(rooms):
        for other in rooms[i+1:]:
            height=min(room['top'],other['top'])-max(room['bottom'],other['bottom'])
            assert height<=1e-8 or shape(room['footprint']).intersection(shape(other['footprint'])).area<1e-8


def test_coincident_inferred_bands_are_combined_without_discarding_references():
    warnings=[]
    bands=[dict(level=z,bottom=z,members=[{'source':str(i)}],domain=box(i,0,i+1,1),walls=[]) for i,z in enumerate([0.,.2,3.])]
    result=IFC._Analytical_redraw_normalise_inferred_bands(bands,warnings,.4)
    assert len(result)==2
    assert len(result[0]['members'])==2
    assert result[0]['domain'].area==pytest.approx(2.)
    assert warnings[0]['code']=='INFERRED_FLOOR_LAYERS_COMBINED'


@pytest.mark.pythonocc_only
def test_owned_faces_share_nonmatching_floor_and_wall_partitions():
    from shapely.geometry import mapping
    from topologicpy.Cell import Cell
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    rooms=[dict(id='lower',footprint=mapping(box(0,0,4,4)),bottom=0.,top=3.),
           dict(id='upper-left',footprint=mapping(box(0,0,2,4)),bottom=3.,top=6.),
           dict(id='upper-right',footprint=mapping(box(2,0,4,4)),bottom=3.,top=6.)]
    topology=IFC._Analytical_cellular_shared_complex(rooms,1e-5)
    cells=Topology.Cells(topology)
    assert len(cells)==3
    assert all(BRepCheck_Analyzer(c.shape).IsValid() for c in cells)
    assert sum(Cell.Volume(c,mantissa=12) for c in cells)==pytest.approx(96.)
    from collections import Counter
    faces=TopTools_IndexedMapOfShape();incidence=Counter()
    for cell in cells:
        for face in Topology.Faces(cell):incidence[faces.Add(face.shape)]+=1
    assert list(incidence.values()).count(2)==3
    assert max(incidence.values())==2


def test_native_incidence_does_not_merge_nearby_edges_by_coordinate_rounding():
    cell=IFC._Analytical_redraw_prism(box(0,0,1,.000001),0.,1.,1e-7)
    Topology.SetDictionary(cell,{'bim2bem_space_id':'thin-valid-cell'})
    rooms,faces,shared=IFC._Analytical_persistence_geometry_signature(cell,tolerance=1e-5)
    assert rooms=={'thin-valid-cell':pytest.approx(.000001)}
    assert shared==0


def test_point_contact_repair_preserves_neighbours_and_manifold_shells():
    from shapely.geometry import mapping,shape
    from shapely.ops import unary_union
    outside=box(0,0,4,4)
    diamond=Polygon([(2,4),(3,3),(2,2),(1,3)])
    rooms=[dict(id='surrounding',source_space_id='surrounding',footprint=mapping(outside.difference(diamond)),bottom=0.,top=3.),
           dict(id='inner',source_space_id='inner',footprint=mapping(diamond),bottom=0.,top=3.)]
    warnings=[]
    IFC._Analytical_redraw_regularise_point_contacts(rooms,warnings)
    polygons=[shape(r['footprint']) for r in rooms]
    assert polygons[0].intersection(polygons[1]).area<1e-12
    assert unary_union(polygons).symmetric_difference(outside).area<1e-12
    assert polygons[0].exterior.distance(polygons[0].interiors[0])>0
    assert warnings[0]['changed_area_m2']<1e-8
    for room,polygon in zip(rooms,polygons):
        cell=IFC._Analytical_redraw_prism(polygon,0.,3.,1e-5)
        Topology.SetDictionary(cell,{'bim2bem_space_id':room['id']})
        assert IFC._Analytical_persistence_geometry_signature(cell)[0]


def test_persistence_preserves_aperture_host_on_nearby_distinct_faces():
    import numpy as np
    from topologicpy.Face import Face
    from topologicpy.Wire import Wire
    from topologicpy.Vertex import Vertex
    polygon=Polygon([(0,0),(1,0),(1,1),(.500001,1),(.500001,.4),(.5,.4),(.5,1),(0,1)])
    cell=IFC._Analytical_redraw_prism(polygon,0.,1.,1e-7)
    boundaries=[];target=None;other=None
    for index,face in enumerate(Topology.Faces(cell)):
        points=[Vertex.Coordinates(v,mantissa=10) for v in Wire.Vertices(Face.ExternalBoundary(face))]
        record=dict(id=f'boundary-{index}',vertices=points,owners=['room'])
        boundaries.append(record)
        x=np.array(points)[:,0]
        if np.max(np.abs(x-.5))<1e-9:target=record
        if np.max(np.abs(x-.500001))<1e-9:other=record
    assert target and other
    assert IFC._Analytical_geometry_face_key(target['vertices'],1e-5)==IFC._Analytical_geometry_face_key(other['vertices'],1e-5)
    boundaries.remove(target);boundaries.insert(0,target)
    boundaries.remove(other);boundaries.append(other)
    meta=dict(origin=[0.,0.,0.],spaces=[dict(id='room',name='Room',seed=[.2,.2,.5])],boundaries=boundaries,
        openings=[dict(id='aperture',kind='window',boundary_id=target['id'],vertices=[[.5,.55,.2],[.5,.85,.2],[.5,.85,.8],[.5,.55,.8]])])
    restored,count=IFC._Analytical_persistence_restore(cell,meta)
    assert count==1
    hosts=[f for f in Topology.Faces(restored) if Topology.Apertures(f)]
    assert len(hosts)==1
    assert all(abs(Vertex.X(v)-.5)<1e-9 for v in Topology.Vertices(hosts[0]))


def test_real_ifc_space_coverage_keeps_its_authoritative_meaning():
    result=IFC.AnalyticalModel(str(SOURCE),coordinateFrame='local',verify=False)
    assert result.fidelity['source_space_coverage_applicable'] is True
    assert result.fidelity['source_space_coverage']==pytest.approx(1.)
    assert 'inferred_reference_coverage' not in result.fidelity


def test_exterior_ledge_alignment_preserves_rooms_and_internal_partition():
    from shapely.geometry import box, mapping, shape
    rooms=[dict(id='lower-a',bottom=0,top=3,footprint=mapping(box(0,0,2,4))),
           dict(id='lower-b',bottom=0,top=3,footprint=mapping(box(2,0,4,4))),
           dict(id='upper',bottom=3,top=6,footprint=mapping(box(0,0,4.0001,4)))]
    warnings=[]
    IFC._Analytical_redraw_remove_exterior_ledges(rooms,warnings,.02)
    lower=[shape(r['footprint']) for r in rooms[:2]]
    assert len(rooms)==3
    assert lower[0].bounds==(0.,0.,2.,4.)
    assert lower[0].intersection(lower[1]).area==0
    assert lower[0].union(lower[1]).equals(shape(rooms[2]['footprint']))
    assert warnings[-1]['code']=='EXTERIOR_LEDGES_REMOVED'
    assert shape(rooms[2]['footprint']).bounds==(0.,0.,4.,4.)


def test_exterior_ledge_alignment_retains_large_setback_and_shaft():
    from shapely.geometry import box, mapping, shape
    lower=box(0,0,4,4).difference(box(1,1,2,2))
    upper=box(0,0,5,4)
    rooms=[dict(bottom=0,top=3,footprint=mapping(lower)),dict(bottom=3,top=6,footprint=mapping(upper))]
    warnings=[]
    IFC._Analytical_redraw_remove_exterior_ledges(rooms,warnings,.02)
    assert shape(rooms[0]['footprint']).equals(lower)
    assert shape(rooms[1]['footprint']).equals(upper)
    assert warnings==[]


def test_unsafe_ledge_cleanup_rolls_back_without_losing_rooms(monkeypatch):
    def propose(rooms,warnings,tolerance):
        warnings.append(dict(code='EXTERIOR_LEDGES_REMOVED',message='Proposed test correction'))
    def reject(topology):
        raise ValueError('Test closure rejection')
    monkeypatch.setattr(IFC,'_Analytical_redraw_remove_exterior_ledges',staticmethod(propose))
    monkeypatch.setattr(IFC,'_Analytical_persistence_geometry_signature',staticmethod(reject))
    result=IFC.AnalyticalModel(str(SOURCE),coordinateFrame='local',verify=False)
    assert result.topology is not None
    assert len(result.spaces)==3
    assert any(d.code=='EXTERIOR_LEDGE_CLEANUP_REQUIRES_REVIEW' for d in result.diagnostics)
