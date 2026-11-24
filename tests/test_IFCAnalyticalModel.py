"""Simplified reconstruction contracts and the complete local BIMData corpus."""
import os
from pathlib import Path
import pytest
from shapely.geometry import box, Polygon, shape
from topologicpy.IFC import IFC, IFCFastEntity
from topologicpy.Topology import Topology

SETTINGS = dict(minimum_clear_width=0.6, minimum_clear_height=1.8,
                simplification=0.1, drafting_grid=0.05, tolerance=1e-5, role_overrides={})
DATA = Path(os.environ.get('TOPOLOGICPY_BIMDATA', Path(__file__).parent/'data'/'bimdata'))
FILES = sorted(DATA.rglob('*.ifc'))

def record(i, polygon, bottom=0., top=3., name='Room'):
    return dict(id=i, source=str(i), type='IFCSPACE', footprint=polygon,
                z0=bottom, z1=top, name=name, long_name='', building=1, storey=2)

def test_actual_clearance_not_area_or_bounding_box():
    assert not IFC._Simplified_occupiable(box(0, 0, 100, .3), 3, SETTINGS)
    assert not IFC._Simplified_occupiable(box(0, 0, 10, 10), .5, SETTINGS)
    assert IFC._Simplified_occupiable(box(0, 0, 1, 1), 2, SETTINGS)

def test_hierarchy_requires_storey_under_building_and_accepts_nested_spaces():
    e = {i: IFCFastEntity(i, kind, []) for i,kind in
         [(1,'IFCBUILDING'),(2,'IFCBUILDINGSTOREY'),(3,'IFCSPACE'),(4,'IFCWALL'),(5,'IFCWALL'),(6,'IFCBUILDINGSTOREY'),(7,'IFCWALL')]}
    for i,parent,children in [(10,1,[2,5]),(11,2,[3]),(12,3,[4]),(13,6,[7])]:
        e[i] = IFCFastEntity(i, 'IFCRELAGGREGATES', [None]*4+[('REF',parent), [('REF',c) for c in children]])
    m = IFC._Simplified_membership(e)
    assert m[4] == (1,2) and m[3] == (1,2)
    assert 5 not in m and 7 not in m

@pytest.mark.pythonocc_only
def test_shared_partition_reassigns_sliver_and_excludes_plenum():
    records = [record(1,box(0,0,4,4)), record(2,box(4.02,0,8,4)),
               record(3,box(4,0,4.02,4)), record(4,box(0,0,8,4),3,3.5,'Plenum')]
    result = IFC._Simplified_building(records, {}, SETTINGS)
    assert result.valid and Topology.IsInstance(result.topology,'CellComplex')
    assert len(result.spaces)==2
    assert result.statistics['verified_shared_faces'] >= 1
    assert all(IFC._Simplified_occupiable(shape(r['footprint']),r['top']-r['bottom'],SETTINGS) for r in result.spaces)
    assert {'3','4'} <= set(result.fidelity['excluded_source_spaces'])

@pytest.mark.pythonocc_only
def test_grid_retry_preserves_room_count(monkeypatch):
    original = IFC._Analytical_redraw_sew_cells
    attempts=[]
    def fail_partition(cells, *args, **kwargs):
        attempts.append(len(cells))
        if len(attempts)==1:
            raise ValueError('Injected first assembly failure')
        return original(cells,*args,**kwargs)
    monkeypatch.setattr(IFC,'_Analytical_redraw_sew_cells',fail_partition)
    result=IFC._Simplified_building([record(1,box(0,0,4,4)),record(2,box(4,0,8,4))],{},SETTINGS)
    assert len(result.spaces)==2 and attempts==[2,2]
    assert not result.fidelity['simplified_storeys']
    assert Topology.IsInstance(result.topology,'CellComplex')
    assert result.statistics['geometry_verified']

@pytest.mark.pythonocc_only
def test_assembly_failure_never_collapses_source_rooms(monkeypatch):
    attempts=[]
    def fail_detailed(cells, *args, **kwargs):
        attempts.append(len(cells))
        raise ValueError('Injected assembly failure')
    monkeypatch.setattr(IFC,'_Analytical_redraw_sew_cells',fail_detailed)
    lower=record(1,box(0,0,4,4))
    upper=dict(record(2,box(0,0,4,4),3,6),storey=3)
    with pytest.raises(ValueError,match='exhausted'):
        IFC._Simplified_building([lower,upper],{},SETTINGS)
    assert attempts==[2,2,2,2]

def test_overlapping_area_record_does_not_swallow_rooms():
    rooms,_=IFC._Simplified_layout([record(1,box(0,0,4,4)),record(2,box(4,0,8,4)),
                                    record(3,box(0,0,8,4),name='Gross area')],{},SETTINGS,1)
    assert {r['source_space_id'] for r in rooms}=={'1','2'}

@pytest.mark.pythonocc_only
@pytest.mark.parametrize('path', FILES, ids=lambda p:p.name)
def test_bimdata_all_files(path):
    result=IFC.AnalyticalModel(str(path),coordinateFrame='local')
    if os.environ.get('TOPOLOGICPY_SIMPLIFIED_REPORTS'):
        import json
        destination=Path(os.environ['TOPOLOGICPY_SIMPLIFIED_REPORTS'])
        destination.mkdir(parents=True,exist_ok=True)
        summary=dict(file=str(path),valid=result.valid,statistics=result.statistics,
                     buildings=[dict(id=b.building_id,statistics=b.statistics,fidelity=b.fidelity,
                                     diagnostics=[dict(code=d.code,message=d.message) for d in b.diagnostics])
                                for b in result.building_models])
        (destination/(path.stem+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    assert result.valid, result.report()
    assert result.building_models
    for building in result.building_models:
        assert Topology.IsInstance(building.topology,'CellComplex'), building.report()
        assert building.statistics['geometry_verified']
        assert building.statistics['building_membership']=='strict_ifc_hierarchy'
        assert len(Topology.Cells(building.topology))==len(building.spaces)
        assert building.statistics['minimum_edge_length'] >= .001-1e-5
        assert building.statistics['minimum_face_area'] >= .001**2
        assert all(IFC._Simplified_occupiable(shape(r['footprint']),r['top']-r['bottom'],SETTINGS) for r in building.spaces)
    minimum_rooms={'Medical_Clinic':240,'Munkerud_House':54,'SRP_1045':65}
    if path.stem in minimum_rooms:
        assert result.statistics['rooms'] >= minimum_rooms[path.stem]
        assert all(not b.fidelity.get('simplified_storeys') for b in result.building_models)
    if path.stem=='Munkerud_House':
        from collections import Counter
        assert result.statistics['rooms'] == 54
        assert sorted(Counter(r['storey'] for r in result.spaces).values()) == [18,18,18]
        assert len({r['source_space_id'] for r in result.spaces}) == 54
        assert not any(r.get('inferred') for r in result.spaces)
    if path.stem=='SRP_1433':
        assert result.statistics['analytical_volume'] < 50000

def test_bimdata_corpus_present():
    assert FILES, 'Set TOPOLOGICPY_BIMDATA to the BIMData IFC fixture directory'

@pytest.mark.pythonocc_only
@pytest.mark.parametrize('two_rooms',[False,True])
def test_public_world_coordinates_and_hierarchy_filter(monkeypatch,two_rooms):
    from types import SimpleNamespace
    from topologicpy.Vertex import Vertex
    entities={i:IFCFastEntity(i,kind,[str(i)]) for i,kind in
              [(1,'IFCBUILDING'),(2,'IFCBUILDINGSTOREY'),(3,'IFCSPACE'),(4,'IFCSPACE')]}
    entities[10]=IFCFastEntity(10,'IFCRELAGGREGATES',[None]*4+[('REF',1),[('REF',2)]])
    entities[11]=IFCFastEntity(11,'IFCRELAGGREGATES',[None]*4+[('REF',2),[('REF',3)]])
    evidence=SimpleNamespace(entities=entities,records=[record(3,box(0,0,4,4)),record(4,box(100,100,200,200))],
                             context={},origin=[1000.,2000.,30.])
    if two_rooms:
        entities[5]=IFCFastEntity(5,'IFCSPACE',['5'])
        entities[11].args[5].append(('REF',5))
        evidence.records.append(record(5,box(4,0,8,4)))
    monkeypatch.setattr(IFC,'_Analytical_evidence_read_evidence',lambda *args:evidence)
    result=IFC.AnalyticalModel('mock')
    assert result.valid, result.report()
    assert result.statistics['ignored_nonhierarchical_products']==1
    assert Topology.IsInstance(result.topology,'CellComplex')
    assert len(result.spaces)==(2 if two_rooms else 1)
    coordinates=[Vertex.Coordinates(v) for v in Topology.Vertices(result.topology)]
    assert min(p[0] for p in coordinates)==pytest.approx(1000.)
    assert max(p[0] for p in coordinates)==pytest.approx(1008. if two_rooms else 1004.)
    assert shape(result.spaces[0]['footprint']).bounds==pytest.approx((1000.,2000.,1004.,2004.))
    assert result.spaces[0]['bottom']==pytest.approx(30.)
    assert result.editable_plan['rooms'][0]['bottom']==pytest.approx(0.)
    assert result.editable_plan['origin']==[1000.,2000.,30.]
    signature=IFC._Analytical_persistence_geometry_signature(result.topology)
    assert signature[2]==(1 if two_rooms else 0)


def test_wide_pocket_does_not_legitimise_wall_appendage():
    wall_with_pocket=box(0,0,1,1).union(box(1,.4,30,.6))
    assert not IFC._Simplified_occupiable(wall_with_pocket,3,SETTINGS)


def test_partial_aggregate_leaves_no_wall_room_or_exterior_fragments():
    from shapely.ops import unary_union
    refs=[record(1,box(0,0,4,4)),record(2,box(4.2,0,8.2,4)),
          record(3,box(0,4.2,4,8.2)),record(4,box(-.1,-.1,8.3,8.3),name='Area summary')]
    # Three rooms account for more than 70% of the summary envelope.
    refs[3]['footprint']=unary_union([r['footprint'] for r in refs[:3]]).buffer(.2,join_style=2)
    rooms,excluded=IFC._Simplified_layout(refs,{},SETTINGS,1)
    assert {r['source_space_id'] for r in rooms}=={'1','2','3'}
    assert len(rooms)==3 and '4' in excluded
    assert not unary_union([shape(r['footprint']) for r in rooms]).contains(box(5,5,7,7))
    for room in rooms:
        p=shape(room['footprint']);source=next(r for r in refs if r['source']==room['source_space_id'])
        assert p.intersection(source['footprint']).area > .9*source['footprint'].area
        assert IFC._Simplified_occupiable(p,room['top']-room['bottom'],SETTINGS)


def test_roomless_roof_not_invented_in_building_with_explicit_rooms():
    refs=[record(1,box(0,0,4,4)),dict(record(2,box(0,0,4,4),3,3.3),type='IFCSLAB',storey=3,predefined_type='FLOOR'),
          dict(record(3,box(0,0,.2,4),3,6),type='IFCWALL',storey=3)]
    rooms,_=IFC._Simplified_layout(refs,{'storeys':{3:{'elevation':3}}},SETTINGS,1)
    assert len(rooms)==1 and rooms[0]['source_space_id']=='1'


def test_convex_cleanup_preserves_central_room_and_assigns_thin_extension():
    from shapely.ops import unary_union
    domain=box(0,0,9,3)
    central=box(0,0,3,3).union(box(3,1.4,9,1.6))
    neighbour=domain.difference(central)
    result=IFC._Simplified_convex_reassign([[record(1,central),central],[record(2,neighbour),neighbour]],3,SETTINGS)
    actual=unary_union([p for _,p in result])
    assert actual.symmetric_difference(domain).area < 1e-8
    own=unary_union([p for r,p in result if r['source']=='1'])
    assert own.symmetric_difference(box(0,0,3,3)).area < 1e-8
    assert len(result)==2
    assert not actual.interiors


def test_straight_wall_becomes_shared_centreline_without_empty_strip():
    from shapely.ops import unary_union
    records=[record(1,box(0,0,4,4)),record(2,box(4.2,0,8.2,4)),
             dict(record(3,box(4,0,4.2,4)),type='IFCWALL')]
    rooms,_=IFC._Simplified_layout(records,{},SETTINGS,1)
    polygons=[shape(r['footprint']) for r in rooms]
    assert len(polygons)==2
    assert unary_union(polygons).symmetric_difference(box(0,0,8.2,4)).area < 1e-8
    shared=polygons[0].boundary.intersection(polygons[1].boundary)
    assert shared.length==pytest.approx(4)
    assert shared.bounds[0]==pytest.approx(4.1)
    assert shared.bounds[2]==pytest.approx(4.1)


def test_convex_cleanup_preserves_existing_courtyard():
    from shapely.ops import unary_union
    footprint=box(0,0,10,10).difference(box(3,3,7,7))
    result=IFC._Simplified_convex_reassign([[record(1,footprint),footprint]],3,SETTINGS)
    actual=unary_union([p for _,p in result])
    assert actual.symmetric_difference(footprint).area < 1e-8
    assert len(actual.interiors)==1
